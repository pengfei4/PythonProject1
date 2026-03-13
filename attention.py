import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass  # 如果你自己要定义 ModelArgs

device = torch.device("cpu")

@dataclass
class ModelArgs:
    dim: int
    n_heads: int
    n_embd: int
    dropout: float
    max_seq_len: int

class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x dim
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x dim（head_dim = dim / n_heads）
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, dim) -> (B, T, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, dim // n_head)，然后交换维度，变成 (B, n_head, T, dim // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, dim // n_head)，再拼接成 (B, T, n_head * dim // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

def tensor_info(name, t: torch.Tensor, max_show=3):
    t_detached = t.detach()
    print(f"\n[{name}]")
    print(f"  shape={tuple(t_detached.shape)}, dtype={t_detached.dtype}, device={t_detached.device}")
    if t_detached.numel() > 0 and torch.is_floating_point(t_detached):
        print(f"  min={t_detached.min().item():.6f}, max={t_detached.max().item():.6f}, mean={t_detached.mean().item():.6f}")
    # 展示少量元素（避免输出爆炸）
    flat = t_detached.flatten()
    show_n = min(flat.numel(), max_show)
    if show_n > 0:
        print(f"  first {show_n} elems: {flat[:show_n].cpu().numpy()}")

@torch.no_grad()
def run_mha_step_by_step(mha: MultiHeadAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        show_b=0, show_head=0, plot=True):
    """
    逐步复现 mha.forward 的计算并打印中间张量信息，同时可视化某个 batch/head 的 attention 权重。
    """
    mha.eval()  # 关闭 dropout 的随机性（如果你想看 dropout 效果就注释掉）
    assert q.shape == k.shape == v.shape, "q/k/v 形状必须一致（这里按自注意力写的）"
    assert q.shape[-1] == mha.wq.in_features, f"q最后一维应等于 args.n_embd={mha.wq.in_features}"

    bsz, seqlen, _ = q.shape
    print(f"\n===== Step-by-step MHA (B={bsz}, T={seqlen}, n_heads={mha.n_heads}, head_dim={mha.head_dim}) =====")

    # Step 0: 输入
    tensor_info("input q", q)
    tensor_info("input k", k)
    tensor_info("input v", v)

    # Step 1: 线性映射 -> xq/xk/xv: (B,T, n_heads*head_dim)
    xq = mha.wq(q)
    xk = mha.wk(k)
    xv = mha.wv(v)
    tensor_info("xq = wq(q)", xq)
    tensor_info("xk = wk(k)", xk)
    tensor_info("xv = wv(v)", xv)

    # Step 2: view + transpose -> (B, n_heads, T, head_dim)
    xq = xq.view(bsz, seqlen, mha.n_heads, mha.head_dim).transpose(1, 2)
    xk = xk.view(bsz, seqlen, mha.n_heads, mha.head_dim).transpose(1, 2)
    xv = xv.view(bsz, seqlen, mha.n_heads, mha.head_dim).transpose(1, 2)
    tensor_info("xq heads", xq)
    tensor_info("xk heads", xk)
    tensor_info("xv heads", xv)

    # Step 3: raw attention scores = QK^T / sqrt(dk) -> (B, nh, T, T)
    scores_raw = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(mha.head_dim)
    tensor_info("scores_raw (before mask)", scores_raw)

    # Step 4: causal mask（如果有）
    if mha.is_causal:
        assert hasattr(mha, "mask"), "is_causal=True 但没找到 mask"
        scores_masked = scores_raw + mha.mask[:, :, :seqlen, :seqlen]
        tensor_info("scores_masked (after mask)", scores_masked)
    else:
        scores_masked = scores_raw

    # Step 5: softmax -> attention weights
    attn = F.softmax(scores_masked.float(), dim=-1).type_as(xq)
    tensor_info("attn = softmax(scores)", attn)

    # Step 6: dropout（eval 模式下 dropout 不生效）
    attn_drop = mha.attn_dropout(attn)
    tensor_info("attn after dropout", attn_drop)

    # Step 7: attention output = attn @ V -> (B, nh, T, head_dim)
    out_heads = torch.matmul(attn_drop, xv)
    tensor_info("out_heads = attn @ V", out_heads)

    # Step 8: transpose + reshape -> (B, T, nh*head_dim)
    out_merge = out_heads.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    tensor_info("out_merge (concat heads)", out_merge)

    # Step 9: output projection -> (B, T, dim)
    out = mha.wo(out_merge)
    out = mha.resid_dropout(out)
    tensor_info("final out", out)

    # 可视化：选择一个 batch、一个 head 的 attention 矩阵 (T,T)
    if plot:
        b = int(show_b)
        h = int(show_head)
        assert 0 <= b < bsz
        assert 0 <= h < mha.n_heads

        # 画 raw scores（mask 后）或 raw（mask 前）都行，这里画 mask 后更直观
        mat_scores = scores_masked[b, h].detach().cpu().float().numpy()
        plt.figure()
        plt.imshow(mat_scores, aspect="auto")
        plt.colorbar()
        plt.title(f"Attention Scores (masked) - batch {b}, head {h}")
        plt.xlabel("Key position")
        plt.ylabel("Query position")
        plt.show()

        mat_attn = attn[b, h].detach().cpu().float().numpy()
        plt.figure()
        plt.imshow(mat_attn, aspect="auto")
        plt.colorbar()
        plt.title(f"Attention Weights (softmax) - batch {b}, head {h}")
        plt.xlabel("Key position")
        plt.ylabel("Query position")
        plt.show()

    return out, {
        "xq": xq, "xk": xk, "xv": xv,
        "scores_raw": scores_raw,
        "scores_masked": scores_masked,
        "attn": attn,
        "out_heads": out_heads,
        "out_merge": out_merge,
        "out": out
    }


# ====== 一个最小可运行例子（你可以按你自己的参数改）======
if __name__ == "__main__":
    torch.manual_seed(0)

    args = ModelArgs(
        dim=8,
        n_heads=2,
        n_embd=8,      # 注意：q/k/v 的最后一维要等于 n_embd
        dropout=0.0,   # 建议调试先设 0，方便复现
        max_seq_len=16
    )

    mha = MultiHeadAttention(args, is_causal=True)

    B, T, C = 1, 6, args.n_embd
    q = torch.randn(B, T, C)
    k = q.clone()
    v = q.clone()

    out, cache = run_mha_step_by_step(mha, q, k, v, show_b=0, show_head=0, plot=True)