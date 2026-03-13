import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ModelArgs:
    dim: int = 768
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps 防止除以 0 情况
        self.eps = eps
        # weight 是一个可学习的参数，全部初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 RMSNorm 的核心部分
        # x.pow(2).mean(-1, keepdim=True) 计算了输入 x 的平方的均值
        # torch.rsqrt 是平方根的倒数，这样就得到 RMSNorm 的分母部分，再加上 eps 防止分母为 0
        # 最后乘以 x，得到 RMSNorm 的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 函数是模型的前向传播
        # 首先将输入的 x 转为 float 类型，然后进行 RMSNorm，最后再转回原来的数据类型
        # 最后乘以 weight，这是 RMSNorm 的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# # 测试
# if __name__ == "__main__":
#     args = ModelArgs()
#     norm = RMSNorm(args.dim, args.norm_eps)
#     x = torch.randn(1, 50, args.dim)
#     output = norm(x)
#     print("output shape:", output.shape)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape

    # 如果重复次数为 1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x

    # 对张量进行拓展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :]  # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 将新添加的维度拓展到 n_rep 大小，实现重复的效果
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重新塑形，合并键值对头的数量和重复次数的维度
    )


# 旋转嵌入
# 注意：此时的 dim 为 dim//n_head，因为我们是对每个 head 进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float() 生成了一个从 0 开始，步长为 2 的序列，长度为 dim 的一半
    # 然后每个元素除以 dim，再取 theta 的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从 0 到 end 的序列，长度为 end
    t = torch.arange(end, device=freqs.device).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs * t[:, None])
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs * t[:, None])
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取 x 的维度数
    ndim = x.ndim

    # 断言，确保 1 在 x 的维度范围内
    assert 0 <= 1 < ndim

    # 断言，确保 freqs_cis 的形状与 x 的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 构造一个新的形状，除了第二维和最后一维，其他维度都为 1，这样做是为了能够将 freqs_cis 与 x 进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    # 将 freqs_cis 调整为新的形状，并返回
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 根据是否指定 n_kv_heads，确定用于 key 和 value 的头的数量
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键值头数整除
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为 1
        model_parallel_size = 1
        # 本地键值头数，等于总数除以模型并行处理大小
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行大小
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于拓展键和值的尺寸
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存 dropout 的概率
        self.dropout = args.dropout

        # 检查是否使用 Flash Attention（需要 PyTorch >= 2.0）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持 Flash Attention，则使用手动实现的注意力机制，并设置 mask
            print("WARNING: using slow attention. Flash Attention requires PyTorch >=2.0")

            # 创建一个上三角矩阵，用于遮蔽未来信息
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型缓存区
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # 计算查询 Q，键 K，值 V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入（RoPE）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 对键值进行拓展以适应重复次数
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持 Flash Attention，选择实现方式
        if self.flash:
            # 使用 Flash Attention
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # 使用手动实现注意力机制
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的 4 倍
        # 然后将其减少到 2/3，最后确保它是 multiple_of 的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第一层线性变换，从输入维度到隐藏层维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三次线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义 dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入 x 通过第一层线性变换和 SiLU 激活函数
        # 然后，结果乘以输入 x 通过第三层变换的结果
        # 最后，通过第二层线性变换和 dropout
        return self.dropout(self.w2(F.silu(self.w1(x) * self.w3(x))))


class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads
        # 定义每个头的维度，等于每个维度除以头数
        self.head_dim = args.dim // args.n_heads
        # 定义 Attention 对象，用于进行注意力计算
        self.attention = Attention(args)
        # 定义 MLP 对象，用于进行前馈神经网络计算
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层的 ID
        self.layer_id = layer_id
        # 定义注意力计算的层归一化
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的层归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 前向传播函数
        # 首先，输入 x 经过注意力层归一化层，然后进行注意力计算，结果与输入 x 相加得到 h
        # 然后，h 经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与 h 相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 初始化模型参数
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers
        self.args = args

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout 层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder 层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 层归一化
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None

    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 前向传播函数
        _bsz, seqlen = tokens.shape
        # 通过词嵌入层和 Dropout 层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过 Decoder 层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # 通过层归一化
        h = self.norm(h)
        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
                reduction='none'
            )
        else:
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits, self.last_loss

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz, seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键 k/v cache。
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            # 前向传播获取序列中最后一个位置的 logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 只保留最后一个时间步的输出

            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:]  # 只返回生成的 token
