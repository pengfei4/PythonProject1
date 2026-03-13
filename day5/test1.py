import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 配置
# -------------------------
@dataclass
class ModelArgs:
    vocab_size: int
    block_size: int
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 4
    dropout: float = 0.1


# -------------------------
# 位置编码（learned）
# 接口：pos_enc(x) -> x + pos_embed
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.block_size = args.block_size
        self.wpe = nn.Embedding(args.block_size, args.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, t, c)
        b, t, c = x.size()
        pos = torch.arange(0, t, device=x.device, dtype=torch.long)  # (t,)
        pos_emb = self.wpe(pos)[None, :, :]  # (1, t, c)
        return x + pos_emb


# -------------------------
# 多头注意力（自实现，支持 causal mask + 外部 mask）
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q,k,v: (b, t, c)
        attn_mask: (tq, tk) 或 (b, 1, tq, tk)；mask 位置为 True 表示要屏蔽
        causal: True 时自动加上下三角因果 mask（仅用于 self-attn）
        """
        b, tq, c = q.size()
        _, tk, _ = k.size()

        q = self.q_proj(q).view(b, tq, self.n_head, self.head_dim).transpose(1, 2)  # (b, h, tq, d)
        k = self.k_proj(k).view(b, tk, self.n_head, self.head_dim).transpose(1, 2)  # (b, h, tk, d)
        v = self.v_proj(v).view(b, tk, self.n_head, self.head_dim).transpose(1, 2)  # (b, h, tk, d)

        # attention scores: (b, h, tq, tk)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask (tq == tk 时常用)
        if causal:
            # 只允许看见当前位置及之前：下三角为1，上三角为0
            causal_mask = torch.triu(
                torch.ones(tq, tk, device=att.device, dtype=torch.bool),
                diagonal=1,
            )  # (tq, tk) True 表示要屏蔽
            att = att.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        # external mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                att = att.masked_fill(attn_mask[None, None, :, :], float("-inf"))
            else:
                att = att.masked_fill(attn_mask, float("-inf"))

        w = F.softmax(att, dim=-1)
        w = self.attn_drop(w)

        y = w @ v  # (b, h, tq, d)
        y = y.transpose(1, 2).contiguous().view(b, tq, c)  # (b, tq, c)

        y = self.out_proj(y)
        y = self.resid_drop(y)
        return y


# -------------------------
# FFN
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Encoder Layer
# -------------------------
class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.attn = MultiHeadAttention(args.n_embd, args.n_head, args.dropout)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.ff = FeedForward(args.n_embd, args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attn（encoder 无 causal）
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), causal=False)
        x = x + self.ff(self.ln2(x))
        return x


# -------------------------
# Decoder Layer
# -------------------------
class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.self_attn = MultiHeadAttention(args.n_embd, args.n_head, args.dropout)

        self.ln2 = nn.LayerNorm(args.n_embd)
        self.cross_attn = MultiHeadAttention(args.n_embd, args.n_head, args.dropout)

        self.ln3 = nn.LayerNorm(args.n_embd)
        self.ff = FeedForward(args.n_embd, args.dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        # masked self-attn（decoder 需要 causal）
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), causal=True)

        # cross-attn（Q 来自 decoder，K/V 来自 encoder）
        x = x + self.cross_attn(self.ln2(x), enc_out, enc_out, causal=False)

        x = x + self.ff(self.ln3(x))
        return x


# -------------------------
# Encoder / Decoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.ln_f(x)


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.ln_f(x)


# -------------------------
# Transformer（整体）
# -------------------------
class Transformer(nn.Module):
    """整体模型（toy版：encoder/decoder 输入同一个序列，便于跑通）"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.block_size is not None

        self.args = args
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(args.vocab_size, args.n_embd),
                wpe=PositionalEncoding(args),
                drop=nn.Dropout(args.dropout),
                encoder=Encoder(args),
                decoder=Decoder(args),
            )
        )
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (b, t) token ids
        targets: (b, t) token ids or None
        """
        b, t = idx.size()
        assert t <= self.args.block_size, (
            f"序列长度 {t} 超过 block_size {self.args.block_size}"
        )

        # token embedding: (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)

        # positional encoding + dropout
        x = self.transformer.wpe(tok_emb)
        x = self.transformer.drop(x)

        # encoder
        enc_out = self.transformer.encoder(x)

        # decoder（toy版：输入也用 x）
        dec_out = self.transformer.decoder(x, enc_out)

        if targets is not None:
            logits = self.lm_head(dec_out)  # (b, t, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(dec_out[:, [-1], :])  # (b, 1, vocab)
            loss = None

        return logits, loss


# -------------------------
# 运行一个最小例子
# -------------------------
def main():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = ModelArgs(
        vocab_size=5000,
        block_size=64,
        n_embd=256,
        n_head=8,
        n_layer=4,
        dropout=0.1,
    )
    model = Transformer(args).to(device)
    model.train()

    # 假数据：batch=4，seq_len=32
    b, t = 4, 32
    idx = torch.randint(0, args.vocab_size, (b, t), device=device)

    # 语言模型常见做法：targets 是 idx 的右移版本（这里简单演示）
    targets = idx.clone()
    targets[:, :-1] = idx[:, 1:]
    targets[:, -1] = -1  # 最后一个位置不监督

    logits, loss = model(idx, targets)
    print("logits shape:", logits.shape)  # (b, t, vocab_size)
    print("loss:", loss.item())

    # 反向传播跑通
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    print("one training step done.")

    # 推理：只拿最后一步 logits
    model.eval()
    with torch.no_grad():
        logits2, loss2 = model(idx, targets=None)
    print("inference logits shape:", logits2.shape)  # (b, 1, vocab_size)
    print("inference loss:", loss2)


if __name__ == "__main__":
    main()

"""
如果你要做真正的 seq2seq（src->tgt），建议把 forward 改成：
forward(self, src_idx, tgt_idx, targets=None)
- encoder 输入 src_idx
- decoder 输入 tgt_idx（通常是目标序列右移后的输入）
- targets 是目标序列（不右移）用于计算 loss
"""

# transformer_seq2seq_toy.py
# 一个完整、可直接运行的 PyTorch Transformer（Encoder-Decoder）示例
# 说明：
# - forward(idx, targets=None) 为了匹配你贴的代码：encoder 和 decoder 都吃同一个序列（toy 版本，便于跑通）
# - 你如果做真正的翻译/摘要（src->tgt），需要把 forward 改成接收 src_idx 和 tgt_idx（我在文件末尾给了注释示例）

