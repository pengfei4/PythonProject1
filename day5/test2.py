import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 配置
# ----------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 5000
    block_size: int = 128   # 最大序列长度 T
    n_embd: int = 256       # 模型宽度 C
    n_head: int = 8         # 注意力头数 H
    n_layer: int = 4        # 堆叠层数
    dropout: float = 0.1
    use_sin_pos: bool = False  # True: sinusoidal; False: learned pos embedding


# ----------------------------
# Token Embedding
# ----------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) -> (B, T, C)
        return self.wte(idx)


# ----------------------------
# Positional Encoding（两种：learned / sinusoidal）
# 接口：pos(x) -> x + pos
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, block_size: int, n_embd: int, use_sin_pos: bool = False):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.use_sin_pos = use_sin_pos

        if not use_sin_pos:
            self.wpe = nn.Embedding(block_size, n_embd)
        else:
            self.register_buffer("sin_pos_table", self._build_sin_table(block_size, n_embd), persistent=False)

    @staticmethod
    def _build_sin_table(T: int, C: int) -> torch.Tensor:
        # (T, C)
        pos = torch.arange(T).unsqueeze(1)  # (T,1)
        div = torch.exp(torch.arange(0, C, 2) * (-math.log(10000.0) / C))  # (C/2,)
        table = torch.zeros(T, C)
        table[:, 0::2] = torch.sin(pos * div)
        table[:, 1::2] = torch.cos(pos * div)
        return table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        assert T <= self.block_size, f"T={T} > block_size={self.block_size}"
        if self.use_sin_pos:
            pos = self.sin_pos_table[:T, :].unsqueeze(0)  # (1, T, C)
        else:
            positions = torch.arange(T, device=x.device, dtype=torch.long)  # (T,)
            pos = self.wpe(positions).unsqueeze(0)  # (1, T, C)
        return x + pos


# ----------------------------
# Multi-Head Attention（causal）
# Q,K,V 来自同一输入 x
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # 一次性做 qkv 投影更清晰也更常用：Linear(C -> 3C)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        return: (B, T, C)
        """
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)  # each (B, T, C)

        # reshape to multi-head: (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # attention scores: (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask: 禁止看未来（上三角屏蔽）
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        w = F.softmax(att, dim=-1)
        w = self.attn_drop(w)

        y = w @ v  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        y = self.proj(y)
        y = self.resid_drop(y)
        return y


# ----------------------------
# FeedForward Network
# ----------------------------
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


# ----------------------------
# Transformer Block：Pre-LN + Residual
# x = x + MHA(LN(x))
# x = x + FFN(LN(x))
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # Residual 1
        x = x + self.ffn(self.ln2(x))   # Residual 2
        return x


# ----------------------------
# Transformer Language Model（Decoder-only）
# Embedding + PosEnc + N blocks + Final LN + LM head
# ----------------------------
class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.block_size is not None

        self.cfg = cfg
        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.n_embd)
        self.pos_enc = PositionalEncoding(cfg.block_size, cfg.n_embd, cfg.use_sin_pos)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.n_embd, cfg.n_head, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])

        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        print(f"number of parameters: {self.num_params()/1e6:.2f}M")

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (B, T)
        targets: (B, T) 或 None
        return:
            logits: (B, T, vocab)
            loss: Tensor 标量 或 None
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"T={T} > block_size={self.cfg.block_size}"

        x = self.tok_emb(idx)      # (B, T, C)
        x = self.pos_enc(x)        # (B, T, C)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)             # (B, T, C)

        x = self.ln_f(x)           # (B, T, C)
        logits = self.lm_head(x)   # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss


# ----------------------------
# 最小可运行测试：给 batch 输入输出 logits
# ----------------------------
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ModelConfig(
        vocab_size=8000,
        block_size=64,
        n_embd=256,
        n_head=8,
        n_layer=4,
        dropout=0.1,
        use_sin_pos=False,
    )

    model = TransformerLM(cfg).to(device)
    model.train()

    B, T = 4, 32
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # LM 常见：targets = idx 的右移；最后一位不监督设为 -1
    targets = idx.clone()
    targets[:, :-1] = idx[:, 1:]
    targets[:, -1] = -1

    logits, loss = model(idx, targets)
    print("idx:", idx.shape)          # (B, T)
    print("logits:", logits.shape)    # (B, T, vocab)
    print("loss:", loss.item())       # 用 .item() 避免 warning

    # 跑一个训练 step
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print("one step ok.")

    # 推理：不传 targets 也能输出 logits
    model.eval()
    with torch.no_grad():
        logits2, loss2 = model(idx)
    print("inference logits:", logits2.shape, "loss:", loss2)


if __name__ == "__main__":
    main()


# transformer_lm.py
# 一个“拼装式”的 Transformer Language Model（Decoder-only, causal）
# 模块包含：Embedding / PosEnc / MHA / FFN / LayerNorm / Residual
# 输入：idx (B, T) token ids
# 输出：logits (B, T, vocab_size)

