"""Shared transformer harness (PROTOCOL_v3.md section 8).

Pre-norm, RoPE, depth 8, head_dim 64, d_ff = 4*d_model. The six constrained
maps (Q, K, V, O, mlp_in, mlp_out) are built by the arm factory; everything else
(norms, embeddings, head) is shared and identical in structure across arms.
Embeddings are tied with the output head and excluded from the constrained
budget. One harness serves all four arms — arms differ only inside the
constrained-linear modules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arms.base import ConstrainedLinear, build_constrained
from .rope import apply_rope, build_rope_cache


@dataclass
class ModelArgs:
    arm: str
    d_model: int
    vocab_size: int = 50257
    depth: int = 8
    head_dim: int = 64
    d_ff_mult: int = 4
    seq_len: int = 1024
    twist: bool | None = None
    init_std: float = 0.02

    @property
    def n_heads(self) -> int:
        # Protocol v3.1 (owner-authorized amendment): attention inner dim is
        # decoupled from d_model. head_dim stays 64; n_heads = round(d_model/64)
        # with HALF ROUNDING UP (floor(x + 0.5), not banker's rounding), so the
        # attention inner dim is n_heads*64 (>= 64), independent of the
        # multiples-of-8 d_model. See docs/PROTOCOL_v3.1_amendment.md.
        return max(1, math.floor(self.d_model / self.head_dim + 0.5))

    @property
    def attn_inner(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def d_ff(self) -> int:
        return self.d_ff_mult * self.d_model


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, gen: torch.Generator | None):
        super().__init__()
        self.args = args
        d, inner = args.d_model, args.attn_inner
        resid_std = args.init_std * (2 * args.depth) ** -0.5
        # v3.1: Q/K/V are d_model -> inner; O is inner -> d_model.
        self.wq = build_constrained(args.arm, d, inner, generator=gen, twist=args.twist,
                                    init_std=args.init_std, role="wq")
        self.wk = build_constrained(args.arm, d, inner, generator=gen, twist=args.twist,
                                    init_std=args.init_std, role="wk")
        self.wv = build_constrained(args.arm, d, inner, generator=gen, twist=args.twist,
                                    init_std=args.init_std, role="wv")
        self.wo = build_constrained(args.arm, inner, d, generator=gen, twist=args.twist,
                                    init_std=resid_std, role="wo")

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H, hd = self.args.n_heads, self.args.head_dim
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, H * hd)
        return self.wo(o)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, gen: torch.Generator | None):
        super().__init__()
        resid_std = args.init_std * (2 * args.depth) ** -0.5
        self.w_in = build_constrained(args.arm, args.d_model, args.d_ff, generator=gen,
                                      twist=args.twist, init_std=args.init_std, role="mlp_in")
        self.w_out = build_constrained(args.arm, args.d_ff, args.d_model, generator=gen,
                                       twist=args.twist, init_std=resid_std, role="mlp_out")

    def forward(self, x):
        return self.w_out(F.gelu(self.w_in(x)))


class Block(nn.Module):
    def __init__(self, args: ModelArgs, gen: torch.Generator | None):
        super().__init__()
        self.norm1 = RMSNorm(args.d_model)
        self.norm2 = RMSNorm(args.d_model)
        self.attn = Attention(args, gen)
        self.mlp = MLP(args, gen)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, gen: torch.Generator | None = None):
        super().__init__()
        self.args = args
        # Embeddings tied with head, fp16 for byte accounting (section 8); trained
        # under the same fp32-master / bf16-autocast regime as everything else.
        self.embed = nn.Embedding(args.vocab_size, args.d_model)
        nn.init.normal_(self.embed.weight, 0.0, args.init_std)
        self.blocks = nn.ModuleList([Block(args, gen) for _ in range(args.depth)])
        self.norm_f = RMSNorm(args.d_model)
        cos, sin = build_rope_cache(args.seq_len, args.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = F.linear(x, self.embed.weight)  # tied head
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # --- budget / packer surface ---------------------------------------
    def constrained_modules(self) -> dict[str, ConstrainedLinear]:
        out: dict[str, ConstrainedLinear] = {}
        for i, blk in enumerate(self.blocks):
            out[f"blk{i}.wq"] = blk.attn.wq
            out[f"blk{i}.wk"] = blk.attn.wk
            out[f"blk{i}.wv"] = blk.attn.wv
            out[f"blk{i}.wo"] = blk.attn.wo
            out[f"blk{i}.mlp_in"] = blk.mlp.w_in
            out[f"blk{i}.mlp_out"] = blk.mlp.w_out
        return out

    def ideal_constrained_payload_bits(self) -> float:
        return sum(m.ideal_payload_bits() for m in self.constrained_modules().values())

    def embedding_bytes(self) -> int:
        # fp16 tied embedding/head (section 8): 2 bytes per entry, counted once.
        return self.embed.weight.numel() * 2
