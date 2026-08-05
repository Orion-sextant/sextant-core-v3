"""Rotary position embeddings (PROTOCOL_v3.md section 8: RoPE positions).

Standard RoPE applied to the query and key projections per head. head_dim is
even (64), so the rotation pairs adjacent dimensions.
"""
from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, head_dim: int, *, base: float = 10000.0, device=None):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # [T, head_dim/2]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [T, head_dim]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, head_dim]; cos/sin: [T, head_dim]."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin
