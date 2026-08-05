"""Scalar ternary + activation quantizers (PROTOCOL_v3.md section 11), shared by
arms A and B. Same rule as the frozen algebra reference: global absmean weight
scale, per-token symmetric int8 activation scale, identity STE, no gradient
through gamma/rounding/clipping.
"""
from __future__ import annotations

import torch

QMAX_ACT = 127


def ternarize_weight_global(w_shadow: torch.Tensor, *, eps: float = 1e-8):
    """One detached absmean scale over the whole tensor. Returns
    (w_effective, q_weight int8 in {-1,0,1}, gamma)."""
    gamma = w_shadow.detach().abs().mean().clamp_min(eps)
    q = torch.round(w_shadow.detach() / gamma).clamp(-1, 1).to(torch.int8)
    w_hard = gamma * q.to(w_shadow.dtype)
    w_eff = w_shadow + (w_hard - w_shadow).detach()
    return w_eff, q, gamma


def quantize_activation_per_token(x: torch.Tensor, *, eps: float = 1e-8):
    """x: [..., F]. One symmetric int8 scale per token over the last dim."""
    absmax = x.detach().abs().amax(dim=-1, keepdim=True)
    delta = absmax.clamp_min(eps) / QMAX_ACT
    q = torch.round(x.detach() / delta).clamp(-QMAX_ACT, QMAX_ACT).to(torch.int8)
    x_hard = delta * q.to(x.dtype)
    x_eff = x + (x_hard - x).detach()
    return x_eff, q, delta
