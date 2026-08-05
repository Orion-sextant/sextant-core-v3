"""Arm B — Monarch two-factor ternary (PROTOCOL_v3.md section 5, primary path).

A budgeted linear map realized as  block-diagonal (W1) x permutation x
block-diagonal (W2), never densely materialized. Both factors ternarized under
the section-11 rule; the input is quantized ONCE; there is no intermediate
requantization, normalization, or nonlinearity in the primary path; the integer
intermediate is int32 with a proven bound; factor scales and the activation
scale are applied AFTER the second contraction.

Factor shapes for a map in -> out with ``g`` blocks:
    W1: (g, a, a)   with a = in // g     (square blocks: p = q = a)
    W2: (g, c, a)   with c = out // g    (r = a, s = c)
so in = g*a, out = g*c, and the intermediate grid k*q = l*r = g*a. The
permutation is the transpose of the (g, a) intermediate grid — the
characteristic Monarch shuffle.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import ConstrainedLinear, QuantizedExport, init_shadow, register_arm
from .quant import QMAX_ACT, quantize_activation_per_token, ternarize_weight_global


def choose_blocks(in_features: int, out_features: int) -> int:
    """Pick block count g dividing both in and out, near sqrt(in) to balance the
    two factors. Deterministic (used by the budget solver too)."""
    target = max(1, int(round(math.sqrt(in_features))))
    best = 1
    for g in range(1, in_features + 1):
        if in_features % g == 0 and out_features % g == 0:
            if abs(g - target) < abs(best - target) or best == 1:
                best = g
    return best


class MonarchTernaryLinear(ConstrainedLinear):
    def __init__(self, in_features: int, out_features: int, *, g: int | None = None,
                 generator=None, init_std: float = 0.02, **_):
        super().__init__()
        g = g or choose_blocks(in_features, out_features)
        if in_features % g or out_features % g:
            raise ValueError(f"g={g} must divide in={in_features} and out={out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.g = g
        self.a = in_features // g       # block inner dim (p=q=r)
        self.c = out_features // g      # second-factor output block (s)
        # int32 accumulator bound (section 5/12): |acc2| <= a * (a * 127).
        self.acc_bound = self.a * self.a * QMAX_ACT
        assert self.acc_bound < 2**31, f"int32 overflow risk: bound {self.acc_bound}"
        # Two block-diagonal ternary factors. Fan-in-aware init per block.
        self.w1 = nn.Parameter(init_shadow((g, self.a, self.a), generator, std=init_std))
        self.w2 = nn.Parameter(init_shadow((g, self.c, self.a), generator, std=init_std))

    # -- shared permutation between the two factors --------------------------
    def _permute(self, o1, b):
        # o1: (b, g, a) -> transpose (g,a) grid -> (b, g, a) reinterpreted (l=g, r=a)
        return o1.transpose(1, 2).reshape(b, self.a * self.g).reshape(b, self.g, self.a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[:-1]
        b = int(torch.tensor(bs).prod().item()) if bs else 1
        x_eff, _, _ = quantize_activation_per_token(x)
        w1_eff, _, _ = ternarize_weight_global(self.w1)
        w2_eff, _, _ = ternarize_weight_global(self.w2)
        xr = x_eff.reshape(b, self.g, self.a)
        o1 = torch.einsum("gij,bgj->bgi", w1_eff, xr)     # (b,g,a)  no nonlinearity
        o1 = self._permute(o1, b)
        o2 = torch.einsum("gci,bgi->bgc", w2_eff, o1)     # (b,g,c)
        return o2.reshape(*bs, self.out_features)

    @torch.no_grad()
    def int_reference(self, x: torch.Tensor):
        """Integer primary path: quantize once, two int matmuls, scale after the
        second contraction. Returns (y_float, acc_int32)."""
        bs = x.shape[:-1]
        b = int(torch.tensor(bs).prod().item()) if bs else 1
        _, q_x, delta = quantize_activation_per_token(x)
        _, q1, g1 = ternarize_weight_global(self.w1)
        _, q2, g2 = ternarize_weight_global(self.w2)
        xr = q_x.to(torch.int32).reshape(b, self.g, self.a)
        o1 = torch.einsum("gij,bgj->bgi", q1.to(torch.int32), xr)
        assert o1.abs().max().item() <= self.a * QMAX_ACT
        o1 = self._permute(o1, b)
        acc = torch.einsum("gci,bgi->bgc", q2.to(torch.int32), o1)  # int32
        assert acc.abs().max().item() <= self.acc_bound
        scale = (g1 * g2).to(torch.float32) * delta.reshape(b, 1, 1).to(torch.float32)
        y = acc.to(torch.float32) * scale
        return y.reshape(*bs, self.out_features), acc

    def ideal_payload_trits(self) -> int:
        return self.w1.numel() + self.w2.numel()

    @torch.no_grad()
    def export_quantized(self, name: str):
        _, q1, g1 = ternarize_weight_global(self.w1)
        _, q2, g2 = ternarize_weight_global(self.w2)
        meta = {"kind": "monarch", "g": self.g, "a": self.a, "c": self.c,
                "in": self.in_features, "out": self.out_features}
        return [
            QuantizedExport(f"{name}.w1", q1.cpu(), g1.reshape(1).to(torch.float16).cpu(), meta),
            QuantizedExport(f"{name}.w2", q2.cpu(), g2.reshape(1).to(torch.float16).cpu(), meta),
        ]


@register_arm("B")
def _build(in_features, out_features, *, generator=None, init_std=0.02, g=None, **_):
    return MonarchTernaryLinear(in_features, out_features, g=g,
                                generator=generator, init_std=init_std)
