"""Arm A — dense scalar ternary (PROTOCOL_v3.md section 5, the null).

Unconstrained scalar matrix; global-absmean ternary weights; per-token int8
activations; fp32 shadow weights in training; identity STE.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import ConstrainedLinear, QuantizedExport, init_shadow, register_arm
from .quant import quantize_activation_per_token, ternarize_weight_global


class DenseTernaryLinear(ConstrainedLinear):
    def __init__(self, in_features: int, out_features: int, *,
                 generator=None, init_std: float = 0.02, **_):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            init_shadow((out_features, in_features), generator, std=init_std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_eff, _, _ = quantize_activation_per_token(x)
        w_eff, _, _ = ternarize_weight_global(self.weight)
        return torch.matmul(x_eff, w_eff.t())

    def ideal_payload_trits(self) -> int:
        return self.out_features * self.in_features

    @torch.no_grad()
    def export_quantized(self, name: str):
        _, q, gamma = ternarize_weight_global(self.weight)
        return [QuantizedExport(
            name=name, trits=q.cpu(),
            scales=gamma.reshape(1).to(torch.float16).cpu(),
            meta={"kind": "dense", "out": self.out_features, "in": self.in_features},
        )]


@register_arm("A")
def _build(in_features, out_features, *, generator=None, init_std=0.02, **_):
    return DenseTernaryLinear(in_features, out_features,
                              generator=generator, init_std=init_std)
