"""Arms C and D — algebra-valued ternary, ONE code path (PROTOCOL_v3.md
sections 3, 5, 11, 12; brief rule 2).

Activations are grouped into 8-component multivector channels. The trainable
object is the ``[O, I, 8]`` coefficient tensor; it is ternarized before the
regular contraction (quantize-first). No independently trainable ``[O,I,8,8]``
parameter exists anywhere. Arms C and D differ ONLY in the structure tensor
(``twist`` on/off) — same init draws, quantizer, contraction code, everything.

The forward contraction is the frozen reference ``algebra_linear_training``;
this module only wires it into a budgeted nn.Module and reshapes grouped
channels to/from the flat feature axis.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..algebra import (
    NCOMP,
    algebra_linear_training,
    build_structure_tensor,
    ternarize_weight_global_ste,
)
from .base import ConstrainedLinear, QuantizedExport, init_shadow, register_arm


class AlgebraTernaryLinear(ConstrainedLinear):
    def __init__(self, in_features: int, out_features: int, *, twist: bool,
                 generator=None, init_std: float = 0.02, **_):
        super().__init__()
        if in_features % NCOMP or out_features % NCOMP:
            raise ValueError(
                f"algebra arm needs features divisible by {NCOMP}; "
                f"got in={in_features}, out={out_features}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.twist = twist
        self.I = in_features // NCOMP
        self.O = out_features // NCOMP
        # The ONLY trainable constrained tensor: [O, I, 8]. Never [O, I, 8, 8].
        self.coeff = nn.Parameter(init_shadow((self.O, self.I, NCOMP), generator, std=init_std))
        S = build_structure_tensor(twist=twist)  # int8 [k, a, r]
        self.register_buffer("S", S, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_dims = x.shape[:-1]
        x = x.reshape(*B_dims, self.I, NCOMP)
        if x.ndim == 3:  # [N, I, 8] -> give the reference its expected [B,T,I,8]
            x = x.unsqueeze(1)
            y, _ = algebra_linear_training(x, self.coeff, self.S)
            y = y.squeeze(1)
        else:
            y, _ = algebra_linear_training(x, self.coeff, self.S)
        return y.reshape(*B_dims, self.out_features)

    def ideal_payload_trits(self) -> int:
        return self.O * self.I * NCOMP

    @torch.no_grad()
    def export_quantized(self, name: str):
        _, q, gamma = ternarize_weight_global_ste(self.coeff)
        return [QuantizedExport(
            name=name, trits=q.cpu(),
            scales=gamma.reshape(1).to(torch.float16).cpu(),
            meta={"kind": "algebra", "twist": self.twist,
                  "O": self.O, "I": self.I, "ncomp": NCOMP},
        )]


def _build(in_features, out_features, *, twist: bool, generator=None, init_std=0.02, **_):
    return AlgebraTernaryLinear(in_features, out_features, twist=twist,
                                generator=generator, init_std=init_std)


@register_arm("C")
def _build_c(in_features, out_features, *, generator=None, init_std=0.02, twist=True, **_):
    return _build(in_features, out_features, twist=True, generator=generator, init_std=init_std)


@register_arm("D")
def _build_d(in_features, out_features, *, generator=None, init_std=0.02, twist=False, **_):
    return _build(in_features, out_features, twist=False, generator=generator, init_std=init_std)
