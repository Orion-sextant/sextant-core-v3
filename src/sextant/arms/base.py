"""Constrained-linear interface shared by all four arms.

The transformer harness is arm-agnostic: it asks a factory for a constrained
linear map from ``in_features`` to ``out_features``. The arm decides the
parameterization (dense scalar ternary, Monarch, or algebra), but every arm
exposes the same surface so one harness, one training loop, and one packer serve
all of them (PROTOCOL_v3.md sections 5, 8; brief rule 2 "one code path for C/D").
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

LOG2_3 = math.log2(3.0)


@dataclass
class QuantizedExport:
    """What the packer serializes for one constrained tensor.

    ``trits`` holds int8 values in {-1, 0, +1} (arm A/B scalar) or the [O,I,8]
    algebra coefficients (arm C/D). ``scales`` holds the fp16 scale(s). ``meta``
    holds arm-specific structure (permutations, factor shapes) that must be
    stored to reconstruct the map.
    """
    name: str
    trits: torch.Tensor          # int8, values in {-1,0,1}
    scales: torch.Tensor         # fp16
    meta: dict


class ConstrainedLinear(nn.Module, abc.ABC):
    """A budgeted linear map. Trainable state is high-precision *shadow* weights;
    the forward path ternarizes them (identity STE) before contraction."""

    in_features: int
    out_features: int

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [.., in] -> [.., out]
        ...

    @abc.abstractmethod
    def ideal_payload_trits(self) -> int:
        """Number of ternary trits in the constrained payload (section 7 measure 1)."""

    @abc.abstractmethod
    def export_quantized(self, name: str) -> list[QuantizedExport]:
        """Quantized tensors for the packer (bit-exact with the forward path)."""

    def ideal_payload_bits(self) -> float:
        return self.ideal_payload_trits() * LOG2_3


# arm letter -> builder(in_features, out_features, *, generator, twist, **kw)
Factory = Callable[..., ConstrainedLinear]
_REGISTRY: dict[str, Factory] = {}


def register_arm(letter: str) -> Callable[[Factory], Factory]:
    def deco(fn: Factory) -> Factory:
        _REGISTRY[letter] = fn
        return fn
    return deco


def build_constrained(
    letter: str,
    in_features: int,
    out_features: int,
    *,
    generator: torch.Generator | None = None,
    **kw,
) -> ConstrainedLinear:
    if letter not in _REGISTRY:
        raise KeyError(f"arm {letter!r} not registered; have {sorted(_REGISTRY)}")
    return _REGISTRY[letter](in_features, out_features, generator=generator, **kw)


def init_shadow(shape, generator, std: float = 0.02, device=None) -> torch.Tensor:
    """N(0, std^2) shadow init (section 10). Drawn from the paired generator so
    arms C and D coincide wherever shapes coincide."""
    t = torch.empty(*shape, device=device, dtype=torch.float32)
    t.normal_(0.0, std, generator=generator)
    return t
