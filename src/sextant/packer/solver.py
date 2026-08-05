"""Deterministic budget solver (PROTOCOL_v3.md section 8; amendment v3.1).

Search d_model, minimize |actual packed constrained bytes - target|, accept
within +/-1%, tie-break to the smaller d_model.

Search grid per arm reflects the divisibility each arm actually requires
(section 8: the "divisible by 8" clause is scoped to C and D — the algebra
8-grouping). Arms A (dense) and B (Monarch factors) have no such constraint, so
A uses a fine grid; C/D use multiples of 8. Under v3.1 the attention inner dim
is decoupled from d_model (n_heads = floor(d_model/64 + 0.5)), so every
candidate has an integer head count by construction — the old head-integrality
conflict no longer binds. See docs/CELL_POLICY_CONFLICT.md and
docs/PROTOCOL_v3.1_amendment.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .serializer import _attn_inner, cell_actual_bytes, cell_ideal_trits

HEAD_DIM = 64
TARGETS = {"low": 2_500_000, "high": 10_000_000}
TOLERANCE = 0.01
# Search granularity per arm (section 8 divisibility scoping).
ARM_GRID = {"A": 1, "B": 8, "C": 8, "D": 8}


@dataclass
class CellSolution:
    arm: str
    budget: str
    d_model: int
    actual_bytes: int
    target: int
    rel_err: float
    within_tolerance: bool
    n_heads: int
    attn_inner: int

    def as_manifest(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "attn_inner": self.attn_inner,
            "actual_constrained_bytes": self.actual_bytes,
            "ideal_constrained_payload_bits": round(
                cell_ideal_trits(self.arm, self.d_model) * 1.584962500721156, 2),
            "rel_err": round(self.rel_err, 5),
            "within_tolerance": self.within_tolerance,
        }


def solve_cell(arm: str, budget: str, *, grid_step: int | None = None,
               d_max: int = 4096) -> CellSolution:
    target = TARGETS[budget]
    step = grid_step if grid_step is not None else ARM_GRID[arm]
    best_d = None
    best_gap = None
    for d in range(step, d_max + 1, step):
        gap = abs(cell_actual_bytes(arm, d) - target)
        if best_gap is None or gap < best_gap:   # strict '<' keeps the smaller d
            best_gap, best_d = gap, d
    ab = cell_actual_bytes(arm, best_d)
    rel = (ab - target) / target
    inner = _attn_inner(best_d)
    return CellSolution(
        arm=arm, budget=budget, d_model=best_d, actual_bytes=ab, target=target,
        rel_err=rel, within_tolerance=abs(rel) <= TOLERANCE,
        n_heads=inner // HEAD_DIM, attn_inner=inner,
    )


def solve_all(**kw) -> dict[str, CellSolution]:
    out = {}
    for arm in ("A", "B", "C", "D"):
        for budget in ("low", "high"):
            out[f"{arm}_{budget}"] = solve_cell(arm, budget, **kw)
    return out
