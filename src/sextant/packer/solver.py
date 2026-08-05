"""Deterministic budget solver (PROTOCOL_v3.md section 8).

Search d_model over multiples of 8; minimize |actual packed constrained bytes -
target|; accept within tolerance (+/-1%); tie-break to the smaller d_model.

Head integrality: head_dim is fixed at 64, so a buildable model needs
d_model % 64 == 0 (integer head count). This is reported alongside the raw
multiples-of-8 optimum so any conflict between the byte tolerance and integer
64-dim heads is explicit rather than silently resolved (brief rule 5).
"""
from __future__ import annotations

from dataclasses import dataclass

from .serializer import cell_actual_bytes, cell_ideal_trits

HEAD_DIM = 64
TARGETS = {"low": 2_500_000, "high": 10_000_000}
TOLERANCE = 0.01


@dataclass
class CellSolution:
    arm: str
    budget: str
    d_model: int
    actual_bytes: int
    target: int
    rel_err: float
    within_tolerance: bool
    integer_heads: bool
    n_heads: float

    def as_manifest(self) -> dict:
        return {
            "d_model": self.d_model,
            "actual_constrained_bytes": self.actual_bytes,
            "ideal_constrained_payload_bits": round(
                cell_ideal_trits(self.arm, self.d_model) * 1.584962500721156, 2),
            "rel_err": round(self.rel_err, 5),
            "within_tolerance": self.within_tolerance,
            "integer_heads": self.integer_heads,
            "n_heads": self.n_heads,
        }


def solve_cell(arm: str, budget: str, *, grid_step: int = 8, d_max: int = 4096,
               require_integer_heads: bool = False) -> CellSolution:
    target = TARGETS[budget]
    best_d = None
    best_gap = None
    for d in range(grid_step, d_max + 1, grid_step):
        if require_integer_heads and d % HEAD_DIM != 0:
            continue
        gap = abs(cell_actual_bytes(arm, d) - target)
        # tie-break to smaller d_model: strict-less keeps the first (smaller) d.
        if best_gap is None or gap < best_gap:
            best_gap, best_d = gap, d
    ab = cell_actual_bytes(arm, best_d)
    rel = (ab - target) / target
    return CellSolution(
        arm=arm, budget=budget, d_model=best_d, actual_bytes=ab, target=target,
        rel_err=rel, within_tolerance=abs(rel) <= TOLERANCE,
        integer_heads=(best_d % HEAD_DIM == 0), n_heads=best_d / HEAD_DIM,
    )


def solve_all(**kw) -> dict[str, CellSolution]:
    out = {}
    for arm in ("A", "B", "C", "D"):
        for budget in ("low", "high"):
            out[f"{arm}_{budget}"] = solve_cell(arm, budget, **kw)
    return out
