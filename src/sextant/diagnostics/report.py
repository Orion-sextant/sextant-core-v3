"""Diagnostics aggregation (PROTOCOL_v3.md section 14).

Walk a model's algebra constrained modules, sample blocks on a deterministic
hash-selected subset, compute per-block rank + component diagnostics, compare
against the section-4 catalogue, and aggregate by projection type. The full
block-level table is preserved (section 14: "preserve the full block-level
table").
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..arms.arm_cd_algebra import AlgebraTernaryLinear
from ..algebra import ternarize_weight_global_ste
from .components import GRADE_SETS, component_report
from .rank import block_report


def _hash_select(o: int, i: int, name: str, keep_frac: float) -> bool:
    if keep_frac >= 1.0:
        return True
    h = hashlib.sha256(f"{name}:{o}:{i}".encode()).digest()
    return (int.from_bytes(h[:4], "big") / 2**32) < keep_frac


@torch.no_grad()
def diagnose_module(name: str, mod: AlgebraTernaryLinear, arm: str,
                    *, keep_frac: float = 1.0) -> list[dict]:
    """Per-block rows for one algebra module (hard-ternarized coefficients)."""
    _, q, gamma = ternarize_weight_global_ste(mod.coeff)
    q = q.cpu().numpy().astype(np.int64)           # [O, I, 8], values in {-1,0,1}
    rows = []
    for o in range(mod.O):
        for i in range(mod.I):
            if not _hash_select(o, i, name, keep_frac):
                continue
            w = q[o, i]
            row = {"module": name, "o": o, "i": i}
            row.update(block_report(w, arm))
            row.update({k: component_report(w)[k]
                        for k in ("participation_ratio", "enrichment_rho",
                                  "grade_entropy_norm", "sign_balance")})
            rows.append(row)
    return rows


def diagnose_model(model, arm: str, *, keep_frac: float = 1.0) -> dict:
    if arm not in ("C", "D"):
        return {"arm": arm, "note": "block/rank catalogue diagnostics apply to algebra arms C/D only",
                "rows": []}
    rows: list[dict] = []
    for name, mod in model.constrained_modules().items():
        if isinstance(mod, AlgebraTernaryLinear):
            rows.extend(diagnose_module(name, mod, arm, keep_frac=keep_frac))
    return {"arm": arm, "n_blocks": len(rows), "aggregates": _aggregate(rows), "rows": rows}


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    er = np.array([r["effective_rank"] for r in rows])
    exact = np.array([r["exact_rank"] for r in rows])
    in_cat = np.array([r["in_catalogue"] for r in rows])
    pr = np.array([r["participation_ratio"] for r in rows])
    enr = np.array([r["enrichment_rho"] for r in rows])  # [n,4]
    return {
        "effective_rank_mean": float(er.mean()),
        "effective_rank_std": float(er.std()),
        "exact_rank_hist": {int(k): int(v) for k, v in
                            zip(*np.unique(exact, return_counts=True))},
        "catalogue_conformance": float(in_cat.mean()),
        "participation_ratio_mean": float(pr.mean()),
        "enrichment_rho_mean_by_grade": enr.mean(axis=0).tolist(),
    }


def write_table(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float))
    return path
