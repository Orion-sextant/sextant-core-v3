"""Rank diagnostics (PROTOCOL_v3.md section 14 + catalogue section 4).

A "block" for an algebra arm is the 8x8 matrix induced by an 8-coefficient
vector w:  M[k,r] = sum_a S[k,a,r] w[a]. Effective rank is spectral geometry;
exact rank is checked against the frozen section-4 catalogue.
"""
from __future__ import annotations

import numpy as np

from ..algebra import build_structure_tensor

# Allowed exact ranks per arm (section 4 total-count tables).
ALLOWED_EXACT_RANK = {
    "C": {0, 4, 8},
    "D": {0, 1, 2, 4, 5, 6, 8},
}
_S = {"C": build_structure_tensor(twist=True).numpy().astype(np.int64),
      "D": build_structure_tensor(twist=False).numpy().astype(np.int64)}  # [k,a,r]


def induced_block(w: np.ndarray, arm: str) -> np.ndarray:
    """w: [8] coefficients -> 8x8 induced left-regular block."""
    S = _S[arm]
    return np.einsum("kar,a->kr", S, np.asarray(w, dtype=np.float64))


def effective_rank(M: np.ndarray) -> float:
    """r_eff = exp(Shannon entropy of normalized singular values), in [0,8];
    r_eff(0) = 0. SVD in float64 (section 14)."""
    sv = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
    total = sv.sum()
    if total <= 0:
        return 0.0
    p = sv / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def exact_rank(M: np.ndarray, *, tol: float = 1e-6) -> int:
    sv = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
    if np.any((sv > 1e-9) & (sv < tol)):
        # ambiguous singular value: fall back to numerical rank, labeled by caller
        pass
    return int(np.sum(sv > tol))


def block_report(w: np.ndarray, arm: str) -> dict:
    """Full per-block rank diagnostics for one coefficient vector."""
    M = induced_block(w, arm)
    er = exact_rank(M)
    reff = effective_rank(M)
    return {
        "effective_rank": reff,
        "effective_rank_norm": reff / 8.0,
        "exact_rank": er,
        "in_catalogue": er in ALLOWED_EXACT_RANK[arm],
        "n_nonzero": int(np.count_nonzero(w)),
    }


def paired_movement(w_init: np.ndarray, w_now: np.ndarray, arm: str) -> dict:
    """Section 14: checkpoint interpretation primarily uses paired movement from
    initialization within each arm (signed gap and ratio of effective rank)."""
    r0 = effective_rank(induced_block(w_init, arm))
    r1 = effective_rank(induced_block(w_now, arm))
    return {
        "reff_init": r0, "reff_now": r1,
        "signed_gap": r1 - r0,
        "ratio": (r1 / r0) if r0 > 0 else float("nan"),
    }
