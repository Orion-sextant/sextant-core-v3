"""Component-use diagnostics (PROTOCOL_v3.md section 14), computed separately
from rank (effective rank is spectral geometry, not coefficient utilization).

Grades: grade_map (0,1,1,2,1,2,2,3); grade dims (1,3,3,1). Enrichment is
measured against the exchangeable-components reference E[normalized m_g] = d_g/8.
"""
from __future__ import annotations

import numpy as np

GRADE_MAP = np.array([0, 1, 1, 2, 1, 2, 2, 3])
GRADE_DIM = np.array([1, 3, 3, 1])          # d_g for g = 0,1,2,3
GRADE_SETS = [np.where(GRADE_MAP == g)[0] for g in range(4)]
LN4 = np.log(4.0)


def participation_ratio(w: np.ndarray) -> float:
    """PR(w) = (sum w^2)^2 / sum w^4, PR(0) = 0. Continuous participation."""
    w = np.asarray(w, dtype=np.float64)
    num = (w ** 2).sum() ** 2
    den = (w ** 4).sum()
    return float(num / den) if den > 0 else 0.0


def grade_masses(w: np.ndarray) -> dict:
    """Raw and normalized grade mass, per-component grade energy, enrichment,
    grade entropy. Masses use squared coefficients."""
    w = np.asarray(w, dtype=np.float64)
    energy = w ** 2
    total = energy.sum()
    m_g = np.array([energy[idx].sum() for idx in GRADE_SETS])         # raw grade mass
    mbar = m_g / total if total > 0 else np.zeros(4)                  # normalized
    per_component = m_g / GRADE_DIM                                   # m_g / d_g
    enrichment = mbar / (GRADE_DIM / 8.0)                            # rho_g
    nz = mbar[mbar > 0]
    grade_entropy = float(-(nz * np.log(nz)).sum() / LN4) if nz.size else 0.0
    return {
        "raw_grade_mass": m_g.tolist(),
        "norm_grade_mass": mbar.tolist(),
        "per_component_grade_energy": per_component.tolist(),
        "enrichment_rho": enrichment.tolist(),
        "grade_entropy_norm": grade_entropy,
    }


def component_report(w: np.ndarray) -> dict:
    w = np.asarray(w, dtype=np.float64)
    out = {
        "n_nonzero": int(np.count_nonzero(w)),
        "participation_ratio": participation_ratio(w),
        "zero_fraction": float(np.mean(w == 0)),
        "sign_balance": float(np.sign(w).sum() / max(1, np.count_nonzero(w))),
        "per_component_mass": (w ** 2).tolist(),
    }
    out.update(grade_masses(w))
    return out


def quantization_error(w_shadow: np.ndarray, w_hard: np.ndarray) -> float:
    a, b = np.asarray(w_shadow, float), np.asarray(w_hard, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def activation_clip_rate(q_activation: np.ndarray, qmax: int = 127) -> float:
    q = np.asarray(q_activation)
    return float(np.mean(np.abs(q) >= qmax))
