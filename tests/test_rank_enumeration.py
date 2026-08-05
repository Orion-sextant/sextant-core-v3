"""Mandatory unit test #23 (PROTOCOL_v3.md section 13): rank enumeration
reproducing section 4 exactly.

This is the installed, pytest-wrapped form of ``reference/verify_algebra.py``.
It exhaustively enumerates all 3^8 = 6561 ternary coefficient vectors per arm,
recomputes the exact-rank catalogue, and asserts equality with the frozen
tables in PROTOCOL_v3.md section 4 (and the singular-value multiplicity and
Walsh-rank claims of section 3.3 / 4).
"""
import itertools

import numpy as np
import pytest

from sextant.algebra import build_structure_tensor


def _Lbasis(twist):
    S = build_structure_tensor(twist=twist).numpy().astype(np.int64)  # [k,a,r]
    return np.stack([S[:, a, :] for a in range(8)])  # [a,k,r]


LC = _Lbasis(True)
LD = _Lbasis(False)


def _exact_rank(M):
    sv = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    assert not np.any((sv > 1e-9) & (sv < 1e-6)), f"ambiguous singular value {sv}"
    return int(np.sum(sv > 1e-6)), sv


# Frozen tables — PROTOCOL_v3.md section 4.
DOC_TOT_C = {0: 1, 4: 672, 8: 5888}
DOC_TOT_D = {0: 1, 1: 16, 2: 112, 4: 672, 5: 896, 6: 1344, 8: 3520}
DOC_COND_C = {
    0: {0: 1}, 1: {8: 16}, 2: {4: 48, 8: 64}, 3: {8: 448},
    4: {4: 144, 8: 976}, 5: {8: 1792}, 6: {4: 384, 8: 1408},
    7: {8: 1024}, 8: {4: 96, 8: 160},
}
DOC_COND_D = {
    0: {0: 1}, 1: {8: 16}, 2: {4: 112}, 3: {8: 448},
    4: {2: 112, 5: 896, 8: 112}, 5: {8: 1792}, 6: {4: 448, 6: 1344},
    7: {8: 1024}, 8: {1: 16, 4: 112, 8: 128},
}


@pytest.fixture(scope="module")
def enumeration():
    tot = {"C": {}, "D": {}}
    cond = {"C": {}, "D": {}}
    mult_ok = True
    walsh_ok = True
    H = np.array([[(-1) ** bin(s & x).count("1") for x in range(8)] for s in range(8)],
                 dtype=np.int64)
    for q in itertools.product([-1, 0, 1], repeat=8):
        w = np.array(q, dtype=np.int64)
        nnz = int(np.count_nonzero(w))
        MC = np.einsum("akr,a->kr", LC, w)
        MD = np.einsum("akr,a->kr", LD, w)
        rC, svC = _exact_rank(MC)
        rD, _ = _exact_rank(MD)
        tot["C"][rC] = tot["C"].get(rC, 0) + 1
        tot["D"][rD] = tot["D"].get(rD, 0) + 1
        cond["C"].setdefault(nnz, {}).setdefault(rC, 0)
        cond["C"][nnz][rC] += 1
        cond["D"].setdefault(nnz, {}).setdefault(rD, 0)
        cond["D"][nnz][rD] += 1
        if rC == 4:
            nz = svC[svC > 1e-6]
            if not (len(nz) == 4 and np.allclose(nz, nz[0], rtol=1e-9)):
                mult_ok = False
        elif rC == 8:
            s_sorted = np.sort(svC)
            g1, g2 = s_sorted[:4], s_sorted[4:]
            if not (np.allclose(g1, g1[0], rtol=1e-9) and np.allclose(g2, g2[0], rtol=1e-9)):
                mult_ok = False
        if rD != int(np.count_nonzero(H @ w)):
            walsh_ok = False
    return tot, cond, mult_ok, walsh_ok


def test_total_counts_match_section4(enumeration):
    tot, _, _, _ = enumeration
    assert tot["C"] == DOC_TOT_C
    assert tot["D"] == DOC_TOT_D
    assert sum(tot["C"].values()) == 6561
    assert sum(tot["D"].values()) == 6561


def test_conditioned_counts_match_section4(enumeration):
    _, cond, _, _ = enumeration
    assert cond["C"] == DOC_COND_C
    assert cond["D"] == DOC_COND_D


def test_C_singular_value_multiplicity(enumeration):
    _, _, mult_ok, _ = enumeration
    assert mult_ok, "arm C: rank-4 -> 4 equal SVs; rank-8 -> two quadruples"


def test_D_rank_equals_nonzero_walsh(enumeration):
    _, _, _, walsh_ok = enumeration
    assert walsh_ok, "arm D block rank must equal number of nonzero Walsh coefficients"
