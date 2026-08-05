"""Section 13 algebra tests: basis mapping, cocycle identity, associativity,
identity element, signature, C anticommutation, D commutativity, left/right
discrimination, regular homomorphism, explicit block equality with the frozen
templates.
"""
import itertools

import numpy as np
import torch

from sextant.algebra import build_structure_tensor, cocycle_sign
from sextant.diagnostics.components import GRADE_MAP, GRADE_SETS


def _Lbasis(twist):
    S = build_structure_tensor(twist=twist).numpy().astype(np.int64)
    return np.stack([S[:, a, :] for a in range(8)])  # [a,k,r]


LC, LD = _Lbasis(True), _Lbasis(False)


def test_basis_mapping():
    assert GRADE_MAP.tolist() == [0, 1, 1, 2, 1, 2, 2, 3]
    assert [s.tolist() for s in GRADE_SETS] == [[0], [1, 2, 4], [3, 5, 6], [7]]


def test_cocycle_identity_512():
    for x, y, z in itertools.product(range(8), repeat=3):
        assert (cocycle_sign(x, y, twist=True) * cocycle_sign(x ^ y, z, twist=True)
                == cocycle_sign(y, z, twist=True) * cocycle_sign(x, y ^ z, twist=True))


def test_associativity_512_both_arms():
    for name, L, tw in [("C", LC, True), ("D", LD, False)]:
        for a, b, c in itertools.product(range(8), repeat=3):
            # (u_a u_b) u_c == u_a (u_b u_c) via left-regular matrices
            assert np.array_equal((L[a] @ L[b]) @ L[c], L[a] @ (L[b] @ L[c])), (name, a, b, c)


def test_identity_element():
    for L in (LC, LD):
        assert np.array_equal(L[0], np.eye(8, dtype=np.int64))


def test_signature_generators_square_to_plus_one():
    for m in (1, 2, 4):  # e1, e2, e3
        assert np.array_equal(LC[m] @ LC[m], np.eye(8, dtype=np.int64))


def test_C_anticommutation():
    for m1, m2 in [(1, 2), (1, 4), (2, 4)]:
        assert np.array_equal(LC[m1] @ LC[m2], -(LC[m2] @ LC[m1]))


def test_D_commutativity():
    for a, b in itertools.product(range(8), repeat=2):
        assert np.array_equal(LD[a] @ LD[b], LD[b] @ LD[a])


def test_left_right_discrimination():
    # e1 e2 = +e12, e2 e1 = -e12  (masks 1,2 -> 3)
    assert cocycle_sign(1, 2, twist=True) == 1
    assert cocycle_sign(2, 1, twist=True) == -1


def test_regular_homomorphism_both_arms():
    for L, tw in [(LC, True), (LD, False)]:
        for a, b in itertools.product(range(8), repeat=2):
            assert np.array_equal(L[a] @ L[b], cocycle_sign(a, b, twist=tw) * L[a ^ b])


# Frozen explicit L_C templates (PROTOCOL section 4.1 / verify_algebra.py).
DOC_IDX = [
    [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
    [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
    [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 7, 6, 1, 0, 3, 2],
    [6, 7, 4, 5, 2, 3, 0, 1], [7, 6, 5, 4, 3, 2, 1, 0],
]
DOC_SGN = [
    [+1, +1, +1, -1, +1, -1, -1, -1], [+1, +1, +1, -1, +1, -1, -1, -1],
    [+1, -1, +1, +1, +1, +1, -1, +1], [+1, -1, +1, +1, +1, +1, -1, +1],
    [+1, -1, -1, -1, +1, +1, +1, -1], [+1, -1, -1, -1, +1, +1, +1, -1],
    [+1, +1, -1, +1, +1, -1, +1, +1], [+1, +1, -1, +1, +1, -1, +1, +1],
]


def test_explicit_block_templates_C():
    S = build_structure_tensor(twist=True).numpy()
    for k in range(8):
        for r in range(8):
            a = k ^ r
            assert DOC_IDX[k][r] == a
            assert S[k, a, r] == DOC_SGN[k][r]


def test_explicit_block_templates_D():
    S = build_structure_tensor(twist=False).numpy()
    for k in range(8):
        for r in range(8):
            assert S[k, k ^ r, r] == 1
