"""Section 13 kernel tests: integer-reference equality, signed-permutation
bit-exactness, accumulator safety.
"""
import torch

from sextant.algebra import (
    build_structure_tensor,
    quantize_activation_per_token_ste,
    ternarize_weight_global_ste,
    algebra_linear_training,
    algebra_linear_integer_reference,
    algebra_linear_signed_permutation_reference,
)


def _setup(O=3, I=4, B=2, T=5, twist=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = build_structure_tensor(twist=twist)
    w = torch.randn(O, I, 8, generator=g, dtype=torch.float64)
    x = torch.randn(B, T, I, 8, generator=g, dtype=torch.float64)
    x_eff, q_x, delta = quantize_activation_per_token_ste(x)
    w_eff, q_w, gamma = ternarize_weight_global_ste(w)
    return S, w, x, q_x, delta, q_w, gamma


def test_integer_reference_equals_training_forward():
    for twist in (True, False):
        S, w, x, q_x, delta, q_w, gamma = _setup(twist=twist)
        y_train, _ = algebra_linear_training(x, w, S)
        # the frozen integer reference emits float32 (the deployment path); compare
        # the dequantized training forward to it in float32.
        y_int = algebra_linear_integer_reference(q_x, q_w, delta, gamma, S)
        assert torch.allclose(y_train.to(torch.float32), y_int, atol=1e-4, rtol=1e-4)


def test_signed_permutation_bit_exact():
    for twist in (True, False):
        S, w, x, q_x, delta, q_w, gamma = _setup(twist=twist)
        acc_sparse = torch.einsum(
            "kar,oia,btir->btok",
            S.to(torch.int32), q_w.to(torch.int32), q_x.to(torch.int32),
        )
        acc_perm = algebra_linear_signed_permutation_reference(q_x, q_w, S)
        assert acc_perm.dtype == torch.int32
        assert torch.equal(acc_sparse, acc_perm)   # bit-exact integer equality


def test_accumulator_safety_algebra():
    # |acc| <= 8 * I * 127 (section 12), int32-safe. Stress at max int8 magnitude.
    I = 16
    S = build_structure_tensor(twist=True)
    q_x = torch.full((1, 1, I, 8), 127, dtype=torch.int8)
    q_w = torch.ones(2, I, 8, dtype=torch.int8)
    acc = algebra_linear_signed_permutation_reference(q_x, q_w, S)
    bound = 8 * I * 127
    assert acc.abs().max().item() <= bound
    assert bound < 2**31
