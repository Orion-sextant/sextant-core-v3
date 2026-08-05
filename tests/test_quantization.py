"""Section 13 quantization/STE tests: quantize-first equality, tied-expansion
equivalence, independent-entry prohibition, weight-STE gradient of ones, scale
detachment, activation flattening consistency, single scale per token,
zero-token safety, algebra-layout preservation.
"""
import torch

from sextant.algebra import (
    build_structure_tensor,
    quantize_activation_per_token_ste,
    ternarize_weight_global_ste,
    algebra_linear_training,
)
from sextant.model.transformer import Transformer, ModelArgs
from sextant.seeding import seed_everything, paired_generator

torch.manual_seed(0)


def _rand(o, i, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(o, i, 8, generator=g, dtype=torch.float64)


def test_quantize_first_equals_expand_from_ternarized_coeff():
    O, I = 3, 4
    S = build_structure_tensor(twist=True).to(torch.float64)
    w = _rand(O, I).requires_grad_(True)
    x = torch.randn(2, 5, I, 8, dtype=torch.float64)
    y_ref, _ = algebra_linear_training(x, w, S)
    # expand ternarized coeff to the 8x8 block, then contract — must agree.
    x_eff, _, _ = quantize_activation_per_token_ste(x)
    w_eff, _, _ = ternarize_weight_global_ste(w)
    block = torch.einsum("kar,oia->oikr", S, w_eff)         # tied [O,I,8,8]
    y_expand = torch.einsum("oikr,btir->btok", block, x_eff)
    assert torch.allclose(y_ref, y_expand, atol=1e-10)


def test_tied_expansion_equivalence_forward_and_gradient():
    O, I = 3, 4
    S = build_structure_tensor(twist=True).to(torch.float64)
    x = torch.randn(2, 5, I, 8, dtype=torch.float64)

    w1 = _rand(O, I).requires_grad_(True)
    y1, _ = algebra_linear_training(x, w1, S)
    y1.pow(2).sum().backward()

    w2 = _rand(O, I).requires_grad_(True)
    x_eff, _, _ = quantize_activation_per_token_ste(x)
    w_eff, _, _ = ternarize_weight_global_ste(w2)
    block = torch.einsum("kar,oia->oikr", S, w_eff)
    y2 = torch.einsum("oikr,btir->btok", block, x_eff)
    y2.pow(2).sum().backward()

    assert torch.allclose(y1, y2, atol=1e-10)
    assert torch.allclose(w1.grad, w2.grad, atol=1e-8)  # coefficient gradients equal


def test_independent_entry_prohibition():
    seed_everything(1); gen = paired_generator(1)
    m = Transformer(ModelArgs(arm="C", d_model=64, vocab_size=64, depth=2, twist=True), gen)
    for n, p in m.named_parameters():
        assert not (p.ndim == 4 and tuple(p.shape[-2:]) == (8, 8)), \
            f"forbidden [O,I,8,8] trainable parameter: {n} {tuple(p.shape)}"
    # the only constrained trainable is [O,I,8] (ndim 3, last dim 8)
    from sextant.arms.arm_cd_algebra import AlgebraTernaryLinear
    for mod in m.constrained_modules().values():
        assert isinstance(mod, AlgebraTernaryLinear)
        assert mod.coeff.ndim == 3 and mod.coeff.shape[-1] == 8


def test_weight_ste_gradient_of_ones():
    w = torch.randn(3, 4, 8, dtype=torch.float64, requires_grad=True)
    w_eff, _, _ = ternarize_weight_global_ste(w)
    w_eff.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))  # dW_eff/dW_shadow = I


def test_scale_detachment():
    w = torch.randn(3, 4, 8, dtype=torch.float64, requires_grad=True)
    _, q, gamma = ternarize_weight_global_ste(w)
    assert not gamma.requires_grad          # gamma detached
    assert q.dtype == torch.int8


def test_activation_flattening_consistency():
    x = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    _, _, delta_grouped = quantize_activation_per_token_ste(x)
    # per-token scale over (I,8) must equal absmax over the flattened I*8 axis / 127
    flat = x.reshape(2, 3, 5 * 8)
    expected = flat.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    assert torch.allclose(delta_grouped.reshape(2, 3, 1), expected)


def test_single_scale_per_token():
    x = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    _, _, delta = quantize_activation_per_token_ste(x)
    assert delta.shape == (2, 3, 1, 1)      # one scale per token, not per component


def test_zero_token_safety():
    x = torch.zeros(2, 3, 5, 8, dtype=torch.float64)
    x_eff, q, delta = quantize_activation_per_token_ste(x)
    assert torch.isfinite(x_eff).all() and torch.isfinite(delta).all()
    assert (q == 0).all()


def test_algebra_layout_preservation():
    # reshape d <-> (I,8) round-trips; output feature f maps to (o=f//8, k=f%8)
    B, T, I = 2, 3, 4
    x = torch.arange(B * T * I * 8, dtype=torch.float64).reshape(B, T, I * 8)
    assert torch.equal(x.reshape(B, T, I, 8).reshape(B, T, I * 8), x)
