"""Sextant Core reference kernels. Frozen conventions; see PROTOCOL_v3.md sections 3, 11, 12.

Basis (bitmask order): 0:1, 1:e1, 2:e2, 3:e12, 4:e3, 5:e13, 6:e23, 7:e123.
Weight is always the LEFT factor. C/D differ only in the structure tensor.
These functions are normative: production code must match them bit-exactly
where the protocol says so (integer reference, signed-permutation kernel).
"""
from __future__ import annotations

import torch
from torch import Tensor

NCOMP = 8
QMAX_ACT = 127


def cocycle_sign(a: int, r: int, *, twist: bool) -> int:
    """F(a, r) = (-1)^B(a, r) with B(a, r) = sum_{j<i} a_i * r_j (mod 2)."""
    if not twist:
        return 1
    parity = 0
    for i in range(3):
        ai = (a >> i) & 1
        for j in range(i):
            rj = (r >> j) & 1
            parity ^= ai & rj
    return -1 if parity else 1


def build_structure_tensor(*, twist: bool, device: torch.device | None = None) -> Tensor:
    """S[k, a, r] defined by basis[a] * basis[r] = S[k, a, r] * basis[k].

    First algebra index is the weight component; implements left multiplication.
    """
    S = torch.zeros(NCOMP, NCOMP, NCOMP, dtype=torch.int8, device=device)
    for a in range(NCOMP):
        for r in range(NCOMP):
            S[a ^ r, a, r] = cocycle_sign(a, r, twist=twist)
    return S


def ternarize_weight_global_ste(w_shadow: Tensor, *, eps: float = 1e-8):
    """w_shadow: [O, I, 8]. One detached absmean scale over the whole tensor.

    Returns (w_effective, q_weight int8, gamma). Identity STE; no gradient
    through gamma, rounding, or clipping. No per-multivector scales.
    """
    if w_shadow.ndim != 3 or w_shadow.shape[-1] != NCOMP:
        raise ValueError(f"Expected [O, I, 8], got {tuple(w_shadow.shape)}")
    gamma = w_shadow.detach().abs().mean().clamp_min(eps)
    q_weight = torch.round(w_shadow.detach() / gamma).clamp(-1, 1).to(torch.int8)
    w_hard = gamma * q_weight.to(w_shadow.dtype)
    w_effective = w_shadow + (w_hard - w_shadow).detach()
    return w_effective, q_weight, gamma


def quantize_activation_per_token_ste(x: Tensor, *, eps: float = 1e-8):
    """x: [B, T, I, 8]. One symmetric int8 scale per token over I*8 coordinates."""
    if x.ndim != 4 or x.shape[-1] != NCOMP:
        raise ValueError(f"Expected [B, T, I, 8], got {tuple(x.shape)}")
    token_absmax = x.detach().abs().amax(dim=(-2, -1), keepdim=True)
    delta_x = token_absmax.clamp_min(eps) / QMAX_ACT
    q_activation = torch.round(x.detach() / delta_x).clamp(-QMAX_ACT, QMAX_ACT).to(torch.int8)
    x_hard = delta_x * q_activation.to(x.dtype)
    x_effective = x + (x_hard - x).detach()
    return x_effective, q_activation, delta_x


def algebra_linear_training(x: Tensor, w_shadow: Tensor, S: Tensor):
    """Training path. y[b,t,o,k] = sum_{i,a,r} S[k,a,r] W[o,i,a] x[b,t,i,r].

    The trainable object is [O, I, 8]; no independently trainable [O, I, 8, 8]
    parameter may exist anywhere in the codebase (mandatory unit test).
    """
    x_eff, q_x, delta_x = quantize_activation_per_token_ste(x)
    w_eff, q_w, gamma_w = ternarize_weight_global_ste(w_shadow)
    y = torch.einsum("kar,oia,btir->btok", S.to(dtype=x.dtype), w_eff, x_eff)
    return y, {
        "q_weight": q_w,
        "gamma_weight": gamma_w,
        "q_activation": q_x,
        "delta_activation": delta_x,
    }


@torch.no_grad()
def algebra_linear_integer_reference(q_activation, q_weight, delta_activation, gamma_weight, S):
    """Integer reference. Per-output-component bound: |acc| <= 8 * I * 127 (int32 safe)."""
    acc = torch.einsum(
        "kar,oia,btir->btok",
        S.to(torch.int32), q_weight.to(torch.int32), q_activation.to(torch.int32),
    )
    return acc.to(torch.float32) * gamma_weight.to(torch.float32) * delta_activation.to(torch.float32)


@torch.no_grad()
def algebra_linear_signed_permutation_reference(q_activation, q_weight, S):
    """Fixed signed-permutation implementation. Must be bit-exact with the
    sparse structure-tensor contraction (mandatory unit test)."""
    B, T, I, K = q_activation.shape
    O = q_weight.shape[0]
    acc = torch.zeros(B, T, O, K, dtype=torch.int32, device=q_activation.device)
    out_index = torch.arange(K, device=q_activation.device)
    for a in range(NCOMP):
        input_index = out_index ^ a
        sign = S[out_index, a, input_index].to(torch.int32)
        x_perm = q_activation[..., input_index].to(torch.int32)
        term = torch.einsum("oi,btik->btok", q_weight[:, :, a].to(torch.int32), x_perm)
        acc += term * sign.view(1, 1, 1, K)
    return acc


if __name__ == "__main__":
    # Convention self-check (full suite lives in verify_algebra.py / tests).
    SC = build_structure_tensor(twist=True)
    SD = build_structure_tensor(twist=False)
    LC = [SC[:, a, :].to(torch.int64) for a in range(8)]
    LD = [SD[:, a, :].to(torch.int64) for a in range(8)]
    for a in range(8):
        for b in range(8):
            fc = cocycle_sign(a, b, twist=True)
            assert torch.equal(LC[a] @ LC[b], fc * LC[a ^ b])
            assert torch.equal(LD[a] @ LD[b], LD[a ^ b])
    assert cocycle_sign(1, 2, twist=True) == 1 and cocycle_sign(2, 1, twist=True) == -1
    print("structure_tensor conventions: PASS")
