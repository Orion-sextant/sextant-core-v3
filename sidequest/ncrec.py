"""S1 exploratory models: multivector product-recurrence with twist on/off.

One code path; the ONLY difference between the twisted (Cl(3,0)) and untwisted
(R[(Z_2)^3]) cores is the structure tensor, imported read-only from the frozen
main-experiment algebra. CPU-only by design.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sextant.algebra import NCOMP, build_structure_tensor  # read-only reuse


class AlgebraScanCore(nn.Module):
    """h_t = rmsnorm_c( L_{f(x_t)} h_{t-1} ), h in R^[B, k, 8].

    variant "pure": update is exactly the algebra product (plus the central
    per-channel scalar normalization) — the clean theory object.
    variant "gated": adds a per-step SiLU gate after the product — practical.
    """

    def __init__(self, vocab: int, k: int = 16, *, twist: bool,
                 variant: str = "pure", seed: int = 0, pool: str = "final"):
        super().__init__()
        assert variant in ("pure", "gated")
        assert pool in ("final", "mean")
        g = torch.Generator().manual_seed(seed)
        self.k, self.variant, self.twist, self.pool = k, variant, twist, pool
        S = build_structure_tensor(twist=twist).to(torch.float32)
        self.register_buffer("S", S)                     # [out, weight, in]
        self.embed = nn.Parameter(torch.randn(vocab, k, NCOMP, generator=g) * 0.2)
        self.h0 = nn.Parameter(torch.zeros(k, NCOMP))
        with torch.no_grad():
            self.h0[:, 0] = 1.0                          # identity element
        if variant == "gated":
            self.gate = nn.Linear(k * NCOMP, k * NCOMP)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        h = self.h0.unsqueeze(0).expand(B, -1, -1).contiguous()
        acc = torch.zeros(B, self.k, NCOMP) if self.pool == "mean" else None
        for t in range(T):                               # sequential scan (CPU)
            a = self.embed[tokens[:, t]]                 # [B, k, 8]
            # left-multiply: h' = L_a h  via S[out, a_comp, in_comp]
            h = torch.einsum("oai,bka,bki->bko", self.S, a, h)
            # central scalar normalization per channel (preserves D-invariance)
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if self.variant == "gated":
                z = torch.nn.functional.silu(self.gate(h.reshape(B, -1)))
                h = z.reshape(B, self.k, NCOMP)
                h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if acc is not None:
                acc = acc + h
        out = (acc / T) if acc is not None else h
        # NOTE (theory): pool="mean" VOIDS the commutative-invariance theorem —
        # prefix products depend on order even in the untwisted algebra. Only
        # pool="final" keeps arm-D provably chance-bound on pure order tasks.
        return out.reshape(B, -1)                        # [B, k*8]


class ScalarProductRNN(nn.Module):
    """Commutative scalar null: h_t = norm(h_{t-1} * f(x_t)), elementwise."""

    def __init__(self, vocab: int, d: int = 128, *, seed: int = 0, pool: str = "final"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.pool = pool
        self.embed = nn.Parameter(torch.randn(vocab, d, generator=g) * 0.2 + 1.0)
        self.h0 = nn.Parameter(torch.ones(d))

    def forward(self, tokens):
        B, T = tokens.shape
        h = self.h0.unsqueeze(0).expand(B, -1).contiguous()
        acc = torch.zeros_like(h) if self.pool == "mean" else None
        for t in range(T):
            h = h * self.embed[tokens[:, t]]
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if acc is not None:
                acc = acc + h
        return (acc / T) if acc is not None else h


class BilinearHead(nn.Module):
    """Quadratic readout: split features into groups of 8, take within-group
    outer products, linear map to classes. On the FINAL state this preserves
    the commutative-invariance null (it is still a function of the final
    state); it tests whether order information is quadratically decodable."""

    def __init__(self, feat_dim: int, n_classes: int = 2):
        super().__init__()
        assert feat_dim % 8 == 0
        self.groups = feat_dim // 8
        self.head = nn.Linear(self.groups * 64, n_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0]
        gfeat = feat.reshape(B, self.groups, 8)
        outer = torch.einsum("bgi,bgj->bgij", gfeat, gfeat).reshape(B, -1)
        return self.head(outer)


class TinyGRU(nn.Module):
    """Order-capable non-algebra control."""

    def __init__(self, vocab: int, d: int = 48, *, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.out_dim = d

    def forward(self, tokens):
        _, h = self.gru(self.emb(tokens))
        return h.squeeze(0)


class Classifier(nn.Module):
    def __init__(self, core: nn.Module, feat_dim: int, n_classes: int = 2,
                 *, readout: str = "linear"):
        super().__init__()
        self.core = core
        if readout == "bilinear":
            self.head = BilinearHead(feat_dim, n_classes)
        else:
            self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, tokens):
        return self.head(self.core(tokens))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
