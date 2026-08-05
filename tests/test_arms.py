"""Section 13 arm tests: C/D path identity (differ only in the structure
tensor), Monarch non-materialization, Monarch primary-path linearity.
"""
import torch

from sextant.arms.arm_b_monarch import MonarchTernaryLinear
from sextant.arms.arm_cd_algebra import AlgebraTernaryLinear
from sextant.arms.quant import ternarize_weight_global
from sextant.algebra import build_structure_tensor


def test_cd_path_identity_only_structure_tensor_differs():
    g = torch.Generator().manual_seed(3)
    c = AlgebraTernaryLinear(64, 64, twist=True, generator=g).double()
    g = torch.Generator().manual_seed(3)
    d = AlgebraTernaryLinear(64, 64, twist=False, generator=g).double()
    # identical init draws, identical code path/class, different S
    assert torch.equal(c.coeff, d.coeff)
    assert not torch.equal(c.S, d.S)
    x = torch.randn(2, 7, 64, dtype=torch.float64)
    yc, yd = c(x), d(x)
    assert not torch.allclose(yc, yd)
    # swapping C's structure tensor for D's reproduces D's forward exactly:
    c.S.copy_(build_structure_tensor(twist=False))
    assert torch.allclose(c(x), yd, atol=1e-12)


def test_monarch_non_materialization():
    m = MonarchTernaryLinear(64, 256)
    shapes = [tuple(p.shape) for _, p in m.named_parameters()]
    # no dense (out,in) or (in,out) parameter is ever formed
    assert all(len(s) == 3 for s in shapes)          # only factor tensors (g,·,·)
    assert (256, 64) not in shapes and (64, 256) not in shapes


def test_monarch_primary_path_linearity():
    m = MonarchTernaryLinear(64, 256)
    _, q1, _ = ternarize_weight_global(m.w1)
    _, q2, _ = ternarize_weight_global(m.w2)

    def core(qx_int):  # the integer primary path: two factor contractions + permutation
        b = qx_int.shape[0]
        xr = qx_int.reshape(b, m.g, m.a)
        o1 = torch.einsum("gij,bgj->bgi", q1.to(torch.int64), xr)
        o1 = m._permute(o1, b)
        return torch.einsum("gci,bgi->bgc", q2.to(torch.int64), o1)

    a = torch.randint(-127, 128, (4, 64), dtype=torch.int64)
    b = torch.randint(-127, 128, (4, 64), dtype=torch.int64)
    assert torch.equal(core(a + b), core(a) + core(b))  # linear, no nonlinearity in path
