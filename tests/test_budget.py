"""Section 13 budget test (v3.1): all eight cells verify within tolerance.

Under the v3.1 amendment (attention inner dim decoupled from d_model, per-arm
search grid), every cell lands within ±1% and every attn_inner is divisible
by 8. The frozen cells in manifest.yaml must agree with the solver.
"""
import re
from pathlib import Path

import yaml

from sextant.packer.serializer import _attn_inner, cell_actual_bytes
from sextant.packer.solver import solve_all, TOLERANCE
from sextant.paths import REPO_ROOT


def test_all_eight_cells_within_tolerance():
    for cell, s in solve_all().items():
        assert abs(s.rel_err) <= TOLERANCE, (cell, s.rel_err)


def test_attn_inner_divisible_by_8():
    for cell, s in solve_all().items():
        assert s.attn_inner % 8 == 0, (cell, s.attn_inner)
        assert s.attn_inner == _attn_inner(s.d_model)


def test_cd_dims_are_multiples_of_8_and_paired():
    sols = solve_all()
    for cell in ("C_low", "C_high", "D_low", "D_high"):
        assert sols[cell].d_model % 8 == 0
    # C and D are dimensionally identical (only the structure tensor differs, §5)
    assert sols["C_low"].d_model == sols["D_low"].d_model
    assert sols["C_high"].d_model == sols["D_high"].d_model


def test_frozen_manifest_cells_match_solver():
    manifest = yaml.safe_load((REPO_ROOT / "manifest.yaml").read_text())
    cells = manifest["budgets"]["model_cells"]
    sols = solve_all()
    for cell, s in sols.items():
        m = cells[cell]
        assert m["d_model"] == s.d_model, (cell, m["d_model"], s.d_model)
        assert m["actual_constrained_bytes"] == s.actual_bytes, cell
        assert m["attn_inner"] == s.attn_inner and m["n_heads"] == s.n_heads
        # solver byte count equals the real packed byte count for this cell
        assert cell_actual_bytes(s.arm, s.d_model) == s.actual_bytes
