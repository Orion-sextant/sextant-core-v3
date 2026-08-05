"""Section 13 budget test: verification within tolerance.

Two policies are checked explicitly because the protocol has an internal
conflict (flagged to the owner): the multiples-of-8 solver hits +/-1% for all
eight cells, but head_dim=64 integrality (d_model % 64 == 0) then fails for most
cells. This test pins BOTH facts so neither can silently regress before the
owner resolves the cell policy.
"""
from sextant.packer.solver import solve_all, TOLERANCE


def test_multiples_of_eight_meet_tolerance():
    sols = solve_all(require_integer_heads=False)
    for cell, s in sols.items():
        assert abs(s.rel_err) <= TOLERANCE, (cell, s.rel_err)
        assert s.d_model % 8 == 0


def test_head_integrality_conflict_is_present():
    # Documented conflict: not all multiples-of-8 optima give integer 64-dim heads.
    sols = solve_all(require_integer_heads=False)
    non_integer = [c for c, s in sols.items() if not s.integer_heads]
    assert non_integer, "expected the flagged head-integrality conflict to be present"


def test_integer_head_policy_cannot_meet_all_cells():
    # Under integer 64-dim heads, at least one cell misses +/-1% (the conflict).
    sols = solve_all(require_integer_heads=True)
    missed = [c for c, s in sols.items() if not s.within_tolerance]
    assert missed, "if this is empty, the head/byte conflict resolved itself — revisit the manifest"
