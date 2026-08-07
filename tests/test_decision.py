"""Fixture tests for the frozen section 16-17 decision code and the two
loss-versus-bytes plots. Every decision-table cell is exercised with a
synthetic run-summary ledger; the emitted outcome must match the fixture.
No real run data is involved.
"""
import itertools

import pytest

from sextant.analysis.decision import LossTable, decide, deltas, beats, equivalent

ARMS = ("A", "B", "C", "D")
BUDGETS = ("low", "high")
SEEDS = (1, 2, 3)


def make_rows(loss_fn):
    """Synthetic section-20 rows: loss_fn(arm, budget, seed) -> loss."""
    rows = []
    for a, b, s in itertools.product(ARMS, BUDGETS, SEEDS):
        rows.append({
            "arm": a, "budget": b, "seed": s, "status": "complete",
            "validation_loss_selected_checkpoint": loss_fn(a, b, s),
            "actual_constrained_bytes": 2_500_000 if b == "low" else 10_000_000,
            "total_deployed_model_bytes": 40_000_000 if b == "low" else 150_000_000,
        })
    return rows


def table(loss_fn):
    return LossTable.from_ledger_rows(make_rows(loss_fn))


BASE = 3.0
STEP = 0.04 * BASE   # 4% of base — comfortably over the 2% margin


def losses_ordered(order):
    """order: best-to-worst arms; each 4% apart -> clean margins everywhere."""
    def fn(a, b, s):
        return BASE + order.index(a) * STEP + 0.001 * s   # tiny seed jitter, sign-stable
    return fn


# ---- each win outcome ------------------------------------------------------
def test_C_wins():
    assert decide(table(losses_ordered(["C", "D", "B", "A"])))["outcome"] == "C_wins_scale_C"


def test_D_wins():
    assert decide(table(losses_ordered(["D", "C", "B", "A"])))["outcome"] == "D_wins_scale_D"


def test_B_wins():
    assert decide(table(losses_ordered(["B", "C", "D", "A"])))["outcome"] == "B_wins_no_algebra_claim"


def test_A_wins():
    assert decide(table(losses_ordered(["A", "B", "C", "D"])))["outcome"] == "A_wins_constraints_harmful"


# ---- practical equivalence outcomes ---------------------------------------
def test_regular_algebra_wins_CD_equivalent():
    def fn(a, b, s):
        if a in ("C", "D"):
            return BASE + (0.001 if a == "D" else 0.0)     # < 2% apart: equivalent
        return BASE + STEP * (2 if a == "B" else 3)         # both algebra arms clear B, A
    assert decide(table(fn))["outcome"] == "regular_wins_scale_D"


def test_factorized_control_wins_B_over_tied_CD():
    def fn(a, b, s):
        if a == "B":
            return BASE
        if a in ("C", "D"):
            return BASE + STEP + (0.001 if a == "D" else 0.0)  # tied C/D, both behind B
        return BASE + 2 * STEP                                  # A worst
    assert decide(table(fn))["outcome"] == "factorized_control_wins"


# ---- unresolved family -----------------------------------------------------
def test_budget_crossed_is_unresolved():
    def fn(a, b, s):
        if a == "C":
            return BASE if b == "low" else BASE + STEP      # C wins low, loses high
        if a == "D":
            return BASE + STEP if b == "low" else BASE
        return BASE + 2 * STEP                              # A, B out of the running
    assert decide(table(fn))["outcome"] == "unresolved"


def test_seed_inconsistent_is_unresolved():
    def fn(a, b, s):
        if a == "C":
            # seeds 1,2 much better; seed 3 slightly WORSE than D -> sign flip
            return BASE - 2 * STEP if s < 3 else BASE + 0.001
        if a == "D":
            return BASE
        return BASE + 3 * STEP
    t = table(fn)
    d = deltas(t, "C", "D", "low")
    assert sum(d) / len(d) >= 0.02 and not all(v > 0 for v in d)  # mean passes, sign fails
    assert not beats(t, "C", "D", "low")
    assert decide(t)["outcome"] == "unresolved"


def test_margin_incomplete_is_unresolved():
    def fn(a, b, s):
        # C beats D and B clearly but only ~1% ahead of A: margin incomplete
        if a == "C":
            return BASE
        if a == "A":
            return BASE * 1.01
        return BASE + 2 * STEP
    assert decide(table(fn))["outcome"] == "unresolved"


def test_exactly_one_outcome_emitted():
    for order in (["C", "D", "B", "A"], ["D", "C", "B", "A"], ["A", "B", "C", "D"]):
        res = decide(table(losses_ordered(order)))
        assert res["outcome"] in {
            "C_wins_scale_C", "regular_wins_scale_D", "D_wins_scale_D",
            "B_wins_no_algebra_claim", "A_wins_constraints_harmful",
            "factorized_control_wins", "unresolved"}
        assert isinstance(res["statement"], str) and res["evidence"]


def test_incomplete_seeds_raise():
    rows = make_rows(lambda a, b, s: BASE)
    rows = [r for r in rows if not (r["arm"] == "B" and r["seed"] == 2)]
    with pytest.raises(ValueError):
        LossTable.from_ledger_rows(rows)


# ---- section 16 arithmetic -------------------------------------------------
def test_delta_formula_and_equivalence_band():
    t = table(losses_ordered(["C", "D", "B", "A"]))
    d = deltas(t, "C", "D", "low")
    ly = t.losses["D"]["low"][1]
    lx = t.losses["C"]["low"][1]
    assert abs(d[0] - (ly - lx) / ly) < 1e-12
    assert not equivalent(t, "C", "D", "low")      # 4% apart: not equivalent
    assert equivalent(t, "C", "C", "low")          # self-delta zero


# ---- plots -----------------------------------------------------------------
def test_plots_render_from_synthetic_ledger(tmp_path):
    from sextant.analysis.plots import plot_primary, plot_coprimary
    rows = make_rows(losses_ordered(["C", "D", "B", "A"]))
    p1 = plot_primary(rows, tmp_path)
    p2 = plot_coprimary(rows, tmp_path)
    assert p1.exists() and p1.stat().st_size > 10_000
    assert p2.exists() and p2.stat().st_size > 10_000
