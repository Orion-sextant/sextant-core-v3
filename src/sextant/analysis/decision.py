"""Frozen decision rule and outcome table (PROTOCOL_v3.md sections 16-17).

Inputs: section-20 ledger rows for the 24 comparative runs (or their
replacements per section 21). The per-run loss used is
``validation_loss_selected_checkpoint`` (checkpoint selection: best validation
loss, section 10).

Section 16:
  delta_{X,Y,b,s} = (L_{Y,b,s} - L_{X,b,s}) / L_{Y,b,s}
  "X beats Y at b"  iff mean_s(delta) >= 0.02 AND delta > 0 for all three seeds.
  "X ~ Y at b" (practical equivalence) iff |delta| < 0.02 for all three seeds.
  Everything else is unresolved at that pairing. No p-values.

Section 17 emits exactly one outcome (or "unresolved"). The checks are
evaluated in the frozen order below; the first that matches wins. Any
budget-crossed, seed-inconsistent, margin-incomplete, or unlisted ordering is
unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass

ARMS = ("A", "B", "C", "D")
BUDGETS = ("low", "high")
SEEDS_REQUIRED = 3
MARGIN = 0.02

OUTCOMES = {
    "C_wins_scale_C": "C clears D, B, A at both budgets: twisted arm wins; scale C.",
    "regular_wins_scale_D": ("C and D practically equivalent at both budgets, both clear "
                              "B and A: regular-algebra structure wins without evidence the "
                              "twist earns its complexity; scale D."),
    "D_wins_scale_D": "D clears C, B, A at both budgets: untwisted wins; twist harmful at this scale.",
    "B_wins_no_algebra_claim": "B clears C, D, A at both budgets: Monarch control wins; no algebraic-advantage claim.",
    "A_wins_constraints_harmful": "A clears every structured arm at both budgets: registered constraints harmful at this scale.",
    "factorized_control_wins": "C and D tie while B clears A and both algebra arms: infer nothing beyond B.",
    "unresolved": "Budget-crossed, seed-inconsistent, margin-incomplete, or unlisted ordering: publish the mixed result; do not scale.",
}


@dataclass(frozen=True)
class LossTable:
    """losses[arm][budget][seed] -> validation loss (selected checkpoint)."""
    losses: dict

    @classmethod
    def from_ledger_rows(cls, rows: list[dict]) -> "LossTable":
        t: dict = {a: {b: {} for b in BUDGETS} for a in ARMS}
        for r in rows:
            if r.get("status") != "complete":
                continue
            t[r["arm"]][r["budget"]][int(r["seed"])] = float(
                r["validation_loss_selected_checkpoint"])
        for a in ARMS:
            for b in BUDGETS:
                if len(t[a][b]) != SEEDS_REQUIRED:
                    raise ValueError(
                        f"cell {a}_{b}: need exactly {SEEDS_REQUIRED} complete seeds, "
                        f"have {sorted(t[a][b])}")
        return cls(t)

    def seeds(self, arm: str, budget: str) -> list[int]:
        return sorted(self.losses[arm][budget])


def deltas(t: LossTable, x: str, y: str, b: str) -> list[float]:
    """delta_{X,Y,b,s} for the seed-paired runs, in seed order."""
    out = []
    for s in t.seeds(y, b):
        ly = t.losses[y][b][s]
        lx = t.losses[x][b][s]
        out.append((ly - lx) / ly)
    return out


def beats(t: LossTable, x: str, y: str, b: str) -> bool:
    d = deltas(t, x, y, b)
    return (sum(d) / len(d)) >= MARGIN and all(v > 0 for v in d)


def equivalent(t: LossTable, x: str, y: str, b: str) -> bool:
    return all(abs(v) < MARGIN for v in deltas(t, x, y, b))


def beats_both(t: LossTable, x: str, y: str) -> bool:
    return all(beats(t, x, y, b) for b in BUDGETS)


def equivalent_both(t: LossTable, x: str, y: str) -> bool:
    return all(equivalent(t, x, y, b) for b in BUDGETS)


def decide(t: LossTable) -> dict:
    """Emit exactly one section-17 outcome (or unresolved), with the evidence."""
    ev = {
        f"{x}_beats_{y}_{b}": beats(t, x, y, b)
        for x in ARMS for y in ARMS if x != y for b in BUDGETS
    }
    ev.update({f"{x}_equiv_{y}_{b}": equivalent(t, x, y, b)
               for x, y in (("C", "D"),) for b in BUDGETS})

    def out(key):
        return {"outcome": key, "statement": OUTCOMES[key], "evidence": ev}

    if all(beats_both(t, "C", y) for y in ("D", "B", "A")):
        return out("C_wins_scale_C")
    if (equivalent_both(t, "C", "D")
            and all(beats_both(t, "C", y) for y in ("B", "A"))
            and all(beats_both(t, "D", y) for y in ("B", "A"))):
        return out("regular_wins_scale_D")
    if all(beats_both(t, "D", y) for y in ("C", "B", "A")):
        return out("D_wins_scale_D")
    # Section 17 lists "B clears C, D, A" before "C/D tie while B clears all";
    # the tie case is a strict refinement of the former, so it is checked FIRST
    # here — otherwise it would be unreachable. This preserves every listed
    # outcome's reachability without changing any decision's substance (both
    # are B-wins cells with different inference statements).
    if (equivalent_both(t, "C", "D")
            and all(beats_both(t, "B", y) for y in ("A", "C", "D"))):
        return out("factorized_control_wins")
    if all(beats_both(t, "B", y) for y in ("C", "D", "A")):
        return out("B_wins_no_algebra_claim")
    if all(beats_both(t, "A", y) for y in ("B", "C", "D")):
        return out("A_wins_constraints_harmful")
    return out("unresolved")
