# P0 conflict: budget grid vs 64-dim head integrality (owner decision required)

**Status:** blocks freezing the eight model cells (build item 6) and therefore the
LR fairness stage (item 9). Flagged per brief rule 5 ("No silent deviations")
and the entry-point instruction ("the protocol governs; stop and flag the
conflict instead of resolving it silently").

## The conflict

PROTOCOL_v3.md section 8 states three things that cannot all hold:

1. The solver searches **d_model over multiples of 8** and must land within
   **±1%** of the actual-packed-byte target (also readiness criterion, §22).
2. **head_dim = 64** is fixed. A buildable transformer therefore needs
   `d_model % 64 == 0` (integer head count) — unless the attention inner
   dimension is decoupled from `d_model`.
3. The illustrative anchors put **A-low ≈ 360** and **A-high ≈ 728**, which are
   *not* multiples of 64 — implying the authors sized `d_model` on the
   multiples-of-8 grid without enforcing integer 64-dim heads.

The solver (byte count == packer output, verified) gives:

| cell | mult-of-8 d_model | bytes | rel_err | heads (d/64) | int heads? | int-head d_model | int-head rel_err | within 1%? |
|------|------|------|------|------|------|------|------|------|
| A_low | 360 | 2,495,060 | -0.20% | 5.625 | False | 384 | +13.57% | False |
| A_high | 720 | 9,960,086 | -0.40% | 11.250 | False | 704 | -4.76% | False |
| B_low | 1472 | 2,508,389 | +0.34% | 23.000 | True | 1472 | +0.34% | True |
| B_high | 3320 | 9,999,617 | -0.00% | 51.875 | False | 4096 | -3.35% | False |
| C_low | 1016 | 2,486,645 | -0.53% | 15.875 | False | 1024 | +1.02% | False |
| C_high | 2040 | 9,996,727 | -0.03% | 31.875 | False | 2048 | +0.76% | True |
| D_low | 1016 | 2,486,693 | -0.53% | 15.875 | False | 1024 | +1.02% | False |
| D_high | 2040 | 9,996,775 | -0.03% | 31.875 | False | 2048 | +0.76% | True |

- **Multiples-of-8 policy:** all 8 cells within ±1%, but 7 of 8 have a
  fractional head count → not buildable with fixed 64-dim heads.
- **Integer-64-head policy:** only 3 of 8 cells (B_low, C_high, D_high) meet
  ±1%; A_low misses by +13.6%, A_high −4.8%, B_high −3.4%, C/D_low +1.02%.

## Options for the owner

- **Option 1 — decouple attention inner dim from d_model (recommended).**
  Keep head_dim = 64, choose `n_heads = round(d_model/64)` so the attention
  inner dim is `n_heads*64` (Q/K/V become d_model→inner, O inner→d_model,
  slightly rectangular). d_model stays on the multiples-of-8 grid; all 8 cells
  hit ±1%; matches the protocol's own anchors. Cost: Q/K/V/O are no longer
  exactly d×d, so byte accounting is recomputed (still deterministic), and the
  current harness (which assumes n_heads = d_model/64 with d×d projections)
  needs a small change to the attention projection shapes. This is a P0
  protocol clarification and bumps the version.

- **Option 2 — require d_model % 64 == 0 and widen the byte tolerance** for the
  non-conforming cells (or re-target the byte budgets) so the readiness
  criterion is met at a stated, larger tolerance. Preserves d×d projections;
  changes the registered ±1% tolerance (a pre-registration change).

- **Option 3 — keep d_model % 64 == 0 and accept documented near-misses**,
  explicitly relaxing §22's "all eight cells meet the byte tolerance". Least
  disruptive to code, largest deviation from the registered spec.

Whichever is chosen, it is a change to a frozen pre-registration value and
therefore bumps the protocol version before the freeze gate.

The solver already supports both policies via
`solve_all(require_integer_heads=...)`; test_budget.py pins the conflict so it
cannot silently regress.
