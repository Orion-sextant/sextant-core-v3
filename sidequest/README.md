# Side-Quest S1 (EXPLORATORY): the noncommutative sequential core

**Status: exploratory. NOT part of the pre-registered Sextant Core protocol.**
Nothing in this folder is covered by the freeze, contributes to the section-17
decision, or may be cited as a pre-registered result. It exists because the
main experiment deliberately does NOT test linguistic-order encoding
(PROTOCOL_v3.md §1's non-claims) — and a *sequential* architecture is where the
cocycle's noncommutativity bears directly on order.

## Hypothesis (exploratory)

In a multivector product-recurrence h_t = L_{f(x_t)} h_{t-1}, the state is an
ordered product of algebra elements. If the algebra is the untwisted
commutative R[(Z_2)^3] (arm-D algebra), the pure product is **provably
order-invariant**: any permutation of the inputs yields the same state, so the
core cannot solve order-discrimination tasks above chance from the product
alone (scalar normalization is central and preserves this). The twisted Cl(3,0)
algebra (arm-C) anticommutes on generators, so order survives in the state.
S1 asks whether that theoretical separation shows up in practice, and at what
cost in optimization difficulty.

## Design

- **Core:** h ∈ R^(k×8), k multivector channels; per token, an embedding maps
  to one 8-coefficient algebra element per channel; update is left-regular
  multiplication (the SAME verified structure tensor as the main experiment,
  imported read-only from `sextant.algebra`), followed by per-channel scalar
  RMS normalization (central, so it preserves the D-invariance theorem).
- **Matched pair:** one code path, `twist` on/off — the S1 echo of arms C/D.
- **Baselines:** scalar elementwise-product RNN (also commutative — a second
  null), and a small GRU (order-capable non-algebra control), parameter-matched
  within ~10%.
- **Tasks (synthetic, no external data):**
  - `ab_order`: does token A appear before token B? (identical multiset — pure
    order; commutative product-cores are chance-bound by construction)
  - `swap_parity`: parity of adjacent transpositions applied to a fixed base
    sequence (harder order structure)
- **Variants:** `pure` (product only — clean theory) and `gated` (per-step
  nonlinearity after the product — practical variant where even commutative
  cores may leak order via the nonlinearity; measures how much the algebra
  helps *beyond* that leak).

## Resource contract

CPU only (`torch.set_num_threads(8)` of 32; GPU never touched), tiny models
(<1M params), results in `sidequest/results/*.json` + a markdown table.
Runs alongside the GPU smoke campaign without competing with it.

## Non-contamination rules

- Never imports from, writes to, or modifies frozen protocol files; structure
  tensors are imported read-only.
- Never reads smoke or comparative data, bins, or ledgers.
- Results live only in this folder; the run ledger here is
  `sidequest/results/ledger_s1.yaml`, distinct from the protocol ledgers.

## v0 + follow-up results (pure variant, 3 seeds, ab_order)

| pool / readout | C (twisted) | D (untwisted) | scalar-prod | GRU |
|---|---|---|---|---|
| final / linear   | 0.497 | 0.503 | 0.495 | 1.000 |
| final / bilinear | **0.599** | 0.501 | 0.491 | — |
| mean / linear    | 0.536 | 0.887 | 0.980 | — |
| mean / bilinear  | 0.551 | 0.877 | 0.988 | — |

Findings:
1. **Invariance theorem verified twice**: with final-state readout the untwisted
   core is at exact chance under BOTH linear and bilinear heads (0.503, 0.501) —
   no function of its final state can see order, as proved.
2. **The twist's order information is real and quadratically decodable**: C at
   0.599 under final/bilinear — the only configuration in which final-state
   order decoding is possible at all, and only the cocycle provides it.
3. **Mean-pooling inverts the ranking**: commutative cores become excellent
   order-timestamp accumulators via prefix products (0.88–0.99), while the
   twisted core's fast mixing washes out temporal means (0.54–0.55).
4. A tiny GRU still dominates everything at this scale — gating, not algebra,
   is the cheap win for order per se.

Interpretation: the cocycle trades **temporal smoothness for state mixing** —
order survives in the final state (retrievable nonlinearly) but running means
are destroyed. Natural next architecture: **mixed-algebra channels** (some
twisted, some untwisted per layer) so the model has both stable accumulators
and order-carrying mixers; readout over {mean, final} jointly.
