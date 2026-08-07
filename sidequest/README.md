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
