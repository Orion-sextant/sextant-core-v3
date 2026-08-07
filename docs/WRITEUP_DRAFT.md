# Algebra-valued ternary weights under matched storage budgets: a pre-registered four-arm comparison

**Status: DRAFT SKELETON.** Methods are written in full from the frozen
protocol (v3.1). Every results section is a placeholder keyed to the section-17
decision table; nothing may be written there except from the frozen decision
code's output after the gated comparative campaign.

## Abstract (placeholder)

[RESULT-KEYED: one sentence per hypothesis outcome; write only after the
decision code emits. State the emitted section-17 outcome verbatim, the two
budgets, the seed count, and the registered margin. No adjectives.]

## 1. Question and attribution frame

We ask a narrow question: **which structured constraint family on a language
model's linear maps gives the best validation performance per unit of
registered storage?** Four arms are compared under matched *actual serialized
bytes* of constrained-layer storage. Interpretation is narrow by construction
(protocol §1). The registered hypothesis labels are exactly:

- **H1 (matched-budget implementation):** at least one of B, C, D clears the
  registered margin against A at matched constrained-layer storage.
- **H2 (selected algebraic structure vs the Monarch control):** at least one of
  C, D clears the margin against B.
- **H3 (cocycle-twisted law):** C clears the margin against D.

Prohibited claim labels (recorded in the manifest): *generic structure*,
*algebra vs sparsity*, *noncommutativity proves order*. A C-over-D result
supports the complete cocycle-twisted operator family as an omnibus package —
not anticommutation alone, not equivariance to a linguistic symmetry group.

## 2. Novelty

The claimed-new elements are: (i) algebra-valued ternary weights acting by the
regular representation in a language model; (ii) the twisted/untwisted
matched-pair design, in which the sign cocycle is the only difference between
two arms; (iii) exhaustively verified exact rank catalogues used as train-time
diagnostics. Two literature searches (Claude, knowledge through early 2026;
Perplexity grounding pass, August 2026) found no prior instance of (i) or (ii).
The searches are correlated, so the claim is **provisional** and is stated as
"to our knowledge" pending the scheduled dedicated novelty search round.
Priority protection is procedural: the protocol, hashed and committed before
any result existed, is the timestamped priority artifact.

## 3. Methods

### 3.1 Arms

- **A — dense scalar ternary (null):** unconstrained scalar matrices;
  global-absmean ternary weights; per-token int8 activations; fp32 shadow
  weights with identity STE.
- **B — Monarch ternary (factorized control):** fixed two-factor Monarch
  parameterization (block-diagonal x permutation x block-diagonal), both
  factors ternarized under the same rule, never densely materialized; input
  quantized once; no intermediate requantization, normalization, or
  nonlinearity; int32 accumulator with a proven bound; scales applied after the
  second contraction.
- **C — Cl(3,0) regular ternary (twisted):** activations in 8-component
  multivector channels; each connection holds 8 ternarized coefficients; the
  induced 8x8 block is the left-regular representation of Cl(3,0).
- **D — R[(Z_2)^3] regular ternary (untwisted):** identical to C in dimensions,
  initialization draws, quantizer, contraction code, optimizer, storage, and
  left-action convention; the only difference is the structure tensor
  (cocycle off), one code path, `--twist {on,off}`.

### 3.2 Algebra (verified)

Group (Z_2)^3, bitmask basis order (1, e1, e2, e12, e3, e13, e23, e123);
cocycle B(x,y) = sum_{j<i} x_i y_j (mod 2), F = (-1)^B. Weight is the left
factor: (L_w)_{k,r} = F(k xor r, r) w_{k xor r}. All statements
machine-verified by exhaustive enumeration (all 512 cocycle triples; the
homomorphism on all 64 basis pairs per arm; the complete exact rank catalogue
over all 6,561 ternary blocks per arm), installed as mandatory unit tests.

### 3.3 Model policy (v3.1)

Pre-norm transformer, RoPE, depth 8, head_dim 64, d_ff = 4 d_model; embeddings
tied with the head, fp16, excluded from the constrained budget. v3.1 amendment
(owner-authorized pre-freeze): n_heads = floor(d_model/64 + 0.5) with half
rounding up; attention inner dim = n_heads*64; Q,K,V map d_model->inner, O maps
inner->d_model; the inner dim is divisible by 8 and no 8-component group
crosses a head boundary. Constrained set: Q, K, V, O, both MLP matrices.

### 3.4 Budgets and cells

Two frozen targets in actual serialized bytes: 2,500,000 and 10,000,000
(+/-1%). A deterministic solver (hashed) minimizes the byte gap; the packed
byte count equals the serializer's output exactly. Frozen cells:
A 356/725, B 1472/2800, C=D 1016/2040 (low/high d_model). Three storage
measures are reported per run: ideal ternary payload, actual packed constrained
bytes, total deployed bytes. Arms are bit-matched, not parameter-, compute-,
activation-, or latency-matched.

### 3.5 Data and training

FineWeb-Edu sample-10BT (revision pinned in the manifest), GPT-2 BPE (50257),
sequence length 1024, EOS-separated contiguous packing, causal masking only.
2.0e9 train tokens per comparative run; 2.0e7 held-out validation tokens.
AdamW (0.9, 0.95), eps 1e-8; weight decay 0.1 on constrained shadow tensors
only; clip 1.0; warmup 2%; cosine to 10% of peak; effective batch 0.5M tokens;
bf16 autocast with fp32 master weights. Checkpoint every 200M tokens; evaluate
every 100M; selection by best validation loss. Peak LR selected per cell by the
registered smoke stage (grid {1e-3, 2e-3, 4e-3}, 100M smoke tokens per point,
lowest smoke validation loss, tie to the lower LR, disjoint smoke data/seeds):

[TABLE: 8 selected peak LRs from the manifest — filled at campaign end]

### 3.6 Decision rule

delta_{X,Y,b,s} = (L_Y - L_X)/L_Y on seed-paired runs. X beats Y at budget b
iff the three-seed mean delta >= 0.02 and delta > 0 for all seeds. Practical
equivalence iff |delta| < 0.02 for all seeds. No p-values for the loss
decision. No scale-up unless the same directional conclusion holds at both
budgets. The frozen decision code (hashed in the manifest) emits exactly one
section-17 outcome or "unresolved."

### 3.7 Probe and diagnostics

Frozen seeded generator: 20 templates x 10 variants over a closed 24-fact
schema; ridge readout (lambda 1.0, standardized) on mean-pooled block-4 output
at the selected checkpoint; macro-F1; template-ID split 12/8; 10,000 paired
cluster-bootstrap replicates (descriptive only); registered minimum effect
+3 macro-F1 points. Rank diagnostics compare hard exact ranks against the
verified catalogue and paired movement of effective rank from initialization;
effective rank is spectral geometry, not coefficient utilization.

## 4. Results

**[PLACEHOLDER — nothing below may be filled before the freeze gate and the
completed campaign; all numbers come from the frozen decision code and the
run-summary ledger.]**

### 4.1 Emitted decision outcome
[One of: C_wins_scale_C | regular_wins_scale_D | D_wins_scale_D |
B_wins_no_algebra_claim | A_wins_constraints_harmful |
factorized_control_wins | unresolved — with the §17 statement verbatim.]

### 4.2 Primary: loss vs actual constrained bytes
[FIGURE: loss_vs_constrained_bytes.png + per-cell three-seed table.]

### 4.3 Co-primary: loss vs total deployed bytes
[FIGURE: loss_vs_total_bytes.png + table.]

### 4.4 Probe (claim-decomposition macro-F1)
[Point estimates with cluster-bootstrap CIs, descriptive only.]

### 4.5 Diagnostics
[Learned-block rank movement vs the verified catalogue and the paired
initialization reference; grade-mass and participation summaries by arm,
budget, layer.]

### 4.6 Engineering measurements
[Actual packed bytes, total bytes, integer ops, permutations, peak memory,
tokens/s, latency — reported separately from the primary comparison.]

## 5. Limitations

Bit-matched but not compute- or latency-matched; two budgets at toy scale;
three seeds with a registered margin instead of p-values; single corpus and
tokenizer; the LR grid is three points; conclusions are restricted to the
selected implementations under the narrowed labels of §1. Every outcome —
including the negative ones — ships a model and publishes a result.

## Reproducibility

Protocol v3.1 (hashed), manifest with every artifact hash, frozen decision
code, bit-exact reference kernels, exhaustive algebra verifier, append-only
run ledger, and the pre-registered queue are all in the repository; the freeze
tag is the public timestamp.
