# Sextant Core: Protocol v3.0

**Status: PRE-REGISTRATION DRAFT.** Becomes **FROZEN** when the readiness criterion in section 22 passes and the owner authorizes the freeze gate. Supersedes v2 (Claude draft), v2.1 (ChatGPT consolidation and revision ledger), and v2.2 (Claude verification and freeze decisions). All P0 revisions from v2.1 Part II are applied inline below; P1 items are applied; P2 items are scheduled in section 22. Where a value can only exist after implementation (a hash, a solved dimension, a selected learning rate) it is marked `TO_FILL` and listed in the freeze checklist.

---

## 1. Question and attribution frame

A four-arm, matched-budget toy experiment: **which structured constraint family on a language-model linear map gives the best validation performance per unit of registered storage?**

Interpretation is narrow by construction:

- A vs B compares one selected ternary Monarch implementation against one selected dense scalar-ternary baseline.
- B vs C/D compares the selected regular-algebra families against that particular factorized control.
- C vs D compares one fixed cocycle-twisted left-regular family against the corresponding untwisted XOR-convolution family, with everything else identical.

The experiment does **not** claim to isolate: structure in general; algebra versus every sparse or factorized family; noncommutativity alone; anticommutation alone; linguistic-order encoding; equivariance to an a priori language symmetry group. A C-over-D result supports the complete cocycle-twisted operator family as an omnibus package.

## 2. Novelty and priority

The claimed-new elements are: (i) algebra-valued ternary weights acting by the regular representation in a language model; (ii) the twisted/untwisted matched-pair design, in which the sign cocycle is the only difference between two arms; (iii) exhaustively verified exact rank catalogues used as train-time diagnostics. Two literature searches (Claude, knowledge through early 2026; Perplexity grounding pass, August 2026) found no prior instance of (i) or (ii). The searches are correlated, so the claim is **provisional**. Priority protection is procedural, not rhetorical: this protocol, hashed and committed (and optionally posted as a public preregistration), is the priority artifact. A dedicated novelty search round is scheduled before any public claim stronger than "to our knowledge."

## 3. Mathematical spine (verified)

Every statement in this section has been machine-verified by exhaustive check; see `reference/verify_algebra.py`.

### 3.1 Basis

Group G = (Z_2)^3. Fixed bitmask basis order `(1, e1, e2, e12, e3, e13, e23, e123)`, masks 0..7, with e1, e2, e3 on bits 0, 1, 2. Grade map `(0,1,1,2,1,2,2,3)`. Grade sets G0={0}, G1={1,2,4}, G2={3,5,6}, G3={7}; grade dimensions (1,3,3,1).

### 3.2 Cocycle

B(x,y) = sum over j<i of x_i * y_j (mod 2), F(x,y) = (-1)^B(x,y). The cocycle identity F(x,y)F(x xor y, z) = F(y,z)F(x, y xor z) holds on all 512 triples (verified), so u_x * u_y = F(x,y) u_(x xor y) is associative. Generators square to +1 and anticommute (verified). The twisted algebra is Cl(3,0), isomorphic to M_2(C). With F = 1 identically, the algebra is the commutative group algebra R[(Z_2)^3], whose regular matrices implement XOR convolution and are diagonalized by the Walsh-Hadamard transform. Note: this is an XOR group-convolution family, **not** an ordinary Z_8 circulant family; block-circulant layers are precedent-adjacent, not identical.

### 3.3 Left-regular blocks

Weight is always the **left** factor: y = L_w x, with (L_w)_{k,r} = F(k xor r, r) * w_(k xor r). The explicit 8x8 signed templates for arm C and arm D are frozen in `reference/structure_tensor.py` and verified entry-by-entry against this formula (64/64 for each arm). The homomorphism L_a L_b = L_(a*b) holds for all 64 basis pairs in both arms (verified).

## 4. Verified exact rank catalogue

Exhaustive enumeration of all 3^8 = 6561 ternary coefficient vectors per arm, reproduced exactly by `reference/verify_algebra.py` (this doubles as mandatory unit test `test_rank_enumeration`).

**Total counts.**

| Exact rank | Arm C: Cl(3,0) | Arm D: R[(Z_2)^3] |
|---:|---:|---:|
| 0 | 1 | 1 |
| 1 | 0 | 16 |
| 2 | 0 | 112 |
| 4 | 672 | 672 |
| 5 | 0 | 896 |
| 6 | 0 | 1344 |
| 8 | 5888 | 3520 |

**Conditioned on nonzero trits N_nz** (each row totals C(8,k) * 2^k).

| N_nz | Arm C | Arm D |
|---:|---|---|
| 0 | r0:1 | r0:1 |
| 1 | r8:16 | r8:16 |
| 2 | r4:48, r8:64 | r4:112 |
| 3 | r8:448 | r8:448 |
| 4 | r4:144, r8:976 | r2:112, r5:896, r8:112 |
| 5 | r8:1792 | r8:1792 |
| 6 | r4:384, r8:1408 | r4:448, r6:1344 |
| 7 | r8:1024 | r8:1024 |
| 8 | r4:96, r8:160 | r1:16, r4:112, r8:128 |

Singular-value structure of arm C blocks (verified on all 6561): rank-4 blocks have four equal nonzero singular values; rank-8 blocks have two values with multiplicity four each. Arm D block rank equals the number of nonzero Walsh coefficients (verified on all 6561).

These tables serve three distinct roles and must be labeled per use: complete algebraic catalogue; uniform counting reference conditioned on N_nz; they are **not** automatically the initialization-induced statistical null (that reference is computed empirically under the frozen initializer and quantizer). Checkpoint interpretation primarily uses paired movement from initialization within each arm. No common absolute rank threshold across arms is valid.

## 5. Arms

- **Arm A, dense scalar ternary.** Unconstrained scalar matrices; global-absmean ternary weights; per-token int8 activations; high-precision shadow weights in training; packed storage measured under the shared serializer.
- **Arm B, Monarch ternary (primary path).** Fixed two-factor Monarch parameterization, block-diagonal x permutation x block-diagonal; both factors ternarized under the same rule; remains factored in training and deployment, never densely materialized; input quantized once; **no** intermediate requantization, normalization, or nonlinearity in the primary control; int32 intermediate accumulator with a proven bound; factor scales and activation scale applied after the second contraction. An int8-intermediate variant exists only as the P2 deployment ablation.
- **Arm C, Clifford regular ternary.** Activations grouped into 8-component multivector channels; each connection holds 8 shadow coefficients, ternarized before the regular contraction; the induced 8x8 block is constrained to the left-regular representation of Cl(3,0); implementable as fixed signed permutations, additions, and global rescaling.
- **Arm D, untwisted XOR group-algebra ternary.** Identical to C in dimensions, layout, initialization draws, quantizer, scale granularity, contraction code, optimizer, storage, and left-action convention. The only difference is the structure tensor: one code path, `--twist {on,off}`.

## 6. Hypotheses

- **H1 (matched-budget implementation):** at least one of B, C, D clears the registered margin against A at matched constrained-layer storage. This compares complete budget-solved implementations; it does not isolate structure independently of the width, activation, compute, and total-size consequences each implementation entails.
- **H2 (selected algebraic structure vs the Monarch control):** at least one of C, D clears the margin against B. Applies only to these selected implementations.
- **H3 (cocycle-twisted law):** C clears the margin against D. Supports the twisted family as an omnibus package only; see section 1 for the non-claims.

## 7. Budgets and storage accounting

Two frozen constrained-layer storage targets, stated in **actual serialized bytes**:

- low: 2,500,000 bytes actual packed constrained storage;
- high: 10,000,000 bytes actual packed constrained storage;
- tolerance: plus or minus 1 percent per cell.

Report three measures for every run: (1) ideal ternary payload N_trit * log2(3); (2) actual packed constrained bytes including trit encoding, padding, alignment, scales, factor metadata, permutation or lookup tables; (3) total deployed model bytes including embeddings, head, norms, and all metadata. An 8-trit connection has 6561 states, at least 13 fixed-width bits addressable; 12.68 bits is information content, never assumed as implemented storage. Validation loss at matched actual constrained bytes is the **primary mechanistic** comparison; validation loss against total deployed bytes is **co-primary** for any efficiency or per-bit claim. The arms are bit-matched under the stated denominators; not parameter-matched, not compute-matched, not activation-memory-matched, not latency-matched; operation counts, bytes moved, peak memory, latency, and throughput are reported separately.

## 8. Model-cell policy and solver

Fixed for all arms and budgets: depth 8 blocks; head_dim 64; d_ff = 4 * d_model; constrained set = Q, K, V, O and both MLP matrices; pre-norm transformer; RoPE positions; embeddings tied with the output head, fp16, excluded from the constrained budget, with shapes and byte counts reported per cell (embedding matrices are **not** identical across arms when width differs; the frozen policy is what is shared). Divisibility: d_model, d_ff, head_dim all divisible by 8 for C and D; no 8-component group crosses an attention-head boundary; residual alignment preserved; tensor-parallel splitting of a component group prohibited.

Solver (deterministic, hashed): search d_model over multiples of 8; minimize |actual packed constrained bytes - target|; accept within tolerance; tie-break to the smaller d_model. The eight cells `A-low ... D-high` are `TO_FILL` by the solver once the packer exists. Illustrative anchors from ideal-payload arithmetic (the solver's actual-byte answer governs): A-low near d_model 360, C/D-low near 1024, A-high near 728, C/D-high near 2048; B solved by the same procedure over its factor shapes.

## 9. Data contract

Corpus: FineWeb-Edu, `sample-10BT` configuration, one revision, hash `TO_FILL` at download. Train tokens per run: exactly 2.0e9 at both budgets. Validation: 2.0e7 held-out tokens, fixed split, hash recorded. Tokenizer: GPT-2 BPE, 50257, artifact hash recorded. Sequence length 1024. Packing: contiguous documents with EOS separators, causal mask only, no cross-document attention masking. Data order: seeded shuffle per paired seed; comparative seeds {1, 2, 3}; smoke seeds and smoke data are disjoint from the comparative study and never enter analysis.

## 10. Training manifest

AdamW, betas (0.9, 0.95), eps 1e-8. Weight decay 0.1 on constrained shadow tensors only. Gradient clip 1.0. Warmup 2 percent of steps; cosine decay to 10 percent of peak. Effective batch 0.5M tokens per step (microbatch per cell to fit memory, gradient accumulation to the effective size). Precision: bf16 autocast with fp32 master weights; no loss scaling. Checkpoint every 200M tokens; evaluate every 100M tokens; selection rule: best validation loss. Initialization: N(0, 0.02^2) on all shadow and unconstrained weights, residual projections scaled by (2 * depth)^(-1/2); C and D use identical initialization draws per paired seed wherever shapes coincide.

**10a. Per-cell learning-rate selection (fairness stage).** A single global LR across arms is a confound; different families have different LR optima. Frozen grid of peak LRs: {1e-3, 2e-3, 4e-3}. For each of the eight arm-budget cells: one smoke run per grid point, 100M tokens, on smoke data and smoke seeds; select by lowest smoke validation loss, tie-break to the lower LR; write the eight selected values into the manifest **before** the freeze gate. Smoke inspection covers correctness, numerical stability, and this selection only. 24 short smoke runs total.

## 11. Quantization and STE

Weights: gamma = mean |W| over the whole constrained tensor (detached); q_W = clip(round(W / gamma), -1, 1); hard tensor gamma * q_W; identity STE W_eff = W_shadow + stopgrad(W_hard - W_shadow), so dW_eff/dW_shadow = I; no gradient through gamma, rounding, or clipping; one scale per constrained tensor, no per-multivector scales (a per-multivector bf16 scale would add 16 bits per 8-trit connection and constitute a different arm). Activations: one symmetric int8 scale per token over the flattened I*8 scalar coordinates; identity STE; no distinct per-component scales.

Order: shadow -> ternarize the 8-coefficient tensor -> structure contraction. The trainable object is always `[O, I, 8]`. **No independently trainable [O, I, 8, 8] parameter may exist.** A correctly tied expand-first identity-STE implementation is mathematically equivalent in forward value and coefficient gradient (the tied gradient is the sum of the eight true partials, exactly the coefficient-first expression); there is **no** factor-of-eight distortion. Quantize-first is nonetheless the mandated implementation: explicit tying, no redundant materialization, no optimization in the unconstrained 64-entry space, no padding entering the scale, and it matches the integer kernel.

## 12. Reference implementation and kernels

`reference/structure_tensor.py` freezes: the structure-tensor builder (twist flag), the weight and activation quantizers, the training-path einsum, the int32 integer reference (per-output-component bound |acc| <= 8 * I * 127), and the signed-permutation kernel, which must be bit-exact with the sparse contraction. C and D differ only in the structure tensor contents; the surrogate coefficient gradient is sum over b,t,k,r of G_btok * S_kar * xq_btir.

## 13. Mandatory unit tests

All must pass before any comparative curve is opened: basis mapping; cocycle identity (512 triples); associativity (512 triples, both arms); identity element; signature; C anticommutation; D commutativity; left/right discrimination (e1e2 = +e12, e2e1 = -e12); regular homomorphism; explicit block equality with the frozen templates; quantize-first equality; tied-expansion equivalence (equal forwards and coefficient gradients); independent-entry prohibition; weight-STE gradient of ones; scale detachment; activation flattening consistency; single scale per token; zero-token safety; integer-reference equality; signed-permutation bit-exactness; C/D path identity (code and config differ only in structure-tensor contents); accumulator safety; **rank enumeration reproducing section 4 exactly**; algebra-layout preservation; Monarch non-materialization; Monarch primary-path linearity; packer round-trip bit-exactness; budget verification within tolerance.

## 14. Diagnostics

Effective rank: r_eff(A) = exp of the Shannon entropy of the normalized singular-value distribution, in [0, 8], with r_eff(0) = 0; SVD in float64; report r_eff and r_eff/8. For each sampled block: pre-shadow r_eff, post-ternary r_eff, their signed gap and ratio (not every nonzero gap is "collapse"), and the hard exact rank checked against the section 4 catalogue. Effective rank is spectral geometry, not coefficient utilization: a single-blade block is a signed permutation with r_eff = 8. Arm C admits only the restricted singular-value geometries of section 4. Component-use diagnostics, computed separately: hard N_nz; continuous participation ratio PR(w) = (sum w_a^2)^2 / sum w_a^4 with PR(0) = 0 (hard PR equals N_nz and is not separately reported); raw grade mass m_g; per-component grade energy m_g / d_g; enrichment rho_g = m_g / (d_g / 8) against the exchangeable-components reference E[m_g] = d_g / 8; grade entropy normalized by ln 4; per-component mass, occupancy, zero fraction, sign balance; activation norms and gradient RMS by component and grade; quantization error and activation clipping rate. Schedule: at initialization and every frozen checkpoint, over all constrained blocks or a deterministic hash-selected subset; compare against the catalogue, against occupancy-matched sign-randomized references (P2 scope), and against the paired initialization-induced distribution; aggregate by arm, budget, seed, layer, projection type, checkpoint; preserve the full block-level table. Whole-layer capacity: frozen representative layer subset with a frozen randomized-SVD procedure, pre and post. Any numerical rank uses a frozen relative tolerance and is labeled numerical rank.

## 15. Metrics

Primary mechanistic: validation cross-entropy at matched actual constrained-layer bytes. Co-primary deployment: validation cross-entropy against total deployed model bytes. Secondary: fixed claim-decomposition probe macro-F1. Engineering: actual packed bytes, total model bytes, integer operations, scale operations, permutations, activation reads and writes, peak GPU memory, measured memory traffic, latency, tokens per second, energy if reliably measurable.

## 16. Decision rule

For lower-is-better losses, delta_{X,Y,b,s} = (L_{Y,b,s} - L_{X,b,s}) / L_{Y,b,s}. X beats Y at budget b only if the three-seed mean of delta is at least 0.02 **and** delta is positive for all three seeds. Practical equivalence at b requires |delta| < 0.02 for all three seeds. Everything else is unresolved. No p-values for the three-seed loss decision. No arm scales unless the same directional conclusion holds at **both** budgets.

## 17. Frozen decision outcomes

- C clears D, B, A at both budgets: the twisted arm wins under the registered rule; scale C.
- C and D practically equivalent at both budgets, both clear B and A: regular-algebra structure wins without evidence the twist earns its complexity; scale D.
- D clears C, B, A at both budgets: untwisted structure wins; the twist is harmful at this scale; scale D.
- B clears C, D, A at both budgets: the selected Monarch control wins; no algebraic-advantage claim.
- A clears every structured arm at both budgets: the registered constraints are harmful at this scale.
- C and D tie while B clears A and both algebra arms: the factorized control wins; infer nothing beyond B.
- Any budget-crossed, seed-inconsistent, margin-incomplete, or unlisted ordering: unresolved; publish the mixed result; do not scale.

Every outcome ships a model and publishes a result, including the negative ones.

## 18. Probe and Acceptance Check 4

Frozen before training: a seeded, hashed generator produces 20 independent templates x 10 variants = 200 items; each item is a 2-4 sentence synthetic passage asserting a subset of a closed 24-fact schema; task is multi-label fact assertion. Readout: ridge regression, lambda 1.0, standardized features, on the mean-pooled output of block 4 at the selected checkpoint; macro-F1; split 12 templates train / 8 evaluation, split by template ID. Uncertainty: 10,000 paired cluster-bootstrap replicates over template IDs (resample templates, keep all variants, preserve arm pairing and pre/post pairing, refit the registered procedure exactly); 95 percent CI, **descriptive only**. Registered minimum effect: +3 macro-F1 points on the point estimate. Acceptance Check 4 (Tier-3 learning) uses the same probe: a ridge-readout update must move the paired point estimate by at least +3 macro-F1 to count as "measurable."

## 19. Run matrix and leakage control

24 comparative runs (4 arms x 2 budgets x 3 seeds) plus 24 LR-selection smoke runs plus correctness smokes. Work order: shared harness; Arm A; Arm C; Arm D; Arm B; serializer and budget solver; diagnostics and probe; unit tests; manifest completion including 10a; freeze and hash; comparative runs. Leakage rules: comparative training begins only after all four arms are complete; no comparative validation curve is opened before the freeze gate; smoke data and seeds never enter the study; no hyperparameter changes after comparative curves open; any post-freeze code change invalidates the affected run family and requires a new protocol version.

## 20. Run summary schema

Every run writes YAML with: run_id, protocol_version, protocol_sha256, repository_commit, arm, budget, seed, model_cell_sha256, dataset_manifest_sha256, tokenizer_sha256, packer_sha256, structure_tensor_sha256 (null for A and B), start/end UTC, training_tokens_completed, validation_loss_final, validation_loss_selected_checkpoint, probe_f1, ideal_constrained_payload_bits, actual_constrained_bytes, total_deployed_model_bytes, peak_gpu_memory_bytes, tokens_per_second, diagnostic_table_path, checkpoint_path, status.

## 21. Failure handling

NaN or overflow invalidates the run. One replacement seed drawn in order from the pre-registered reserve {11, 12, 13}. Failed runs remain in the evidence ledger marked invalid. Interrupted runs resume from the last checkpoint; a run that cannot resume is treated as failed.

## 22. Readiness criterion and post-primary work

The protocol relabels to **READY TO RUN** only when: every `TO_FILL` is populated; all hashes recorded; all four arms implemented; all eight cells meet the byte tolerance; every mandatory unit test passes; the serializer round-trips bit-exactly; probe and analysis code are frozen and hashed; no comparative curve has been viewed; the owner authorizes the freeze gate. Post-primary (P2 scope): serializer-normalized sensitivity analysis (actual packing vs ideal payload vs naive 2-bit); Monarch int8-intermediate ablation; occupancy-matched sign-randomized rank references; separate optimizer-state cost audit; the scheduled dedicated novelty search round.
