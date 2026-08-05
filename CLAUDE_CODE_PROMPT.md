# Claude Code brief: build and run Sextant Core v3.0

You are implementing a pre-registered four-arm machine-learning experiment. The normative specification is `PROTOCOL_v3.md` in this folder. If anything in this brief, the manifest, or the reference code conflicts with the protocol, the protocol governs; stop and flag the conflict instead of resolving it silently.

## File inventory

- `PROTOCOL_v3.md`: the frozen protocol. Normative. Read it fully before writing code.
- `MANIFEST_TEMPLATE.yaml`: the manifest with every decided value filled; you populate the `TO_FILL` fields and record every hash.
- `reference/structure_tensor.py`: frozen algebra conventions and reference kernels. Production code must agree with the integer reference and the signed-permutation kernel bit-exactly.
- `reference/verify_algebra.py`: exhaustive verifier for the algebra and the exact rank catalogue. Install it as `tests/test_rank_enumeration.py` (wrap in pytest; assertions are already present).
- `README.md`: orientation and priority statement.

## Non-negotiable rules

1. **Freeze discipline.** No comparative training run starts, and no comparative validation curve is opened, until every mandatory unit test passes, every `TO_FILL` in the manifest is populated, all hashes are recorded, and the owner has explicitly authorized the freeze gate. The gate flag in the manifest starts false; only the owner flips it.
2. **One code path for C and D.** The arms differ only in the structure tensor selected by `--twist {on,off}`. Any C/D divergence beyond structure-tensor contents is a bug and a protocol violation.
3. **Quantize-first.** The trainable object is the `[O, I, 8]` coefficient tensor. No trainable `[O, I, 8, 8]` parameter may exist anywhere; write the unit test that scans the parameter registry and proves it.
4. **Smoke and comparative never mix.** Smoke runs (correctness, stability, and the learning-rate fairness stage of protocol section 10a) use disjoint data and seeds and never enter analysis.
5. **No silent deviations.** If a spec value is unbuildable as written (a dimension that cannot satisfy divisibility, a memory limit, a dependency issue), stop, document the conflict, and ask the owner. Any post-freeze change invalidates the affected run family and bumps the protocol version.
6. **Every run writes the section 20 YAML summary row.** The results ledger is append-only.

## Environment

- Single RTX 4090 (24 GB), Linux, PyTorch with bf16 autocast. Assume interruptions: every runner must checkpoint and resume cleanly (protocol section 21).
- Disk: corpus subset, tokenized cache, checkpoints for 24 comparative runs at two scales; keep only the selected and final checkpoints per run after evaluation, plus diagnostics tables.
- Total comparative compute is roughly one to two 4090-weeks; the LR smoke stage adds 24 short runs first. Schedule long runs sequentially with a queue file the owner can inspect.

## Build order (protocol section 19)

1. Repo scaffold, pytest, config system, deterministic seeding, results ledger.
2. Shared transformer harness (pre-norm, RoPE, depth 8, head_dim 64, d_ff = 4 d_model), data pipeline per the section 9 contract, tokenizer cache.
3. Arm A (dense scalar ternary, section 11 quantizers).
4. Arms C and D (algebra channels, structure tensor from `reference/structure_tensor.py`, single code path, twist flag).
5. Arm B (Monarch two-factor, primary-path constraints of section 5; never densely materialized).
6. Packer (base-243, section 7 and manifest spec), then the budget solver; populate the eight model cells within tolerance and record them in the manifest.
7. Diagnostics suite (section 14) and the probe generator plus ridge readout (section 18).
8. Full mandatory unit-test suite (section 13), including the installed rank-enumeration test and bit-exactness tests against the reference kernels.
9. LR fairness stage: 24 smoke runs per section 10a; write the eight selected peak LRs into the manifest.
10. Freeze: hash protocol, code, data manifests, analysis code; present the completed manifest and the readiness checklist to the owner; **wait for explicit sign-off**.
11. Comparative runs: 24 runs, queue order interleaving arms within each budget (all low-budget cells before high), diagnostics at every frozen checkpoint.
12. Analysis with the frozen decision code only (sections 16-17); produce the decision-table outcome, the two loss-versus-bytes plots (constrained bytes primary, total deployed bytes co-primary), the probe results with cluster-bootstrap CIs, and the diagnostics report comparing learned blocks against the verified catalogue and the paired initialization reference.
13. Write-up draft using only the narrowed hypothesis labels of section 1, with the novelty claim phrased exactly as section 2 requires.

## Definition of done

The manifest has zero `TO_FILL` entries, the freeze gate was authorized and logged, all 24 comparative runs have status complete or a documented failure-and-replacement per section 21, the decision code emits exactly one section 17 outcome (or "unresolved"), and the results ledger, diagnostics tables, plots, and draft write-up are committed. Nothing in the analysis was computed by code that was not hashed at the freeze.

## Reporting cadence to the owner

After each build-order item: a three-line status (done, next, blockers). Before item 11: the full readiness checklist. During comparative runs: one line per completed run from its summary YAML. Never open or summarize a comparative curve before the gate; report only completion status until the freeze is satisfied.
