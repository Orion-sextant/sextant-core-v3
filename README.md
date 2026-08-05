# Sextant Core, document set v3.0

The local cognitive core experiment for Sextant: a four-arm, matched-budget, pre-registered comparison asking whether algebra-valued ternary weights, and specifically the Clifford sign cocycle, buy language modeling performance per stored bit.

One sentence per arm: **A** dense scalar ternary (the null); **B** Monarch ternary (structure as factorization); **C** Cl(3,0) ternary (algebra-valued, twisted); **D** (Z_2)^3 group-algebra ternary (algebra-valued, untwisted; same code, cocycle off). The cleanest comparison is C versus D: identical in every respect except the fixed signs of the structure tensor. The cocycle is the hypothesis.

## Novelty and priority

This project is novel where it claims to be, and the claim is stated precisely: algebra-valued ternary regular-representation weights for language modeling, and the twisted/untwisted matched-pair attribution design, have no located prior instance across two independent-but-correlated literature searches (Claude, knowledge through early 2026; Perplexity grounding, August 2026). The claim stays "to our knowledge" until the scheduled dedicated search round. Priority is protected procedurally: this protocol, hashed at the freeze and committed (optionally posted as a public preregistration), is the timestamped priority artifact, and it exists before any result does. The verified mathematics is not provisional: the algebra, the explicit blocks, and the complete exact rank catalogues in `PROTOCOL_v3.md` section 4 were confirmed by exhaustive machine enumeration (`reference/verify_algebra.py`), all 6,561 ternary blocks per arm.

## Files

- `PROTOCOL_v3.md`: the normative frozen protocol; everything else serves it.
- `CLAUDE_CODE_PROMPT.md`: the operational brief; hand the whole folder to Claude Code with this as the entry point.
- `MANIFEST_TEMPLATE.yaml`: decided values filled; implementation fills `TO_FILL` and hashes.
- `reference/structure_tensor.py`: frozen algebra conventions and reference kernels.
- `reference/verify_algebra.py`: exhaustive verifier; becomes the rank-enumeration unit test.

## Lineage

v1 design brief and v2 four-arm protocol (Claude); Perplexity grounding pass (Monarch control, rank-degeneracy diagnostic, effect sizes); v2.1 consolidation and revision ledger (ChatGPT, adversarial critic pass); v2.2 verification and freeze decisions (Claude, exhaustive enumeration plus P0 resolutions); v3.0 this set, all required revisions applied. Disposition at handoff: pre-registration draft, algebraic core verified, implementation work order assigned. The freeze gate belongs to the owner.
