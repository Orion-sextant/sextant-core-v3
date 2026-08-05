# Protocol amendment v3.0 -> v3.1 (owner-authorized)

**Authorized by the owner** in the build session; resolves the P0 conflict in
[CELL_POLICY_CONFLICT.md](CELL_POLICY_CONFLICT.md). This is a pre-registration
change to a §8 value and therefore bumps the protocol version (v3.0 -> v3.1)
before the freeze gate. The mathematical spine (§3, §4), the arms, the
quantizers, the packer, and the decision rule are unchanged.

## What changed (Option 1: decouple attention inner dim from d_model)

1. **n_heads is frozen as `floor(d_model/64 + 0.5)`** (round to nearest, *half
   rounds up* — not banker's rounding). head_dim stays 64.
2. **Attention inner dim = `n_heads * 64`**, decoupled from d_model. The
   constrained attention maps become rectangular:
   - Q, K, V:  `d_model -> attn_inner`
   - O:        `attn_inner -> d_model`
   - MLP unchanged: `d_model -> 4*d_model` and `4*d_model -> d_model`.
3. **New layout invariant: `attn_inner` is divisible by 8** (holds automatically
   since attn_inner is a multiple of 64), so the algebra 8-grouping never
   crosses a head boundary in the attention path.
4. **Solver search grid is per-arm**, reflecting §8's scoping of the
   "divisible by 8" rule to arms C and D (the algebra grouping): C/D search
   multiples of 8; the dense arm A searches a fine (unit) grid; B searches
   multiples of 8. This lets A_low meet ±1% (it cannot on the multiples-of-8
   grid because at d≈356 an 8-step moves ~3.7% of the byte budget, wider than
   the 2%-wide tolerance window).

## Consequence: every candidate d_model has an integer head count by
construction, so the old head-integrality conflict no longer binds.

## Frozen cells (all within ±1%)

| cell | d_model | n_heads | attn_inner | actual bytes | rel_err |
|------|--------:|--------:|-----------:|-------------:|--------:|
| A_low  | 356  | 6  | 384  | 2,506,324  | +0.25% |
| A_high | 725  | 11 | 704  | 10,002,070 | +0.02% |
| B_low  | 1472 | 23 | 1472 | 2,508,389  | +0.34% |
| B_high | 2800 | 44 | 2816 | 10,077,570 | +0.78% |
| C_low  | 1016 | 16 | 1024 | 2,492,789  | −0.29% |
| C_high | 2040 | 32 | 2048 | 10,009,015 | +0.09% |
| D_low  | 1016 | 16 | 1024 | 2,492,837  | −0.29% |
| D_high | 2040 | 32 | 2048 | 10,009,063 | +0.09% |

C and D remain identical in dimensions (1016 / 2040), init draws, quantizer,
layout, and contraction code — only the structure tensor differs (§5).
