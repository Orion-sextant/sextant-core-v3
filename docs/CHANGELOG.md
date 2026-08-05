# Changelog

## v3.1 — cell-policy amendment (owner-authorized, pre-freeze)
- Decouple attention inner dim from d_model: `n_heads = floor(d_model/64 + 0.5)`
  (half rounds up), `attn_inner = n_heads*64`; Q/K/V are `d_model->attn_inner`,
  O is `attn_inner->d_model`; MLP unchanged.
- New layout invariant: `attn_inner` divisible by 8.
- Per-arm solver grid (§8 scopes "divisible by 8" to C/D): C/D on multiples of 8,
  dense arm A on a fine grid, B on multiples of 8.
- Froze all 8 model cells within ±1%. Resolves the P0 conflict in
  [CELL_POLICY_CONFLICT.md](CELL_POLICY_CONFLICT.md); see
  [PROTOCOL_v3.1_amendment.md](PROTOCOL_v3.1_amendment.md).

## v3.0 — build session (implementation of the frozen protocol)
- Items 1–8 implemented: scaffold/config/seeding/ledger; transformer harness
  (pre-norm, RoPE) + data pipeline + GPT-2 tokenizer; Arm A (dense ternary);
  Arms C/D (algebra, single code path, twist flag, quantize-first [O,I,8]);
  Arm B (Monarch two-factor, non-materialized, int32-bounded); base-243 packer
  (bit-exact) + budget solver; diagnostics (rank catalogue) + probe + ridge
  readout; full mandatory unit-test suite (28 checks, 35 tests, all passing).
