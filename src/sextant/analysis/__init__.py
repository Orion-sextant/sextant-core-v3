"""Frozen analysis code (PROTOCOL_v3.md sections 16-17).

This package is hashed into the manifest (analysis_code_sha256) before the
freeze gate. Post-freeze changes invalidate the affected run family (section
19). Nothing here reads training curves; inputs are the section-20 run-summary
ledger rows only.
"""
