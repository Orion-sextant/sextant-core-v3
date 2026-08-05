"""Append-only results ledger (PROTOCOL_v3.md section 20). Every run writes one
YAML summary row. The ledger lives outside the synced repo tree; rows are never
mutated or deleted (failed runs stay, marked invalid — section 21).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import yaml

from .paths import ledger_dir

# Section 20 schema — the exact key set every row must carry.
SUMMARY_FIELDS = (
    "run_id",
    "protocol_version",
    "protocol_sha256",
    "repository_commit",
    "arm",
    "budget",
    "seed",
    "model_cell_sha256",
    "dataset_manifest_sha256",
    "tokenizer_sha256",
    "packer_sha256",
    "structure_tensor_sha256",  # null for A and B
    "start_utc",
    "end_utc",
    "training_tokens_completed",
    "validation_loss_final",
    "validation_loss_selected_checkpoint",
    "probe_f1",
    "ideal_constrained_payload_bits",
    "actual_constrained_bytes",
    "total_deployed_model_bytes",
    "peak_gpu_memory_bytes",
    "tokens_per_second",
    "diagnostic_table_path",
    "checkpoint_path",
    "status",
)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _ledger_path(mode: str) -> Path:
    # Smoke and comparative never mix (section 4 of the brief); separate files.
    return ledger_dir() / f"ledger_{mode}.yaml"


def append_row(row: dict[str, Any], *, mode: str = "comparative") -> Path:
    """Validate against the section 20 schema and append. Append-only: we open
    in append mode and write one YAML document, so prior rows are immutable."""
    missing = [k for k in SUMMARY_FIELDS if k not in row]
    if missing:
        raise ValueError(f"ledger row missing required fields: {missing}")
    extra = [k for k in row if k not in SUMMARY_FIELDS]
    if extra:
        raise ValueError(f"ledger row has unknown fields: {extra}")
    path = _ledger_path(mode)
    with open(path, "a", encoding="utf-8") as f:
        yaml.safe_dump([{k: row[k] for k in SUMMARY_FIELDS}], f, sort_keys=False)
    return path


def read_rows(mode: str = "comparative") -> list[dict]:
    path = _ledger_path(mode)
    if not path.exists():
        return []
    rows: list[dict] = []
    for doc in yaml.safe_load_all(path.read_text()):
        if isinstance(doc, list):
            rows.extend(doc)
        elif doc:
            rows.append(doc)
    return rows
