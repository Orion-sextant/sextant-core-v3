"""SHA-256 helpers for the freeze manifest (PROTOCOL_v3.md sections 20, 22).

Every hash recorded in the manifest is produced here so the algorithm is
uniform and auditable. Canonical JSON (sorted keys, no whitespace) is used for
structured objects so a cell hash is stable across runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_canonical(obj: Any) -> str:
    """Hash of a JSON-serializable object with sorted keys and compact
    separators — stable regardless of insertion order or whitespace."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(blob.encode("utf-8"))
