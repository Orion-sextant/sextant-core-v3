"""Path resolution. Large artifacts (corpus, tokenized cache, checkpoints,
ledger) live OUTSIDE the OneDrive-synced repo tree so they are never synced.

Default data root: ``C:/Users/<user>/sextant-data`` (local, non-synced).
Override with the ``SEXTANT_DATA_DIR`` environment variable, or a
``.sextant_local.yaml`` at the repo root with a ``data_dir:`` key.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    return Path.home() / "sextant-data"


def data_dir() -> Path:
    env = os.environ.get("SEXTANT_DATA_DIR")
    if env:
        return Path(env)
    local = REPO_ROOT / ".sextant_local.yaml"
    if local.exists():
        import yaml

        cfg = yaml.safe_load(local.read_text()) or {}
        if cfg.get("data_dir"):
            return Path(cfg["data_dir"])
    return _default_data_dir()


def _sub(name: str) -> Path:
    p = data_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpus_dir() -> Path:
    return _sub("corpus")


def tokenized_dir() -> Path:
    return _sub("tokenized")


def checkpoints_dir() -> Path:
    return _sub("checkpoints")


def ledger_dir() -> Path:
    return _sub("ledger")
