"""Prepare smoke tokenized shards (PROTOCOL_v3.md sections 9, 10a).

Streams FineWeb-Edu sample-10BT, tokenizes with GPT-2 BPE, and writes a smoke
train bin and a smoke val bin under the local (non-synced) data dir. Smoke data
is DISJOINT from the comparative study (brief rule 4): it consumes a reserved
prefix of the stream; the recorded ``reserved_docs`` count is where comparative
preparation must START so the two never overlap. Smoke data never enters
analysis.

Usage:
    python scripts/prepare_smoke_data.py --train-tokens 120000000 --val-tokens 6000000
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from sextant.data.pipeline import DTYPE
from sextant.data.tokenizer import EOS_ID, tokenizer_sha256, _load
from sextant.paths import tokenized_dir


def _stream(config: str):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=config,
                      split="train", streaming=True)
    for row in ds:
        yield row["text"]


def tokenize_prefix(n_train: int, n_val: int, config: str) -> dict:
    tok = _load()
    out_train = tokenized_dir() / "smoke_train.bin"
    out_val = tokenized_dir() / "smoke_val.bin"
    need = n_train + n_val
    docs = 0
    written = 0
    buf_train = open(out_train, "wb")
    buf_val = open(out_val, "wb")
    target = buf_train
    for text in _stream(config):
        ids = tok.encode(text).ids
        ids.append(EOS_ID)
        arr = np.asarray(ids, dtype=DTYPE)
        # fill train first, then val
        if written < n_train:
            take = min(len(arr), n_train - written)
            buf_train.write(arr[:take].tobytes())
            if take < len(arr):
                buf_val.write(arr[take:].tobytes())
        else:
            take_val = min(len(arr), need - written)
            buf_val.write(arr[:take_val].tobytes())
        written += len(arr)
        docs += 1
        if docs % 2000 == 0:
            print(f"  {docs:,} docs, {written:,}/{need:,} tokens", flush=True)
        if written >= need:
            break
    buf_train.close(); buf_val.close()
    from sextant.hashing import sha256_file
    manifest = {
        "corpus": "HuggingFaceFW/fineweb-edu", "config": config,
        "role": "smoke", "reserved_docs": docs,
        "train_bin": str(out_train), "train_tokens": min(written, n_train),
        "train_sha256": sha256_file(out_train),
        "val_bin": str(out_val), "val_sha256": sha256_file(out_val),
        "tokenizer_sha256": tokenizer_sha256(), "eos_id": EOS_ID,
        "note": "comparative preparation must start after reserved_docs to stay disjoint",
    }
    (tokenized_dir() / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=int, default=120_000_000)
    ap.add_argument("--val-tokens", type=int, default=6_000_000)
    ap.add_argument("--config", default="sample-10BT")
    a = ap.parse_args()
    m = tokenize_prefix(a.train_tokens, a.val_tokens, a.config)
    print(json.dumps(m, indent=2))
