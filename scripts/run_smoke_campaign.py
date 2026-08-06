"""Item 9 orchestration: prepare the full smoke shard if needed, then run the
24-run LR fairness campaign (PROTOCOL_v3.md section 10a) at the registered
100M tokens/point. Resumable: completed runs are skipped; interrupted runs
resume from checkpoint. Writes selected peak LRs into the manifest at the end.

Run in background; progress is logged to sextant-data/ledger/smoke_campaign.log
via stdout redirection by the launcher.
"""
from __future__ import annotations

import argparse
import json

from sextant.data.pipeline import TokenBin
from sextant.paths import tokenized_dir
from sextant.train import smoke

REQUIRED_TRAIN_TOKENS = 105_000_000
PREP_TRAIN = 110_000_000
PREP_VAL = 6_000_000


def ensure_data(smoke_tokens: int):
    man = tokenized_dir() / "smoke_manifest.json"
    train_bin = tokenized_dir() / "smoke_train.bin"
    need = max(REQUIRED_TRAIN_TOKENS, smoke_tokens + 5_000_000)
    have = len(TokenBin(train_bin)) if train_bin.exists() else 0
    if not man.exists() or have < need:
        print(f"[data] preparing smoke shard: have {have:,} train tokens, need {need:,}", flush=True)
        from prepare_smoke_data import tokenize_prefix
        m = tokenize_prefix(max(PREP_TRAIN, need), PREP_VAL, "sample-10BT")
        print(f"[data] prepared {m['train_tokens']:,} train tokens, "
              f"reserved_docs={m['reserved_docs']}", flush=True)
    else:
        print(f"[data] smoke shard ready: {have:,} train tokens", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-tokens", type=int, default=100_000_000)
    a = ap.parse_args()
    ensure_data(a.smoke_tokens)
    print(f"[campaign] starting 24-run LR fairness stage @ {a.smoke_tokens:,} tokens/point", flush=True)
    selected = smoke.run_all(smoke_tokens=a.smoke_tokens)
    print("[campaign] selected peak LRs:", json.dumps(selected), flush=True)
    print("[campaign] DONE", flush=True)
