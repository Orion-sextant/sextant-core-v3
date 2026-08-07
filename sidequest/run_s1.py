"""S1 runner: train each core on each order task, CPU-only, small.

Usage: python sidequest/run_s1.py [--variant pure|gated] [--steps 400]
Writes sidequest/results/ledger_s1.yaml (append) and prints a table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

torch.set_num_threads(8)          # resource contract: 8 of 32 cores
DEVICE = "cpu"

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ncrec import AlgebraScanCore, ScalarProductRNN, TinyGRU, Classifier, n_params
from tasks import TASKS, VOCAB

RESULTS = Path(__file__).parent / "results"


def build_models(variant: str, seed: int):
    out = {}
    c = AlgebraScanCore(VOCAB, k=16, twist=True, variant=variant, seed=seed)
    d = AlgebraScanCore(VOCAB, k=16, twist=False, variant=variant, seed=seed)
    out["C_twist"] = Classifier(c, 16 * 8)
    out["D_untwist"] = Classifier(d, 16 * 8)
    out["scalar_prod"] = Classifier(ScalarProductRNN(VOCAB, d=128, seed=seed), 128)
    out["gru"] = Classifier(TinyGRU(VOCAB, d=48, seed=seed), 48)
    return out


def train_eval(model, task_fn, *, steps: int, T: int, seed: int):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    xtr, ytr = task_fn(4096, T, seed=seed)
    xte, yte = task_fn(1024, T, seed=seed + 7777)     # disjoint eval draw
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, xtr.shape[0], (128,))
        logits = model(xtr[idx])
        loss = torch.nn.functional.cross_entropy(logits, ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(xte).argmax(-1) == yte).float().mean().item()
    return acc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pure", choices=["pure", "gated"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for task_name, task_fn in TASKS.items():
        for seed in a.seeds:
            models = build_models(a.variant, seed)
            for name, m in models.items():
                acc, dt = train_eval(m, task_fn, steps=a.steps, T=a.T, seed=seed)
                rows.append({"variant": a.variant, "task": task_name, "model": name,
                             "seed": seed, "acc": round(acc, 4),
                             "params": n_params(m), "train_s": round(dt, 1)})
                print(f"{a.variant:5} {task_name:11} {name:12} seed{seed} "
                      f"acc={acc:.3f} ({n_params(m):,}p, {dt:.0f}s)", flush=True)

    with open(RESULTS / "ledger_s1.yaml", "a", encoding="utf-8") as f:
        yaml.safe_dump(rows, f, sort_keys=False)

    # aggregate table
    agg: dict = {}
    for r in rows:
        agg.setdefault((r["task"], r["model"]), []).append(r["acc"])
    print("\n=== mean acc over seeds ===")
    for (task, model), accs in sorted(agg.items()):
        print(f"  {task:11} {model:12} {sum(accs)/len(accs):.3f}")
    (RESULTS / f"summary_{a.variant}.json").write_text(json.dumps(
        {f"{t}/{m}": round(sum(v) / len(v), 4) for (t, m), v in agg.items()}, indent=2))


if __name__ == "__main__":
    main()
