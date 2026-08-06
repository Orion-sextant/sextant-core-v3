"""LR fairness stage driver (PROTOCOL_v3.md section 10a; brief item 9).

For each of the 8 arm-budget cells, run one smoke run per peak-LR grid point
{1e-3, 2e-3, 4e-3} for ``smoke_tokens`` on smoke data/seeds (disjoint from the
comparative study), then select the peak LR with the lowest smoke validation
loss (tie-break to the lower LR) and write the eight selected values into the
manifest. Smoke runs never enter analysis; each writes a row to the smoke
ledger and its checkpoint is deleted after the result is recorded (smoke
inspection covers correctness, stability, and selection only).

Resumable: a (cell, lr) whose smoke-ledger row is already 'complete' is skipped.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from ..config import RunConfig, TrainManifest
from ..hashing import sha256_file
from ..ledger import append_row, read_rows, utc_now
from ..paths import REPO_ROOT, tokenized_dir
from .runner import Runner

LR_GRID = [1.0e-3, 2.0e-3, 4.0e-3]
MICROBATCH = {356: 8, 725: 6, 1016: 4, 1472: 3, 2040: 2, 2800: 1}
SMOKE_SEED_BASE = 1000  # disjoint from comparative {1,2,3} and reserve {11,12,13}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _hashes(arm: str) -> dict:
    ref = REPO_ROOT / "reference" / "structure_tensor.py"
    st = sha256_file(ref) if arm in ("C", "D") else None
    return {
        "protocol_sha256": sha256_file(REPO_ROOT / "PROTOCOL_v3.md"),
        "packer_sha256": sha256_file(REPO_ROOT / "src" / "sextant" / "packer" / "serializer.py"),
        "structure_tensor_sha256": st,
    }


def run_cell_lr(arm: str, budget: str, cell: dict, lr: float, *, smoke_tokens: int,
                train_bin: Path, val_bin: Path, data_sha: str, tok_sha: str) -> dict:
    d_model = cell["d_model"]
    mb = MICROBATCH.get(d_model, 2)
    cell_idx = ["A_low", "B_low", "C_low", "D_low", "A_high", "B_high", "C_high", "D_high"].index(f"{arm}_{budget}")
    seed = SMOKE_SEED_BASE + cell_idx
    cfg = RunConfig(arm=arm, budget=budget, seed=seed, mode="smoke", peak_lr=lr,
                    train_tokens=smoke_tokens, d_model=d_model, microbatch=mb,
                    train=TrainManifest(eval_every_tokens=smoke_tokens))
    tag = f"smoke-{arm}-{budget}-lr{lr:g}"
    r = Runner(cfg, train_bin=train_bin, val_bin=val_bin, ckpt_tag=tag)
    r.maybe_resume()
    res = r.train(on_step=lambda s, t, l, lrr: None)
    h = _hashes(arm)
    row = {
        "run_id": tag, "protocol_version": "3.1", "protocol_sha256": h["protocol_sha256"],
        "repository_commit": _git_commit(), "arm": arm, "budget": budget, "seed": seed,
        "model_cell_sha256": cell["cell_sha256"], "dataset_manifest_sha256": data_sha,
        "tokenizer_sha256": tok_sha, "packer_sha256": h["packer_sha256"],
        "structure_tensor_sha256": h["structure_tensor_sha256"],
        "start_utc": None, "end_utc": utc_now(),
        "training_tokens_completed": res.tokens_completed,
        "validation_loss_final": res.val_loss_final,
        "validation_loss_selected_checkpoint": res.val_loss_selected,
        "probe_f1": None,
        "ideal_constrained_payload_bits": cell["ideal_constrained_payload_bits"]
            if "ideal_constrained_payload_bits" in cell else None,
        "actual_constrained_bytes": cell["actual_constrained_bytes"],
        "total_deployed_model_bytes": None,
        "peak_gpu_memory_bytes": res.peak_gpu_memory_bytes,
        "tokens_per_second": res.tokens_per_second,
        "diagnostic_table_path": None, "checkpoint_path": res.checkpoint_path,
        "status": res.status,
    }
    append_row(row, mode="smoke")
    Path(res.checkpoint_path).unlink(missing_ok=True)  # smoke ckpts not retained
    return row


def already_done(tag: str) -> bool:
    return any(r.get("run_id") == tag and r.get("status") == "complete"
               for r in read_rows(mode="smoke"))


def run_all(*, smoke_tokens: int = 100_000_000):
    manifest = yaml.safe_load((REPO_ROOT / "manifest.yaml").read_text())
    cells = manifest["budgets"]["model_cells"]
    sm = json.loads((tokenized_dir() / "smoke_manifest.json").read_text())
    train_bin, val_bin = Path(sm["train_bin"]), Path(sm["val_bin"])
    data_sha, tok_sha = sm["train_sha256"], sm["tokenizer_sha256"]

    order = ["A_low", "B_low", "C_low", "D_low", "A_high", "B_high", "C_high", "D_high"]
    results = {}
    for cellname in order:
        arm, budget = cellname.split("_")
        for lr in LR_GRID:
            tag = f"smoke-{arm}-{budget}-lr{lr:g}"
            if already_done(tag):
                print(f"[skip] {tag} already complete", flush=True)
                continue
            print(f"[run ] {tag} ({smoke_tokens:,} tokens)", flush=True)
            row = run_cell_lr(arm, budget, cells[cellname], lr, smoke_tokens=smoke_tokens,
                              train_bin=train_bin, val_bin=val_bin,
                              data_sha=data_sha, tok_sha=tok_sha)
            print(f"[done] {tag} val_loss={row['validation_loss_final']:.4f} "
                  f"tok/s={row['tokens_per_second']:.0f}", flush=True)
    return select_and_write_manifest()


def select_and_write_manifest():
    """Per cell: lowest smoke val loss, tie-break lower LR -> manifest."""
    rows = [r for r in read_rows(mode="smoke") if r.get("status") == "complete"]
    by_cell: dict[str, list] = {}
    for r in rows:
        by_cell.setdefault(f"{r['arm']}_{r['budget']}", []).append(r)
    selected = {}
    for cell, rs in by_cell.items():
        # sort by (val_loss, lr): lr encoded in run_id suffix
        def lr_of(r):
            return float(r["run_id"].split("lr")[-1])
        best = min(rs, key=lambda r: (r["validation_loss_final"], lr_of(r)))
        selected[cell] = lr_of(best)
    _write_selected(selected)
    return selected


def _write_selected(selected: dict):
    path = REPO_ROOT / "manifest.yaml"
    manifest = yaml.safe_load(path.read_text())
    sel = manifest["training"]["lr_fairness_stage"]["selected_peak_lr"]
    for cell, lr in selected.items():
        if cell in sel:
            sel[cell] = float(lr)
    # write back only the selected block via targeted replacement to keep formatting
    lines = path.read_text().splitlines(keepends=True)
    out, in_block = [], False
    for ln in lines:
        if ln.strip().startswith("selected_peak_lr:"):
            in_block = True; out.append(ln); continue
        if in_block and ":" in ln and any(ln.strip().startswith(c + ":") for c in sel):
            cell = ln.strip().split(":")[0]
            out.append(ln.replace("TO_FILL", str(sel[cell])) if "TO_FILL" in ln
                       else f"      {cell}: {sel[cell]}\n")
        else:
            if in_block and not ln.startswith(" " * 6):
                in_block = False
            out.append(ln)
    path.write_text("".join(out))
