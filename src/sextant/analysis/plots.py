"""Frozen loss-versus-bytes plots (PROTOCOL_v3.md sections 7, 15).

Two figures from section-20 ledger rows only:
  1. PRIMARY   — validation loss vs actual packed constrained bytes.
  2. CO-PRIMARY — validation loss vs total deployed model bytes.

Each point is one run (arm marker, budget hollow/filled); per-cell three-seed
mean drawn as a horizontal tick. No training curves are read or drawn.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_COLOR = {"A": "#666666", "B": "#1f77b4", "C": "#d62728", "D": "#2ca02c"}
ARM_MARKER = {"A": "o", "B": "s", "C": "^", "D": "v"}


def _plot(rows, xfield: str, xlabel: str, title: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    cells: dict = {}
    for r in rows:
        if r.get("status") != "complete":
            continue
        key = (r["arm"], r["budget"])
        cells.setdefault(key, []).append(r)
    for (arm, budget), rs in sorted(cells.items()):
        xs = [float(r[xfield]) for r in rs]
        ys = [float(r["validation_loss_selected_checkpoint"]) for r in rs]
        filled = budget == "high"
        ax.scatter(xs, ys, s=42, marker=ARM_MARKER[arm], zorder=3,
                   facecolors=ARM_COLOR[arm] if filled else "none",
                   edgecolors=ARM_COLOR[arm],
                   label=f"{arm} {budget}")
        ax.plot([min(xs), max(xs)], [sum(ys) / len(ys)] * 2,
                color=ARM_COLOR[arm], lw=1, alpha=0.6, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("validation cross-entropy (selected checkpoint)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_primary(rows, out_dir: Path) -> Path:
    return _plot(rows, "actual_constrained_bytes",
                 "actual packed constrained bytes (log)",
                 "PRIMARY: loss vs constrained bytes",
                 Path(out_dir) / "loss_vs_constrained_bytes.png")


def plot_coprimary(rows, out_dir: Path) -> Path:
    return _plot(rows, "total_deployed_model_bytes",
                 "total deployed model bytes (log)",
                 "CO-PRIMARY: loss vs total deployed bytes",
                 Path(out_dir) / "loss_vs_total_bytes.png")
