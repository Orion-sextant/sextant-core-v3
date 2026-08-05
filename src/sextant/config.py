"""Config system (PROTOCOL_v3.md sections 8, 10). A run is fully described by
its arm, budget, seed, and the frozen manifest. Nothing here is a tunable knob
that could differ between arms beyond what the protocol permits.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import yaml

from .paths import REPO_ROOT

Arm = Literal["A", "B", "C", "D"]
Budget = Literal["low", "high"]
Mode = Literal["smoke", "comparative"]


@dataclass(frozen=True)
class ModelPolicy:
    depth_blocks: int = 8
    head_dim: int = 64
    d_ff_multiplier: int = 4
    constrained_set: tuple[str, ...] = ("wq", "wk", "wv", "wo", "mlp_in", "mlp_out")


@dataclass(frozen=True)
class TrainManifest:
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_fraction: float = 0.02
    min_lr_fraction: float = 0.10
    effective_batch_tokens: int = 500_000
    sequence_length: int = 1024
    checkpoint_every_tokens: int = 200_000_000
    eval_every_tokens: int = 100_000_000
    init_std: float = 0.02


@dataclass(frozen=True)
class RunConfig:
    arm: Arm
    budget: Budget
    seed: int
    mode: Mode = "comparative"
    peak_lr: float = 2e-3
    train_tokens: int = 2_000_000_000
    d_model: int | None = None            # resolved by the budget solver
    monarch_factors: Any = None           # arm B only
    microbatch: int = 8
    policy: ModelPolicy = field(default_factory=ModelPolicy)
    train: TrainManifest = field(default_factory=TrainManifest)

    @property
    def twist(self) -> bool | None:
        if self.arm == "C":
            return True
        if self.arm == "D":
            return False
        return None

    @property
    def cell(self) -> str:
        return f"{self.arm}_{self.budget}"

    @property
    def run_id(self) -> str:
        return f"{self.mode}-{self.arm}-{self.budget}-s{self.seed}-lr{self.peak_lr:g}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["twist"] = self.twist
        d["cell"] = self.cell
        return d


def load_manifest(path: str | Path | None = None) -> dict:
    p = Path(path) if path else REPO_ROOT / "manifest.yaml"
    if not p.exists():  # fall back to the template before the working copy exists
        p = REPO_ROOT / "MANIFEST_TEMPLATE.yaml"
    return yaml.safe_load(p.read_text())
