"""Deterministic seeding (PROTOCOL_v3.md sections 9, 10, 21).

C and D use identical initialization draws per paired seed wherever shapes
coincide; that is achieved by seeding both arms with the same ``seed`` and
drawing the constrained shadow tensors in the same order (see arms.base).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS reproducibility for matmul-heavy training.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def paired_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """A CPU/CUDA generator keyed only on the paired seed, for init draws that
    must coincide between arms C and D."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
