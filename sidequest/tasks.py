"""S1 synthetic order tasks. No external data; seeded generators only.

ab_order:    random token sequences containing exactly one A and one B;
             label = 1 iff A appears before B. Multisets identical across
             classes, so any order-invariant core is chance-bound.
swap_parity: a fixed base sequence with r adjacent transpositions applied;
             label = parity of r. Same multiset always; order structure only.
"""
from __future__ import annotations

import torch

VOCAB = 32
A_TOK, B_TOK = 30, 31          # reserved marker tokens


def ab_order(n: int, T: int, *, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 30, (n, T), generator=g)
    pos = torch.multinomial(torch.ones(n, T), 2, replacement=False, generator=g)
    lo, hi = pos.min(1).values, pos.max(1).values
    y = torch.randint(0, 2, (n,), generator=g)
    rows = torch.arange(n)
    x[rows, lo] = torch.where(y == 1, A_TOK, B_TOK)   # y=1: A first
    x[rows, hi] = torch.where(y == 1, B_TOK, A_TOK)
    return x, y


def swap_parity(n: int, T: int, *, seed: int, max_swaps: int = 6):
    g = torch.Generator().manual_seed(seed)
    base = torch.randperm(30, generator=g)[:T] % 30
    x = base.unsqueeze(0).repeat(n, 1)
    r = torch.randint(0, max_swaps + 1, (n,), generator=g)
    for i in range(n):
        for _ in range(int(r[i])):
            j = int(torch.randint(0, T - 1, (1,), generator=g))
            x[i, j], x[i, j + 1] = x[i, j + 1].clone(), x[i, j].clone()
    return x, (r % 2)


TASKS = {"ab_order": ab_order, "swap_parity": swap_parity}
