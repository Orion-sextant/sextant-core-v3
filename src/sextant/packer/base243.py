"""Base-243 trit packing (PROTOCOL_v3.md section 7; manifest packer spec).

Five trits per byte: each trit in {-1,0,+1} maps to {0,1,2}; a group of five is
encoded as sum_i d_i * 3^i in [0, 242] < 256. Round-trips bit-exactly.
"""
from __future__ import annotations

import numpy as np

TRITS_PER_BYTE = 5
_POW3 = np.array([1, 3, 9, 27, 81], dtype=np.int32)


def pack_trits(trits: np.ndarray) -> bytes:
    """trits: integer array with values in {-1,0,1} (any shape; row-major)."""
    t = np.asarray(trits).reshape(-1).astype(np.int8)
    if t.size and (t.min() < -1 or t.max() > 1):
        raise ValueError("trit values must be in {-1,0,1}")
    u = (t + 1).astype(np.int32)  # {0,1,2}
    pad = (-u.size) % TRITS_PER_BYTE
    if pad:
        u = np.concatenate([u, np.zeros(pad, dtype=np.int32)])
    groups = u.reshape(-1, TRITS_PER_BYTE)
    vals = (groups * _POW3).sum(axis=1).astype(np.uint8)
    return vals.tobytes()


def unpack_trits(data: bytes, n_trits: int) -> np.ndarray:
    vals = np.frombuffer(data, dtype=np.uint8).astype(np.int32).copy()
    out = np.empty((vals.size, TRITS_PER_BYTE), dtype=np.int8)
    for i in range(TRITS_PER_BYTE):
        out[:, i] = (vals % 3).astype(np.int8)
        vals //= 3
    return out.reshape(-1)[:n_trits] - 1


def packed_len(n_trits: int) -> int:
    return (n_trits + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
