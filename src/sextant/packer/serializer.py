"""Model constrained-weight serializer (PROTOCOL_v3.md section 7).

File layout:  [u32 header_len][header_json][data_section][scales_section]
- data_section: per-export base-243 trit block, each padded to ``ALIGN`` bytes.
- scales_section: one fp16 scale per export, concatenated, padded to ALIGN.
- header_json: per-export {name, kind, n_trits, meta, data_offset, data_len,
  scale_index} plus arm/d_model. Offsets are relative to data/scales sections,
  so header length is independent of the offsets it stores (no circularity).

``actual_constrained_bytes`` = 4 + len(header_json) + len(data) + len(scales),
i.e. everything that must be stored to reconstruct the constrained maps
(section 7 measure 2: trit encoding, padding, alignment, scales, metadata).

The SAME layout math (``compute_layout``) drives both the real packer and the
budget solver, so the solver's byte count is exactly what the packer produces.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np

from .base243 import packed_len, pack_trits, unpack_trits

ALIGN = 64
SCALE_BYTES = 2  # fp16


def _round_up(n: int, a: int = ALIGN) -> int:
    return ((n + a - 1) // a) * a


@dataclass
class ExportSpec:
    name: str
    kind: str            # "dense" | "algebra" | "monarch"
    n_trits: int
    meta: dict = field(default_factory=dict)


@dataclass
class Layout:
    header_json: bytes
    data_len: int
    scales_len: int
    entries: list          # per-export dict with data_offset/data_len/scale_index

    @property
    def actual_constrained_bytes(self) -> int:
        return 4 + len(self.header_json) + self.data_len + self.scales_len


def compute_layout(specs: list[ExportSpec], *, arm: str, d_model) -> Layout:
    entries = []
    data_cursor = 0
    for i, s in enumerate(specs):
        dl = _round_up(packed_len(s.n_trits))
        entries.append({
            "name": s.name, "kind": s.kind, "n_trits": s.n_trits, "meta": s.meta,
            "data_offset": data_cursor, "data_len": dl, "scale_index": i,
        })
        data_cursor += dl
    data_len = data_cursor
    scales_len = _round_up(len(specs) * SCALE_BYTES)
    header = {"arm": arm, "d_model": d_model, "n_exports": len(specs),
              "align": ALIGN, "entries": entries}
    header_json = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return Layout(header_json, data_len, scales_len, entries)


# --------------------------------------------------------------------------
# Analytic specs for a cell (mirror the transformer's constrained modules)
# --------------------------------------------------------------------------
def _cell_maps(d_model: int):
    d, ff = d_model, 4 * d_model
    per_block = [("wq", d, d), ("wk", d, d), ("wv", d, d), ("wo", d, d),
                 ("mlp_in", d, ff), ("mlp_out", ff, d)]  # (name, in, out)
    for blk in range(8):
        for name, i, o in per_block:
            yield f"blk{blk}.{name}", i, o


def specs_for_cell(arm: str, d_model: int) -> list[ExportSpec]:
    from ..arms.arm_b_monarch import choose_blocks

    specs: list[ExportSpec] = []
    for name, i, o in _cell_maps(d_model):
        if arm == "A":
            specs.append(ExportSpec(name, "dense", o * i, {"in": i, "out": o}))
        elif arm in ("C", "D"):
            specs.append(ExportSpec(name, "algebra", (o // 8) * (i // 8) * 8,
                                    {"in": i, "out": o, "twist": arm == "C"}))
        elif arm == "B":
            g = choose_blocks(i, o)
            a, c = i // g, o // g
            specs.append(ExportSpec(f"{name}.w1", "monarch", g * a * a, {"g": g, "a": a, "c": c}))
            specs.append(ExportSpec(f"{name}.w2", "monarch", g * c * a, {"g": g, "a": a, "c": c}))
        else:
            raise ValueError(arm)
    return specs


def cell_actual_bytes(arm: str, d_model: int) -> int:
    specs = specs_for_cell(arm, d_model)
    return compute_layout(specs, arm=arm, d_model=d_model).actual_constrained_bytes


def cell_ideal_trits(arm: str, d_model: int) -> int:
    return sum(s.n_trits for s in specs_for_cell(arm, d_model))


# --------------------------------------------------------------------------
# Real pack / unpack round-trip
# --------------------------------------------------------------------------
def pack_model(model, *, arm: str, d_model) -> tuple[bytes, Layout, dict]:
    """Serialize a model's constrained modules. Returns (blob, layout, payload)
    where payload maps export-name -> (trits np.int8, scale np.float16) for the
    round-trip test.

    The header/layout is driven by ``specs_for_cell`` — the SAME source the
    budget solver uses — so the packed byte count equals the solver's count
    exactly. The model supplies only the trit/scale data, keyed by export name.
    """
    payload = {}
    for mod_name, mod in model.constrained_modules().items():
        for exp in mod.export_quantized(mod_name):
            trits = exp.trits.numpy().astype(np.int8)
            scale = exp.scales.numpy().astype(np.float16).reshape(-1)[0]
            payload[exp.name] = (trits, scale)

    exports = specs_for_cell(arm, d_model)
    for s in exports:  # data must agree with the frozen spec
        if s.name not in payload:
            raise KeyError(f"model missing export {s.name!r} expected by cell spec")
        if payload[s.name][0].size != s.n_trits:
            raise ValueError(
                f"{s.name}: model has {payload[s.name][0].size} trits, "
                f"spec expects {s.n_trits}")
    layout = compute_layout(exports, arm=arm, d_model=d_model)

    data = bytearray(layout.data_len)
    for e in layout.entries:
        trits, _ = payload[e["name"]]
        blob = pack_trits(trits)
        data[e["data_offset"] : e["data_offset"] + len(blob)] = blob
    scales = np.zeros(_round_up(len(exports) * SCALE_BYTES) // SCALE_BYTES, dtype=np.float16)
    for e in layout.entries:
        scales[e["scale_index"]] = payload[e["name"]][1]

    blob = struct.pack("<I", len(layout.header_json)) + layout.header_json \
        + bytes(data) + scales.tobytes()
    assert len(blob) == layout.actual_constrained_bytes
    return blob, layout, payload


def unpack_model(blob: bytes) -> dict:
    (hlen,) = struct.unpack("<I", blob[:4])
    header = json.loads(blob[4 : 4 + hlen].decode("utf-8"))
    base = 4 + hlen
    data_len = sum(e["data_len"] for e in header["entries"])
    data = blob[base : base + data_len]
    scales = np.frombuffer(blob[base + data_len :], dtype=np.float16)
    out = {}
    for e in header["entries"]:
        chunk = data[e["data_offset"] : e["data_offset"] + e["data_len"]]
        trits = unpack_trits(chunk, e["n_trits"])
        out[e["name"]] = (trits, float(scales[e["scale_index"]]), e["meta"])
    return out
