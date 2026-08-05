"""Probe item generator (PROTOCOL_v3.md section 18).

Seeded and hashed: 20 templates x 10 variants = 200 items over a closed 24-fact
schema. Each template asserts a fixed subset (2-4 facts) of the schema; its 10
variants are surface rewordings that assert the SAME fact subset, so template ID
is the cluster unit for the split and the bootstrap. Task: multi-label fact
assertion (24 binary labels).

Frozen before training: the generator seed and a sha256 over the full item set
(text + labels) become ``generator_seed`` / ``generator_sha256`` in the manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..hashing import sha256_canonical

SUBJECTS = ["the archivist", "the courier", "the gardener",
            "the mason", "the pilot", "the weaver"]
PREDICATES = ["arrived at dawn", "carried a sealed letter",
              "spoke of the harvest", "wore a grey cloak"]
N_FACTS = len(SUBJECTS) * len(PREDICATES)  # 24
assert N_FACTS == 24

N_TEMPLATES = 20
N_VARIANTS = 10
_CONNECTIVES = ["It is recorded that", "Witnesses agree that", "The ledger notes that",
                "By all accounts", "Records confirm that", "Locals say that",
                "The report states that", "As is known,", "Notably,", "Plainly,"]


def _fact_text(fidx: int) -> str:
    s, p = divmod(fidx, len(PREDICATES))
    return f"{SUBJECTS[s]} {PREDICATES[p]}"


@dataclass
class ProbeItem:
    template_id: int
    variant_id: int
    text: str
    labels: list          # 24-dim {0,1}
    fact_subset: list


def generate_items(seed: int) -> list[ProbeItem]:
    rng = np.random.default_rng(seed)
    template_subsets = []
    for _t in range(N_TEMPLATES):
        k = int(rng.integers(2, 5))                 # 2..4 facts
        subset = sorted(rng.choice(N_FACTS, size=k, replace=False).tolist())
        template_subsets.append(subset)

    items: list[ProbeItem] = []
    for t, subset in enumerate(template_subsets):
        labels = [1 if f in subset else 0 for f in range(N_FACTS)]
        for v in range(N_VARIANTS):
            order = subset[:]
            rng.shuffle(order)
            conn = _CONNECTIVES[v % len(_CONNECTIVES)]
            sentences = [f"{conn} {_fact_text(f)}." for f in order]
            items.append(ProbeItem(t, v, " ".join(sentences), labels, subset))
    return items


def generate_matrix(seed: int):
    """Return (texts, labels [200,24] int, template_ids [200] int)."""
    items = generate_items(seed)
    texts = [it.text for it in items]
    labels = np.array([it.labels for it in items], dtype=np.int64)
    tids = np.array([it.template_id for it in items], dtype=np.int64)
    return texts, labels, tids


def generator_sha256(seed: int) -> str:
    payload = [asdict(it) for it in generate_items(seed)]
    return sha256_canonical(payload)
