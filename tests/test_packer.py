"""Section 13 packer test: base-243 round-trip and model pack/unpack
bit-exactness; solver byte count equals the real packed byte count.
"""
import numpy as np

from sextant.model.transformer import Transformer, ModelArgs
from sextant.seeding import seed_everything, paired_generator
from sextant.packer.base243 import pack_trits, unpack_trits
from sextant.packer.serializer import pack_model, unpack_model, cell_actual_bytes


def test_base243_round_trip():
    rng = np.random.default_rng(0)
    for n in (0, 1, 4, 5, 6, 7, 999, 12345):
        t = rng.integers(-1, 2, size=n).astype(np.int8)
        assert np.array_equal(unpack_trits(pack_trits(t), n), t)


def test_model_pack_round_trip_and_byte_agreement():
    for arm in ("A", "B", "C", "D"):
        seed_everything(1); gen = paired_generator(1)
        m = Transformer(ModelArgs(arm=arm, d_model=64, vocab_size=64, depth=8), gen)
        blob, layout, payload = pack_model(m, arm=arm, d_model=64)
        # solver's analytic byte count == real serialized length
        assert len(blob) == cell_actual_bytes(arm, 64) == layout.actual_constrained_bytes
        out = unpack_model(blob)
        for name, (trits, _scale) in payload.items():
            assert np.array_equal(trits.reshape(-1), out[name][0])
