import os
import tempfile

# Keep tests off the real data dir; nothing here needs the corpus.
os.environ.setdefault("SEXTANT_DATA_DIR", os.path.join(tempfile.gettempdir(), "sextant-test-data"))
