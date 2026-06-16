from __future__ import annotations

import os
import tempfile
from pathlib import Path


os.environ.setdefault("F1DI_ENV", "test")
os.environ.setdefault("F1DI_VECTOR_BACKEND", "memory")
os.environ.setdefault("F1DI_LLM_BACKEND", "rules")

# Without this, any test that hits db_session() without its own DB fixture
# (e.g. test_api_inference.py, the regression suite's real_replay.py) falls
# through to the default storage_url and writes real rows into ./f1di.db —
# silently mixing test/regression fixtures into production flywheel data.
# Route the whole test session at a single isolated sqlite file instead.
_TEST_DB_PATH = Path(tempfile.gettempdir()) / f"f1di_test_{os.getpid()}.db"
os.environ.setdefault("F1DI_STORAGE_URL", f"sqlite:///{_TEST_DB_PATH}")

