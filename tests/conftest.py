from __future__ import annotations

import os


os.environ.setdefault("F1DI_ENV", "test")
os.environ.setdefault("F1DI_VECTOR_BACKEND", "memory")
os.environ.setdefault("F1DI_LLM_BACKEND", "rules")

