"""Pytest configuration shared by all suites.

* Adds ``src`` to ``sys.path`` so the in-repo packages import cleanly without
  requiring the project to be ``pip install``-ed.
* Injects deterministic env vars so importing :class:`Settings` in test
  modules does not raise validation errors when secrets are absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


_DEFAULTS: dict[str, str] = {
    "TAXBOT_POSTGRES_DSN": "postgresql://postgres:postgres@localhost:5432/postgres",
    "TAXBOT_UNSTRUCTURED_API_KEY": "test-unstructured-key",
    "TAXBOT_HUGGINGFACE_API_TOKEN": "test-hf-token",
    "TAXBOT_GEMINI_API_KEY": "test-gemini-key",
    "TAXBOT_QDRANT_URL": "https://test.qdrant.io:6333",
    "TAXBOT_QDRANT_API_KEY": "test-qdrant-key",
}

for key, value in _DEFAULTS.items():
    os.environ.setdefault(key, value)
