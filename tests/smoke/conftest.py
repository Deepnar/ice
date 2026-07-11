"""G22 smoke suite — pure-logic, no GPU, no DB, runs in seconds.

Run before every commit:  uv run pytest tests/smoke -q
"""

import pathlib
import sys

# Repo root on sys.path (same convention as the standalone tests/ scripts).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
