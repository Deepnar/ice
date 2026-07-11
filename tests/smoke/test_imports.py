"""Every module under src/ must import cleanly (catches G21-class breakage:
bogus includes, top-level typos, missing deps) — the single highest-value
smoke check for a one-person project."""

import importlib
import pathlib

import pytest

# Modules excluded from the sweep, each with the reason and the fix that
# retires the exclusion. Keep this list SHRINKING.
# (src.api.main is NOT excluded: its classifier loads lazily in lifespan(),
# so importing it is cheap and covers the app wiring.)
EXCLUDED = {
    # instantiates its own PyTorchClassifier at import (G13); retire with G13.
    "src.workers.drop_zone",
}


def _modules():
    # Filesystem glob, not pkgutil — src/ uses namespace packages (no
    # __init__.py), which pkgutil.walk_packages silently fails to traverse.
    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root.parent)
        if "__pycache__" in rel.parts:
            continue
        parts = rel.parent.parts if p.name == "__init__.py" else rel.with_suffix("").parts
        mod = ".".join(parts)
        if mod in EXCLUDED or any(mod.startswith(e + ".") for e in EXCLUDED):
            continue
        yield mod


@pytest.mark.parametrize("module_name", sorted(_modules()))
def test_module_imports(module_name):
    importlib.import_module(module_name)
