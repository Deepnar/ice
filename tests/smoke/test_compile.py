"""Syntax sweep: every file under src/ must byte-compile (covers the modules
the import sweep excludes, e.g. src/api/main.py)."""

import compileall


def test_src_compiles():
    assert compileall.compile_dir("src", quiet=2, force=True)
