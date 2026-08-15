#!/usr/bin/env python3
"""Provenance block for EVERY experiment artifact. Import this, don't re-invent it.

**Why this exists.** Repeatedly this cycle, a number was found sitting in a file
with no way to tell what produced it: which model, which settings, which corpus,
which commit, whether the tree was dirty. The 2026-08-12 retrieval run recorded
its settings and was still unreproducible, because it did not record that the
scorer skipped the budget setter — and the 0.250 it reported turned out to be
0.508 once the harness matched production. A result whose *conditions* are not
captured is not a result; it is a number somebody will trust later without being
able to check it.

So: **every experiment artifact carries `run_meta()` output.** It is cheap, it
is boring, and it is the difference between "we measured X" and "we measured X
on commit abc123, dirty, with gemma4:e4b, over 293 turns of 3 conversations,
with these 14 settings resolved to these values".

Usage:

    from scripts.z1.run_meta import run_meta
    payload = {"meta": run_meta(script=__file__, args=vars(args),
                                settings_keys=["retrieval_rrf_k", ...],
                                extra={"corpus_turns": 293}),
               "results": ...}

⚠ Never put a credential in `extra`. `run_meta` redacts anything whose key looks
like a secret, but it cannot redact what it is not shown.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

_SECRET_HINTS = ("key", "token", "secret", "password", "passwd", "auth",
                 "credential", "api_key")


def _redact(mapping: dict) -> dict:
    """Blank anything whose NAME suggests a secret, keeping its length.

    Name-based rather than value-based on purpose: a value-based rule has to
    guess what a credential looks like, and it will be wrong about the one that
    matters.
    """
    out = {}
    for k, v in (mapping or {}).items():
        if any(h in str(k).lower() for h in _SECRET_HINTS):
            out[k] = f"<redacted:{len(str(v))} chars>" if v else "<empty>"
        else:
            out[k] = v
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return ""


def file_digest(path) -> dict | None:
    """sha256 + size of an input file, so a corpus change is detectable later."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p), "sha256": h.hexdigest()[:16], "bytes": p.stat().st_size}


def run_meta(*, script: str, args: dict | None = None,
             settings_keys: list | None = None,
             inputs: list | None = None,
             extra: dict | None = None) -> dict:
    """The provenance block. Safe to call from any experiment script.

    `settings_keys` are read off the live Settings object and recorded RESOLVED
    — the declared default is not what ran if `.env` overrode it, and a tuning
    sweep is precisely a pile of `.env` overrides.
    """
    resolved = {}
    if settings_keys:
        try:
            from src.api.config import settings
            for k in settings_keys:
                resolved[k] = getattr(settings, k, "<absent>")
        except Exception as exc:                        # noqa: BLE001
            resolved["<error>"] = str(exc)[:200]

    dirty = _git("status", "--porcelain")
    gpu = ""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:                                   # noqa: BLE001
        pass

    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "unix": int(time.time()),
        "script": str(Path(script).name),
        "script_path": str(Path(script).resolve()),
        "argv": sys.argv[1:],
        "args": _redact(args or {}),
        "git": {
            "commit": _git("rev-parse", "HEAD")[:12],
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            # ⚑ A dirty tree means the commit does NOT describe what ran.
            "dirty": bool(dirty),
            "dirty_files": dirty.splitlines()[:40],
        },
        "settings_resolved": _redact(resolved),
        "inputs": [d for d in (inputs or []) if d],
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "gpu": gpu,
            "cwd": os.getcwd(),
        },
        "extra": _redact(extra or {}),
    }


if __name__ == "__main__":
    print(json.dumps(run_meta(script=__file__, args={"demo": 1, "api_key": "abc123"},
                              settings_keys=["retrieval_rrf_k"]), indent=1))
