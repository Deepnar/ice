"""G31: ICE's own file paths must not depend on the working directory.

WHY THIS IS A SMOKE TEST AND NOT A UNIT TEST
  The bug is not "does `resolve()` join two strings correctly" — that cannot
  fail. The bug is that a *whole process* reads different files depending on
  where it was launched, and the only honest way to check that is to launch a
  process somewhere else and ask it what it sees. So the real checks below run
  in a subprocess with `cwd=/tmp`, which is exactly the condition that produced
  two false findings on 2026-08-03 (TRAPS #10).

  No GPU, no DB, no model load — the subprocess imports config/registry and
  reads paths only, which is why this belongs in the smoke suite.
"""

import json
import os
import subprocess
import sys
import textwrap

from src.paths import REPO_ROOT, resolve

# The probe runs in a fresh interpreter; it must not inherit this one's cwd.
_PROBE = textwrap.dedent(
    """
    import json, os, sys
    sys.path.insert(0, {root!r})

    from src.api.config import settings
    from src.model_registry import registry
    from src.paths import REPO_ROOT, resolve

    print("ICE_PROBE" + json.dumps({{
        "cwd": os.getcwd(),
        "repo_root": str(REPO_ROOT),
        "registry_path": registry.REGISTRY_PATH,
        "registry_models": len(registry.load_registry().get("models", {{}})),
        "fallback_model": registry.get_fallback_model(),
        "ner_path": resolve("models/ner/ner_model.pt"),
        "classifier_path": resolve(settings.classifier_model_path),
        "schema_path": resolve(settings.label_schema_path),
        "confidence_fallback_threshold": settings.confidence_fallback_threshold,
        "database_url": settings.database_url,
    }}))
    """
)


def _probe(cwd: str) -> dict:
    """Import ICE in a subprocess rooted at *cwd* and report what it resolved."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(root=str(REPO_ROOT))],
        cwd=cwd, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"probe from {cwd} exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    line = next(ln for ln in proc.stdout.splitlines()
                if ln.startswith("ICE_PROBE"))
    return json.loads(line[len("ICE_PROBE"):])


def test_resolution_is_identical_from_any_cwd():
    """The whole point: same answers from the repo root and from /tmp."""
    here = _probe(str(REPO_ROOT))
    away = _probe("/tmp")

    # Sanity that the two really did run in different places — without this the
    # comparison below could pass vacuously (TRAPS #5: check the negative side).
    assert here["cwd"] != away["cwd"]
    assert away["cwd"] == "/tmp"

    for key in ("repo_root", "registry_path", "ner_path", "classifier_path",
                "schema_path", "registry_models", "fallback_model",
                "confidence_fallback_threshold", "database_url"):
        assert here[key] == away[key], (
            f"{key} differs by working directory: "
            f"{here[key]!r} (repo root) vs {away[key]!r} (/tmp)"
        )


def test_env_file_is_read_from_outside_the_repo():
    """`.env` was the silent one: unanchored, every setting took its code
    default whenever ICE started elsewhere. Asserted against the file rather
    than a hardcoded number so it keeps testing the mechanism after a re-tune.
    """
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        import pytest
        pytest.skip(".env not present in this checkout")

    declared = {}
    for raw in env_file.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        declared[k.strip().lower()] = v.strip()

    away = _probe("/tmp")
    for key in ("confidence_fallback_threshold", "database_url"):
        if key in declared:
            expected = declared[key]
            actual = str(away[key])
            assert actual == expected or float_eq(actual, expected), (
                f"{key} from /tmp is {actual!r}, but .env declares {expected!r}"
                " — the .env file is not being read outside the repo root"
            )


def float_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


def test_resolve_leaves_absolute_paths_alone():
    absolute = os.path.join(os.sep, "opt", "ice", "model.pt")
    assert resolve(absolute) == absolute


def test_resolve_anchors_relative_paths_to_the_install():
    assert resolve("models/x.pt") == str(REPO_ROOT / "models" / "x.pt")


def test_ice_home_overrides_the_install_root(tmp_path):
    """Track F / E7's seam. Set before import, per the module contract."""
    env = dict(os.environ, ICE_HOME=str(tmp_path))
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
         " from src.paths import REPO_ROOT, resolve;"
         " print(resolve('models/x.pt'))"],
        cwd="/tmp", capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path / "models" / "x.pt")
