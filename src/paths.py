"""Where ICE's own files live — anchored to the install, never to the CWD.

WHY THIS MODULE EXISTS (roadmap G31)
  ICE reads five things off disk by name: the `.env` file, the model registry
  JSON, the micro-NER checkpoint, the classifier checkpoint, and the label
  schema. Every one of those paths used to be interpreted relative to
  *whatever directory the process happened to start in*. Inside the repo root
  that is correct by accident; anywhere else it is silently wrong.

  Silently is the operative word, and it is why this is a module rather than a
  one-line fix. Measured 2026-08-03, both by accident, and **both produced
  results that looked like findings** until someone checked `pwd`:

    - `models/ner/ner_model.pt` missed  ⇒ the micro-NER fell back to a
      capitalized-word regex and one run's entity count dropped 52.7 → 34.1
      per turn.
    - `models/model_registry.json` missed ⇒ `load_registry()` returned empty,
      so `get_fallback_model()` dropped to `settings.default_fallback_model`
      (`qwen2.5:7b`) — a quant measured to produce word salad.

  Found 2026-08-08 while fixing those, and worse than either: the `.env` file
  is read by the same rule. Outside the repo root pydantic-settings finds no
  file and **every setting silently falls back to its code default**. Today
  exactly one value diverges — `confidence_fallback_threshold`, 0.5 in `.env`
  against 0.75 in `config.py` — and that one gates the wide-net fallback in
  `orchestrator.py`, i.e. a *degraded single-leg retrieval mode*. The other
  three keys in `.env` happen to equal their defaults, which is luck, not
  design: adding one divergent key is enough to change ICE's behavior based on
  nothing but where it was launched from.

  That failure shape is catalogued as TRAPS #10 and it is exactly what Z1 —
  which measures components in isolation across many separately-launched
  scripts — cannot tolerate.

THE RULE
  A path to one of ICE's OWN artifacts resolves through `resolve()` here. A
  path to the USER's material does not: `coding/project_facts.py` and
  `coding/code_graph.py` walk a repo the user points ICE at, and anchoring
  those to ICE's install directory would be a bug, not a fix.

FORWARD-COMPATIBILITY (Track F, E7)
  `ICE_HOME` is the single override, and it exists because two shipped
  destinations will need it. Track F's packaged app has no repo root — its
  models live beside the bundle or in a user data directory. E7's headless
  `ice-mcp` boot works today only because `--directory <repo>` happens to set
  the CWD to the right place; with this module it works because the code knows
  where it lives, and `ICE_HOME` is how a package overrides that. Keeping the
  override in ONE place is the point: the alternative is Track F re-deriving a
  root in each of the modules this file replaces.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/paths.py → parents[0] is src/, parents[1] is the repo root.
_INSTALL_ROOT = Path(__file__).resolve().parents[1]


def _resolve_root() -> Path:
    override = os.environ.get("ICE_HOME", "").strip()
    return Path(override).expanduser().resolve() if override else _INSTALL_ROOT


#: The install root. Resolved ONCE at import, which means ``ICE_HOME`` must be
#: set before ICE is imported — the same contract as every other environment
#: setting, and the same thing the five hand-rolled roots this replaces did.
#: The alternative (re-reading the env per call) would let ``REPO_ROOT`` and
#: ``resolve()`` disagree inside one process, which is a worse trap than the
#: one this module exists to close.
#:
#: Use this to build a path yourself (``REPO_ROOT / "docker" / "x.yml"``); use
#: ``resolve()`` for anything that arrived from settings, since that also
#: handles the already-absolute case.
REPO_ROOT = _resolve_root()


def resolve(path: str | os.PathLike) -> str:
    """Anchor a relative ICE path to the install root; leave absolute ones alone.

    Returning ``str`` rather than ``Path`` is deliberate — every caller feeds
    the result to ``torch.load``, ``open`` or ``os.path.exists``, all of which
    take either, and the settings fields these come from are typed ``str``.
    """
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)
