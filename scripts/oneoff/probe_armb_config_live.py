"""Does arm B's env ACTUALLY reach NuNER as 19 labels, and does it use them?

⚑ WHAT THIS IS FOR. Arm B is defined by two environment variables. Every link
between them and NuNER's label conditioning has been proved SEPARATELY —
env→setting, setting→`_grounding_ner_labels()`, `labels=`→`_extract_background`
— and separately is not the same as together. A ~110-minute arm that turns out
to have run on the narrow list is indistinguishable from one that ran correctly
until someone reads the entity names. So this asserts the whole chain at once,
through `extract_triplets`, the real entry point, under arm B's real env.

It also answers "are the two opened-up types actually being USED", which is a
different question from "were they passed": NuNER could receive `concept` and
`object` and assign neither. The mirror reports the label of every returned
span, so a zero there would mean the widening is inert.

Run it the way arm B will be run:
    CODEX_EXTRACTION_NER_TIER=background \
    CODEX_GROUNDING_NER_EXTRA_TYPES=concept,object \
    uv run python scripts/oneoff/probe_armb_config_live.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.oneoff.probe_grounding_label_set import (background_with_labels,  # noqa: E402
                                                      sample_turns)
from src.api.config import settings                                # noqa: E402
from src.memory.embedder import get_embedder                       # noqa: E402
from src.workers import codex_extractor                            # noqa: E402

FAILURES = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> int:
    print("=" * 74)
    print("ARM B LIVE CONFIG CHECK")
    print("=" * 74)
    print(f"  env CODEX_EXTRACTION_NER_TIER        = "
          f"{os.environ.get('CODEX_EXTRACTION_NER_TIER')!r}")
    print(f"  env CODEX_GROUNDING_NER_EXTRA_TYPES  = "
          f"{os.environ.get('CODEX_GROUNDING_NER_EXTRA_TYPES')!r}")
    print(f"  settings.codex_extraction_ner_tier   = "
          f"{settings.codex_extraction_ner_tier!r}")
    print(f"  settings.codex_grounding_ner_extra_types = "
          f"{settings.codex_grounding_ner_extra_types!r}\n")

    print("── 1. the settings the arm will actually run on ──")
    check(settings.codex_extraction_ner_tier == "background",
          "tier resolves to background")
    check("concept" in settings.codex_grounding_ner_extra_types
          and "object" in settings.codex_grounding_ner_extra_types,
          "extra types resolve to concept+object")

    print("\n── 2. what the codex call site computes ──")
    labels = codex_extractor._grounding_ner_labels()
    check(labels is not None, "_grounding_ner_labels() is not None")
    check(labels is not None and "concept" in labels, "list contains concept")
    check(labels is not None and "object" in labels, "list contains object")
    check(labels is not None and len(labels) == 19,
          "list is 19 labels", f"got {len(labels) if labels else 0}")

    print("\n── 3. what the PRODUCTION call site passes, spied in flight ──")
    seen = {}
    real = codex_extractor.extract_entities

    def spy(chunk, embedder, max_chars=None, tier="preflight", labels=None):
        out = real(chunk, embedder, max_chars=max_chars, tier=tier, labels=labels)
        seen.setdefault("tier", tier)
        seen.setdefault("labels", list(labels) if labels else None)
        seen.setdefault("entities", out)
        # ⚠ THE CHUNK, NOT THE TURN. `extract_triplets` runs `_chunk_text` first
        # and calls this once per chunk (A1). The first version of this probe
        # mirrored the FULL TURN against the FIRST CHUNK's output and reported
        # drift — 24 spans against 13 — which was the harness comparing two
        # different inputs, not the code disagreeing with itself. Same shape as
        # the three harness defects in HANDOFF §6: call what production called,
        # on what production called it WITH.
        seen.setdefault("chunk", chunk)
        return out

    codex_extractor.extract_entities = spy
    turn = sample_turns(1)[1]          # middle conversation, middle turn
    print(f"  real corpus turn: conv {turn['conv']} · turn {turn['turn']} "
          f"· {turn['chars']} chars")
    t0 = time.time()
    triplets = codex_extractor.extract_triplets(
        turn["text"], model_override="qwen3:4b-instruct")
    elapsed = time.time() - t0
    codex_extractor.extract_entities = real

    check(seen.get("tier") == "background",
          "call site received tier=background", repr(seen.get("tier")))
    got = seen.get("labels")
    check(got is not None and "concept" in got and "object" in got,
          "call site received the WIDE label list",
          f"{len(got) if got else 0} labels")

    print("\n── 4. did NuNER actually USE the two opened-up types? ──")
    ents = seen.get("entities") or []
    chunk = seen.get("chunk") or ""
    print(f"  mirroring the SAME chunk production saw: {len(chunk)} chars "
          f"of the turn's {turn['chars']}")
    mirrored = background_with_labels(chunk, got or [])
    names = [n for n, _ in mirrored]
    check(names == ents, "mirror matches production output",
          f"{len(names)} vs {len(ents)}")
    hist = Counter(lab for _, lab in mirrored)
    total = sum(hist.values()) or 1
    opened = hist.get("concept", 0) + hist.get("object", 0)
    print("  label histogram: " + ", ".join(
        f"{k} {v} ({100*v/total:.0f}%)" for k, v in hist.most_common(8)))
    check(opened > 0,
          "the opened-up types are actually assigned, not inert",
          f"concept {hist.get('concept', 0)} + object {hist.get('object', 0)} "
          f"= {opened} of {total} ({100*opened/total:.0f}%)")
    print(f"  entities ({len(ents)}): {ents[:30]}")
    print(f"  triplets produced: {len(triplets)} "
          f"(NOT a metric — ±60% run-to-run at temp 0)")
    print(f"  wall clock for this turn's full extraction: {elapsed:.1f}s")

    print("\n── 5. NER cost per turn, both arms, same 3 turns ──")
    print("  (the NER call only — extraction is dominated by the LLM, but this")
    print("   is the part the arm actually changes)")
    embedder = get_embedder()
    from src.retrieval.ner_utils import extract_entities as raw_extract
    for t in sample_turns(1):
        ta = time.time(); a = raw_extract(t["text"], embedder, tier="preflight")
        ta = time.time() - ta
        tb = time.time(); b = raw_extract(t["text"], embedder,
                                          tier="background", labels=got)
        tb = time.time() - tb
        print(f"    conv {t['conv']} {t['chars']:6d} chars  "
              f"micro {ta*1000:7.0f} ms ({len(a):3d} ents)   "
              f"nuner-wide {tb*1000:7.0f} ms ({len(b):3d} ents)")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"⛔ {len(FAILURES)} CHECK(S) FAILED — DO NOT START ARM B:")
        for f in FAILURES:
            print(f"     - {f}")
        return 1
    print("✅ ARM B CONFIG VERIFIED END TO END — safe to start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
