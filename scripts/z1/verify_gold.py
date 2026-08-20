#!/usr/bin/env python3
"""Does the GROUND TRUTH check out? Do the gold turns actually contain the answer?

Distinct from the source-truncation defect (fixed 2026-08-20). That was the
judge being shown too little of the gold. **This asks whether the gold is RIGHT
AT ALL** — a probe whose `gold_turns` do not contain its own `answer` scores
every arm as a failure no matter how well retrieval works, and no amount of
system improvement can pass it.

It matters for two metrics at once:
  · `score_typed` computes recall/coverage AGAINST these turn ids;
  · `judge_answers` shows these turns to the judge as SOURCE.
If the gold is wrong, both are measuring against a bad reference and every
number in the run inherits it.

Reads the FULL gold turns (no caps) and asks whether the probe's own expected
answer is supported by them. No arm, no retrieval, no system under test —
just probe vs corpus.

Run:
  uv run python scripts/z1/verify_gold.py --n 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx                                       # noqa: E402
from sqlalchemy import text as _sql                # noqa: E402

from src.api.config import settings                # noqa: E402
from src.api.db import SessionLocal                # noqa: E402

PROBES = Path("experiments/curation_files/typed_probes.stratified120.json")
MANIFEST = Path("experiments/curation_files/seeded_store.json")

SYSTEM = """You check whether a question's stated ANSWER is actually supported by \
the SOURCE turns it was labelled against.

You are NOT judging an AI system. You are checking the label itself.

Return exactly one:
supported     - the SOURCE contains what the ANSWER asserts.
partial       - the SOURCE contains some of it; material parts are absent.
unsupported   - the SOURCE does not contain the ANSWER's substance at all.
unanswerable  - the QUESTION cannot be answered from the SOURCE by anyone, \
because the SOURCE simply does not address it.

Return ONLY JSON: {"label":"<label>","why":"<max 15 words>"}"""


def build_index(db):
    man = json.loads(MANIFEST.read_text())
    raw = {r[0]: r[1] for r in db.execute(
        _sql("select id::text, raw_text from episodic_memory")).fetchall()}
    idx = {}
    for cid, d in man["conversations"].items():
        for t in d["turns"]:
            txt = raw.get(t["episodic_id"])
            if txt:
                idx[(cid, int(t["turn_number"]))] = txt
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--probes", default=str(PROBES))
    args = ap.parse_args()

    probes = json.loads(Path(args.probes).read_text())["probes"][:args.n]
    db = SessionLocal()
    idx = build_index(db)
    db.close()

    client = httpx.Client(base_url=settings.probe_api_base_url,
                          headers={"Authorization": f"Bearer {settings.probe_api_key}"})
    tally, by_type, missing = Counter(), {}, 0
    for i, p in enumerate(probes, 1):
        turns = [idx.get((p["conversation"], int(t))) for t in (p.get("gold_turns") or [])]
        turns = [t for t in turns if t]
        if not turns:
            missing += 1
            continue
        src = "\n\n".join(turns)
        user = (f"QUESTION:\n{p['question']}\n\n"
                f"STATED ANSWER:\n{p.get('answer','')}\n\n"
                f"SOURCE ({len(turns)} turns):\n{src}")
        try:
            r = client.post("/chat/completions", json={
                "model": settings.probe_model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}],
                "temperature": 0.0, "max_tokens": 300,
                "response_format": {"type": "json_object"}}, timeout=240)
            r.raise_for_status()
            lab = str(json.loads(
                r.json()["choices"][0]["message"]["content"]).get("label", "")).lower()
        except Exception as exc:
            print(f"  ! probe {i}: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        tally[lab] += 1
        by_type.setdefault(p["probe_type"], Counter())[lab] += 1
        if i % 20 == 0:
            print(f"  {i}/{len(probes)}")

    tot = sum(tally.values()) or 1
    print(f"\n=== IS THE GROUND TRUTH SOUND? (n={tot}, gold turns UNTRUNCATED) ===")
    for lab in ("supported", "partial", "unsupported", "unanswerable"):
        print(f"  {lab:<13} {tally[lab]:>3}  ({100*tally[lab]/tot:5.1f}%)")
    bad = tally["unsupported"] + tally["unanswerable"]
    print(f"\n  ⇒ probes whose OWN gold cannot support their OWN answer: "
          f"{bad} ({100*bad/tot:.1f}%)  <- unpassable by any system")
    if missing:
        print(f"  ⚠ {missing} probes had no resolvable gold turns at all")
    print("\nBY TYPE:")
    for k in sorted(by_type):
        c = by_type[k]; n = sum(c.values()) or 1
        print(f"  {k:<20} n={n:>3}  supported {100*c['supported']/n:5.1f}%  "
              f"partial {100*c['partial']/n:5.1f}%  "
              f"unsupported {100*c['unsupported']/n:5.1f}%  "
              f"unanswerable {100*c['unanswerable']/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
