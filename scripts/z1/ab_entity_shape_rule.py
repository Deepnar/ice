#!/usr/bin/env python3
"""Paired A/B: does a SHAPE rule in the extraction prompt stop fragment entities?

**The question.** `_ground_triplets` cannot reject `what the flaw` without also
rejecting `emotional validation` — both are a confirmed entity plus extra
tokens, both occur verbatim in the turn (measured 2026-08-17: option A demotes
24.6% including real entities, option C rejects 0 of 6 fragments). So the only
remaining place to act is where the string is produced: the prompt.

**The metric, and why it is not a fragment lexicon.** A hand-written lexicon was
tried and over-credited badly (`get validation` scored as a real link). Instead
the primary metric is the MECHANISM, which is deterministic: what fraction of
returned subjects/objects ground ONLY via the superset arm — i.e. they contain a
NER-confirmed entity plus something else. That is exactly the population the
fragments live in, and it needs no judgement to count.

Reported per turn and paired, because absolute counts on this pipeline vary
±60% run to run (PROVENANCE 2026-08-03: the same arm scored 104 and 166 on the
same turns at temperature 0). Only within-pair deltas are trustworthy.

Raw triplets are checkpointed per turn to a sidecar so a killed run is
salvageable and a metric change is a re-read, not a re-run (TRAPS #36).

Run:
  uv run python scripts/z1/ab_entity_shape_rule.py --turns 30
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text  # noqa: E402

from scripts.z1.run_meta import run_meta  # noqa: E402
from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402

OUT = Path("experiments/curation_files/ab_entity_shape")


def _arms(term, ner_entities, normalize):
    """Which grounding arm admits this term: exact / subset / superset-only."""
    nt = normalize(term)
    if not nt:
        return None
    norm_ents, tok_sets = set(), []
    for e in ner_entities:
        ne = normalize(e)
        if ne:
            norm_ents.add(ne)
            tok_sets.append(frozenset(ne.split()))
    tt = frozenset(nt.split())
    if nt in norm_ents:
        return "exact"
    if any(tt <= e for e in tok_sets):
        return "subset"
    if any(e <= tt for e in tok_sets):
        return "superset"       # ← the fragment mechanism
    return "ungrounded"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--model", default="qwen3:4b-instruct",
                    help="must match the arm that seeded the store")
    ap.add_argument("--tag", default="entity-shape")
    args = ap.parse_args()

    from src.memory.embedder import get_embedder
    from src.retrieval.ner_utils import extract_entities
    from src.workers.codex_extractor import _normalize_term, extract_triplets

    db = SessionLocal()
    # ⚑ STRATIFY BY CONVERSATION, AND DO NOT CAP TURN LENGTH. Two sampling bugs
    # in one query, both found 2026-08-17 after the run they invalidated:
    #
    #  1. `ORDER BY id LIMIT 30` drew 25 of 30 turns from ONE conversation, with
    #     zero from the 87-turn narrative one where every fragment entity
    #     (`what the flaw`, `to krishna`, `when orien`) lives.
    #  2. `length BETWEEN 300 AND 2000` then excluded **238 of 293 turns (81%)**
    #     and did so UNEVENLY — median turn length is 2540 / 6253 / 5264 chars
    #     per conversation, so the cap kept 45 turns of one conversation and
    #     **2 of 87** of the narrative one. Long turns are also exactly where
    #     copied spans should concentrate.
    #
    # The extractor chunks internally (A1), so long turns are handled; they only
    # cost more calls. Floor stays at 300 to skip trivial turns.
    rows = db.execute(text("""
        SELECT raw_text, conversation_id FROM (
            SELECT raw_text, conversation_id,
                   row_number() OVER (PARTITION BY conversation_id ORDER BY id) AS rn
            FROM episodic_memory
            WHERE idempotency_key LIKE 'z1seed-%'
              AND length(raw_text) >= 300
        ) s
        ORDER BY rn, conversation_id
        LIMIT :n
    """), {"n": args.turns}).fetchall()
    turns = [r.raw_text for r in rows]
    from collections import Counter as _C
    spread = _C(str(r.conversation_id)[:8] for r in rows)
    print(f"paired A/B over {len(turns)} turns · model={args.model}")
    print(f"  conversation spread: {dict(spread)}")

    embedder = get_embedder()
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sidecar = OUT / f"{stamp}_{args.tag}.raw.jsonl"

    per_turn = []
    for i, turn in enumerate(turns, 1):
        ner = extract_entities(turn, embedder)
        rec = {"turn_index": i, "n_ner": len(ner), "ner": ner[:40], "arms": {}}
        for arm, flag in (("off", False), ("on", True)):
            settings.codex_extraction_entity_shape_rule = flag
            try:
                trips = extract_triplets(turn, model_override=args.model)
            except Exception as err:                      # a dead arm must be visible
                print(f"  turn {i} arm {arm} FAILED: {type(err).__name__}: {err}")
                rec["arms"][arm] = {"error": f"{type(err).__name__}: {err}"}
                continue
            counts = {"exact": 0, "subset": 0, "superset": 0, "ungrounded": 0}
            terms = []
            for t in trips:
                for slot in ("subject", "object"):
                    a = _arms(t.get(slot, ""), ner, _normalize_term)
                    if a:
                        counts[a] += 1
                        terms.append({"term": t.get(slot, ""), "arm": a, "slot": slot})
            total = sum(counts.values())
            rec["arms"][arm] = {
                "triplets": len(trips), "terms": total, "counts": counts,
                "superset_rate": round(counts["superset"] / total, 4) if total else None,
                "grounded_rate": round(
                    (counts["exact"] + counts["subset"] + counts["superset"]) / total, 4
                ) if total else None,
                "terms_detail": terms,          # so a metric change is a re-read
            }
        settings.codex_extraction_entity_shape_rule = False
        per_turn.append(rec)
        with sidecar.open("a") as fh:           # checkpoint EVERY turn (TRAPS #36)
            fh.write(json.dumps(rec, default=str) + "\n")
        a_off, a_on = rec["arms"].get("off", {}), rec["arms"].get("on", {})
        print(f"  turn {i:>3}: off superset={a_off.get('superset_rate')} "
              f"({a_off.get('triplets')} trip) | "
              f"on superset={a_on.get('superset_rate')} ({a_on.get('triplets')} trip)")

    def _series(arm, key):
        return [r["arms"][arm][key] for r in per_turn
                if arm in r["arms"] and r["arms"][arm].get(key) is not None]

    print(f"\n{'='*70}\nPAIRED RESULT  ({len(per_turn)} turns)\n{'='*70}")
    summary = {}
    for key in ("superset_rate", "grounded_rate", "triplets"):
        off, on = _series("off", key), _series("on", key)
        if not off or not on:
            continue
        pairs = [(r["arms"]["off"][key], r["arms"]["on"][key]) for r in per_turn
                 if r["arms"].get("off", {}).get(key) is not None
                 and r["arms"].get("on", {}).get(key) is not None]
        deltas = [b - a for a, b in pairs]
        wins = sum(1 for d in deltas if d < 0)
        ties = sum(1 for d in deltas if d == 0)
        summary[key] = {
            "off_mean": round(statistics.mean(off), 4),
            "on_mean": round(statistics.mean(on), 4),
            "median_delta": round(statistics.median(deltas), 4) if deltas else None,
            "on_lower_count": wins, "ties": ties, "n_pairs": len(pairs),
        }
        print(f"  {key:<16} off={summary[key]['off_mean']:<9} "
              f"on={summary[key]['on_mean']:<9} "
              f"median Δ={summary[key]['median_delta']:<9} "
              f"on-lower {wins}/{len(pairs)} (ties {ties})")

    print("\n  ⚑ superset_rate DOWN is the win. triplets DOWN a lot is the cost —")
    print("    the permissive whitelist exists to raise yield, so a fix that")
    print("    halves triplet count is buying precision with real facts.")

    out = OUT / f"{stamp}_{args.tag}.json"
    out.write_text(json.dumps({
        "meta": run_meta(script=__file__, args=vars(args),
                         settings_keys=["codex_extraction_entity_shape_rule",
                                        "codex_conf_grounded", "codex_conf_rejected",
                                        "codex_relation_open_vocabulary"],
                         extra={"model": args.model, "n_turns": len(turns)}),
        "summary": summary,
        "per_turn": per_turn,
    }, indent=1, default=str))
    print(f"\nwrote {out}\n      {sidecar}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
