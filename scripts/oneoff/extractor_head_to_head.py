#!/usr/bin/env python3
"""ICE's production extractor vs NuExtract3, same turns, fair sample.

⚑ WHY THIS EXISTS. The 2026-08-24 look at NuExtract3 used turns SELECTED
BECAUSE ICE REVERSED A FACT ON THEM. A merely-average model looks brilliant on a
sample chosen for the incumbent's failures. This draws turns at random instead.

⚑ MOST OF THIS NEEDS NO JUDGE. Entity quality, relation-vocabulary explosion and
name usability are computed by running **ICE's own gates** over BOTH extractors'
output — `_ground_triplets`, `is_unusable_entity_name`, `is_clausal_relation`.
Those are counts, not opinions, and counting the store directly is the one class
of measurement that survived scrutiny this week.

Only DIRECTION needs judgement, and no model we have judges it reliably
(best available scores 60% against the maintainer's labels). So this writes a
BLIND sheet: facts from both extractors, shuffled, source hidden.

ICE side calls `extract_triplets()` — the real production function, not a
reimplementation of its prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx                                       # noqa: E402
from sqlalchemy import text                        # noqa: E402

from src.api.config import settings                # noqa: E402
from src.api.db import SessionLocal                # noqa: E402
from src.retrieval.ner_utils import extract_entities   # noqa: E402
from src.workers.codex_extractor import (          # noqa: E402
    _ground_triplets, _grounding_ner_labels, embedder, extract_triplets,
    is_clausal_relation, is_unusable_entity_name,
)

OUT = Path("experiments/curation_files/extractor_ab")
TPL = '{"facts": [{"subject": "", "relation": "", "object": ""}]}'


def nuextract(turn: str, model: str, ner_entities=None, num_predict: int = 3000):
    """⚑ Template-filling, not instructions. Reasoning model — split on </think>.

    ⚑ FAIRNESS. `ner_entities` exists because the first version of this
    comparison was rigged. ICE's prompt is HANDED the NER-confirmed entity list
    ("use ONLY these as subjects"), and the scoring then asks how many extracted
    entities are on that list. ICE scored 70.3% and NuExtract3 39.7% — a
    measurement of who was told the answer, not who is better. Both extractors
    now receive the same list.
    """
    body = turn
    if ner_entities:
        confirmed = ", ".join(dict.fromkeys(ner_entities))
        body = (f"{turn}\n\nCONFIRMED ENTITIES (use ONLY these as subjects, and as "
                f"objects for relations between two entities; do NOT introduce "
                f"named entities not in this list):\n{confirmed}")
    p = f"<|input|>\n### Template:\n{TPL}\n### Text:\n{body}\n<|output|>\n"
    r = httpx.post("http://localhost:11434/api/generate", timeout=900,
                   json={"model": model, "prompt": p, "stream": False,
                         "options": {"temperature": 0.0, "num_predict": num_predict}})
    body = r.json().get("response", "")
    body = body.split("</think>")[-1].strip()
    try:
        facts = json.loads(body).get("facts", [])
    except Exception:
        return None
    out = []
    for f in facts:
        s, rel, o = f.get("subject"), f.get("relation"), f.get("object")
        if all(isinstance(x, str) and x.strip() for x in (s, rel, o)):
            out.append({"subject": s.strip(), "relation": rel.strip(),
                        "object": str(o).strip()})
    return out


def gates(facts: list, turn: str, ner_entities: list) -> dict:
    """Run ICE's OWN write-path gates over any extractor's output.

    This is the fair part: whatever ICE would refuse from itself, it refuses
    from NuExtract3 too. No judge, no opinion — the same predicates the
    production write path applies.
    """
    if not facts:
        return {}
    grounded, rejected = _ground_triplets(
        [{"subject": f["subject"], "relation": f["relation"], "object": f["object"]}
         for f in facts], ner_entities, turn)
    unusable = sum(1 for f in facts
                   if is_unusable_entity_name(f["subject"])
                   or is_unusable_entity_name(f["object"]))
    clausal = sum(1 for f in facts if is_clausal_relation(f["relation"]))
    rels = Counter(f["relation"] for f in facts)
    return {
        "n": len(facts),
        "grounded": len(grounded),
        "rejected": len(rejected),
        "unusable_names": unusable,
        "clausal_relations": clausal,
        "distinct_relations": len(rels),
        "singleton_relations": sum(1 for v in rels.values() if v == 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--seed", default="ab-20260824")
    ap.add_argument("--ice-model", default="qwen3:4b-instruct",
                    help="the model EVERY published ICE measurement used")
    ap.add_argument("--nu-model", default="hf.co/numind/NuExtract3-GGUF:Q8_0")
    args = ap.parse_args()

    db = SessionLocal()
    rows = db.execute(text("""
        select id::text as id, raw_text, topic_tags
        from episodic_memory
        where lossless_flag = true and length(raw_text) between 600 and 6000
    """)).fetchall()
    db.close()
    rng = random.Random(args.seed)
    picks = rng.sample(list(rows), min(args.turns, len(rows)))
    print(f"{len(picks)} RANDOM lossless turns (seed {args.seed})")
    print(f"  ICE        : {args.ice_model}  (via extract_triplets, production path)")
    print(f"  NuExtract3 : {args.nu_model}\n")

    ner_labels = _grounding_ner_labels()
    agg = {"ice": Counter(), "nu": Counter()}
    per_turn, all_facts = [], []
    t_ice = t_nu = 0.0
    fails = {"ice": 0, "nu": 0}

    for i, row in enumerate(picks, 1):
        turn = row.raw_text

        # ⚑ ONE NER list, built FIRST, exactly the way `extract_triplets` builds
        # its own (same tagger, same tier, same labels). It is handed to BOTH
        # extractors and used for BOTH grounding checks. Building it after the
        # extractions — as the first version of this script did — meant only ICE
        # received it, and the grounding comparison measured who was told.
        try:
            ner = extract_entities(turn, embedder,
                                   tier=settings.codex_extraction_ner_tier,
                                   labels=ner_labels)
        except Exception as exc:
            print(f"  ! turn {i} NER failed: {type(exc).__name__}: {str(exc)[:70]}")
            ner = []

        # --- ICE, real production function
        s = time.time()
        try:
            ice = extract_triplets(turn, args.ice_model, topic_tags=row.topic_tags)
            ice = [{"subject": t["subject"], "relation": t["relation"],
                    "object": t["object"]} for t in ice
                   if isinstance(t, dict) and all(
                       isinstance(t.get(k), str) for k in ("subject", "relation", "object"))]
        except Exception as exc:
            print(f"  ! turn {i} ICE failed: {type(exc).__name__}: {str(exc)[:80]}")
            ice, fails["ice"] = [], fails["ice"] + 1
        t_ice += time.time() - s

        # --- NuExtract3, given the SAME confirmed-entity list ICE gets
        s = time.time()
        nu = nuextract(turn, args.nu_model, ner_entities=ner)
        if nu is None:
            print(f"  ! turn {i} NuExtract3 unparseable")
            nu, fails["nu"] = [], fails["nu"] + 1
        t_nu += time.time() - s

        g_ice = gates(ice, turn, ner)
        g_nu = gates(nu, turn, ner)
        for k, v in g_ice.items():
            agg["ice"][k] += v
        for k, v in g_nu.items():
            agg["nu"][k] += v
        per_turn.append({"turn_id": row.id, "ice": g_ice, "nu": g_nu})
        for src_name, facts in (("ice", ice), ("nu", nu)):
            for f in facts:
                all_facts.append({**f, "src": src_name, "turn_id": row.id})
        if i % 5 == 0:
            print(f"  {i}/{len(picks)} turns", flush=True)

    n = len(picks)
    print(f"\n=== HEAD TO HEAD — {n} random turns ===")
    print(f"{'metric':<26}{'ICE':>12}{'NuExtract3':>14}")
    print("-" * 52)
    rows_out = [
        ("facts extracted", "n"),
        ("  grounded by ICE's NER", "grounded"),
        ("  rejected by ICE's NER", "rejected"),
        ("unusable entity names", "unusable_names"),
        ("clausal relations", "clausal_relations"),
        ("distinct relations", "distinct_relations"),
        ("  used exactly once", "singleton_relations"),
    ]
    for label, k in rows_out:
        print(f"{label:<26}{agg['ice'][k]:>12}{agg['nu'][k]:>14}")
    for name in ("ice", "nu"):
        a = agg[name]
        if a["n"]:
            a["_ground_pct"] = round(100 * a["grounded"] / a["n"], 1)
            a["_singleton_pct"] = round(100 * a["singleton_relations"]
                                        / max(a["distinct_relations"], 1), 1)
    print(f"\n{'grounded %':<26}{agg['ice'].get('_ground_pct',0):>11}%"
          f"{agg['nu'].get('_ground_pct',0):>13}%")
    print(f"{'singleton relation %':<26}{agg['ice'].get('_singleton_pct',0):>11}%"
          f"{agg['nu'].get('_singleton_pct',0):>13}%")
    print(f"\n{'seconds/turn':<26}{t_ice/n:>12.1f}{t_nu/n:>14.1f}")
    print(f"{'turns failed':<26}{fails['ice']:>12}{fails['nu']:>14}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "head_to_head.json").write_text(json.dumps({
        "seed": args.seed, "turns": n,
        "ice_model": args.ice_model, "nu_model": args.nu_model,
        "aggregate": {k: dict(v) for k, v in agg.items()},
        "per_turn": per_turn, "facts": all_facts,
        "seconds_per_turn": {"ice": t_ice / n, "nu": t_nu / n},
        "failed_turns": fails,
    }, indent=1))
    print(f"\nwrote {OUT}/head_to_head.json   ({len(all_facts)} facts for blind labelling)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
