#!/usr/bin/env python3
"""G63 Phase 1 — the guard sweep. Every condition, same turns, one table.

Runs the open rows of `docs/specs/G63_extractor_decision.md` in one pass so the
comparison is against ONE baseline on ONE sample:

  B1   ice_baseline   ICE's production path, qwen3:4b-instruct
  G1   nu_ref         NuExtract3 template + NuNER list, whole turn, 3000 tok
  G5   nu_vocab       ... + the relation vocabulary offered as a preference
  G6   nu_canonrule   ... + the canonicalisation rule only
  G7   nu_chunked     ... but chunked at `chunk_tokens` like ICE does
  G8   nu_tok1200     ... but with ICE's 1200-token cap
  G10  nu_micro       ... but the entity list comes from the micro-NER

⚑ EVERY condition is scored by the SAME post-filters ICE applies to itself, and
relations are counted AFTER `canonical_relation` for all of them — early runs
compared ICE's canonicalised count against NuExtract3's raw one, which flattered
NuExtract3. Fixed here (row G9).

⚑ Checkpoints after every condition. This is a ~70 minute unattended run and a
crash in the last condition must not cost the first six.
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
    _build_grouped_relation_block, _ground_triplets, _grounding_ner_labels,
    canonical_relation, embedder, extract_triplets, is_clausal_relation,
    is_unusable_entity_name, known_relations,
)
from src.memory.chunking import chunk_text         # noqa: E402

OUT = Path("experiments/curation_files/extractor_ab")
NU = "hf.co/numind/NuExtract3-GGUF:Q8_0"
ICE_MODEL = "qwen3:4b-instruct"
TPL = '{"facts": [{"subject": "", "relation": "", "object": ""}]}'

CANON_RULE = (
    "\nRule: canonicalise every subject and object — lowercase, singular, no "
    "punctuation, concise. Write `postgresql` not `PostgreSQL`, `goo blade` not "
    "`the goo blade`, `technical university of munich` not "
    "`Technical University of Munich (TUM)`.\n")


def entity_block(ner):
    if not ner:
        return ""
    return ("\n\nCONFIRMED ENTITIES (use ONLY these as subjects, and as objects "
            "for relations between two entities; do NOT introduce named entities "
            f"not in this list):\n{', '.join(dict.fromkeys(ner))}")


def nu_call(body: str, num_predict: int):
    p = f"<|input|>\n### Template:\n{TPL}\n### Text:\n{body}\n<|output|>\n"
    r = httpx.post("http://localhost:11434/api/generate", timeout=900,
                   json={"model": NU, "prompt": p, "stream": False,
                         "options": {"temperature": 0.0, "num_predict": num_predict}})
    raw = r.json().get("response", "")
    out = raw.split("</think>")[-1].strip()
    try:
        facts = json.loads(out).get("facts", [])
    except Exception:
        return None
    keep = []
    for f in facts:
        s, rel, o = f.get("subject"), f.get("relation"), f.get("object")
        if all(isinstance(x, str) and x.strip() for x in (s, rel, o)):
            keep.append({"subject": s.strip(), "relation": rel.strip(),
                         "object": str(o).strip()})
    return keep


def score(facts, turn, ner, known):
    """⚑ The SAME gates ICE applies to itself, plus canonicalisation (row G9)."""
    if facts is None:
        return None
    canon = []
    for f in facts:
        c = canonical_relation(f["relation"], known)
        canon.append({**f, "relation": (c or f["relation"])})
    g, rej = _ground_triplets(
        [{"subject": f["subject"], "relation": f["relation"], "object": f["object"]}
         for f in canon], ner, turn)
    rels = Counter(f["relation"] for f in canon)
    return {
        "n": len(canon), "grounded": len(g), "rejected": len(rej),
        "unusable": sum(1 for f in canon if is_unusable_entity_name(f["subject"])
                        or is_unusable_entity_name(f["object"])),
        "clausal": sum(1 for f in canon if is_clausal_relation(f["relation"])),
        "distinct_rel": len(rels),
        "singleton_rel": sum(1 for v in rels.values() if v == 1),
        "facts": canon,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=60)
    ap.add_argument("--seed", default="g63-sweep-20260824")
    ap.add_argument("--only", default="", help="comma-separated condition names")
    args = ap.parse_args()

    db = SessionLocal()
    rows = db.execute(text("""
        select id::text as id, raw_text, topic_tags from episodic_memory
        where lossless_flag = true and length(raw_text) between 600 and 6000
    """)).fetchall()
    db.close()
    picks = random.Random(args.seed).sample(list(rows), min(args.turns, len(rows)))
    print(f"{len(picks)} turns · seed {args.seed}", flush=True)

    labels = _grounding_ner_labels()
    known = known_relations() if settings.codex_relation_open_vocabulary else []
    vocab_block = "\n\nPREFER these relation words where one fits:\n" \
                  + _build_grouped_relation_block()

    # ⚑ NER computed ONCE per turn per tier and reused by every condition.
    print("building NER lists (both tiers)...", flush=True)
    ner_bg, ner_micro = {}, {}
    for i, r in enumerate(picks, 1):
        ner_bg[r.id] = extract_entities(r.raw_text, embedder, tier="background",
                                        labels=labels)
        ner_micro[r.id] = extract_entities(r.raw_text, embedder, tier="preflight",
                                           labels=None)
        if i % 20 == 0:
            print(f"  ner {i}/{len(picks)}", flush=True)

    CONDS = {
        "ice_baseline": None,
        "nu_ref":       dict(vocab=False, canon=False, chunk=False, tok=3000, ner="bg"),
        "nu_vocab":     dict(vocab=True,  canon=False, chunk=False, tok=3000, ner="bg"),
        "nu_canonrule": dict(vocab=False, canon=True,  chunk=False, tok=3000, ner="bg"),
        "nu_chunked":   dict(vocab=False, canon=False, chunk=True,  tok=3000, ner="bg"),
        "nu_tok1200":   dict(vocab=False, canon=False, chunk=False, tok=1200, ner="bg"),
        "nu_micro":     dict(vocab=False, canon=False, chunk=False, tok=3000, ner="micro"),
        # ⚑ EVERY guard that measured a WIN, together, and nothing else.
        # Entity list (grounding 6.6->38.2) · vocabulary (in-vocab 50.9->75.2) ·
        # canonicalisation rule (caps 54->30) · chunking (failures 3->0) ·
        # 3000 tokens (1200 loses half the turns). The ONE guard excluded is
        # ICE's nine-rule prompt, the only one that measured a loss.
        "nu_combined":  dict(vocab=True,  canon=True,  chunk=True,  tok=3000, ner="bg"),
        # ⚑ G12 — THE CANDIDATE PRODUCTION CONFIG. Every guard that survived the
        # 2026-08-24 judged isolation, and NOT the one that failed it.
        # vocabulary alone measured 22% correct (-38 pts vs the bare template);
        # canonicalisation alone 80%; chunking alone 80%. This is those two,
        # plus the entity list and a real token budget, with NO vocabulary.
        # Never run as a combination — `nu_combined` carried the vocabulary.
        "nu_novocab":   dict(vocab=False, canon=True,  chunk=True,  tok=3000, ner="bg"),
    }
    if args.only:
        keep = set(args.only.split(","))
        CONDS = {k: v for k, v in CONDS.items() if k in keep}

    results = {}
    ck = OUT / "g63_sweep.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if ck.exists():
        results = json.load(open(ck)).get("conditions", {})
        print(f"resuming; already have: {sorted(results)}", flush=True)

    for name, cfg in CONDS.items():
        if name in results:
            continue
        agg, allf, fails = Counter(), [], 0
        t0 = time.time()
        print(f"\n>>> {name}", flush=True)
        for i, r in enumerate(picks, 1):
            ner = ner_bg[r.id] if (cfg or {}).get("ner") == "bg" else ner_micro[r.id]
            if cfg is None:                       # ICE production path
                ner = ner_bg[r.id]
                try:
                    out = extract_triplets(r.raw_text, ICE_MODEL,
                                           topic_tags=r.topic_tags)
                    facts = [{"subject": t["subject"], "relation": t["relation"],
                              "object": t["object"]} for t in out
                             if isinstance(t, dict) and all(
                                 isinstance(t.get(k), str)
                                 for k in ("subject", "relation", "object"))]
                except Exception as exc:
                    print(f"  ! turn {i}: {type(exc).__name__}", flush=True)
                    facts = None
            else:
                body = r.raw_text
                if cfg["canon"]:
                    body += CANON_RULE
                if cfg["vocab"]:
                    body += vocab_block
                body += entity_block(ner)
                if cfg["chunk"]:
                    # ⚑ The chunked path must carry the SAME additions as the
                    # whole-turn path, per chunk. An earlier version appended
                    # only the entity block, which would have silently dropped
                    # the vocabulary and canonicalisation rule from any chunked
                    # condition — i.e. `nu_combined` would not have been combined.
                    suffix = ""
                    if cfg["canon"]:
                        suffix += CANON_RULE
                    if cfg["vocab"]:
                        suffix += vocab_block
                    suffix += entity_block(ner)
                    facts = []
                    for ch in chunk_text(r.raw_text):
                        part = nu_call(ch + suffix, cfg["tok"])
                        if part:
                            facts += part
                else:
                    facts = nu_call(body, cfg["tok"])
            s = score(facts, r.raw_text, ner, known)
            if s is None:
                fails += 1
            else:
                for k in ("n", "grounded", "rejected", "unusable", "clausal",
                          "distinct_rel", "singleton_rel"):
                    agg[k] += s[k]
                for f in s["facts"]:
                    allf.append({**f, "turn_id": r.id, "cond": name})
            if i % 15 == 0:
                print(f"  {i}/{len(picks)}", flush=True)
        results[name] = {"agg": dict(agg), "fails": fails,
                         "sec_per_turn": (time.time() - t0) / len(picks),
                         "facts": allf}
        ck.write_text(json.dumps({"seed": args.seed, "turns": len(picks),
                                  "conditions": results}, indent=1))
        print(f"  done: {agg['n']} facts, {fails} failed, "
              f"{(time.time()-t0)/len(picks):.1f}s/turn  [checkpointed]", flush=True)

    print(f"\n{'condition':<15}{'facts':>7}{'grnd%':>7}{'junk%':>7}"
          f"{'rels':>6}{'sing%':>7}{'fail':>6}{'s/turn':>8}")
    print("-" * 63)
    for name, r in results.items():
        a = r["agg"]
        n = max(a.get("n", 0), 1)
        print(f"{name:<15}{a.get('n',0):>7}"
              f"{100*a.get('grounded',0)/n:>7.1f}"
              f"{100*a.get('unusable',0)/n:>7.1f}"
              f"{a.get('distinct_rel',0):>6}"
              f"{100*a.get('singleton_rel',0)/max(a.get('distinct_rel',1),1):>7.1f}"
              f"{r['fails']:>6}{r['sec_per_turn']:>8.1f}")
    print(f"\nwrote {ck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
