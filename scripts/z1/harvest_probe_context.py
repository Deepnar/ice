#!/usr/bin/env python3
"""Z1: dump what retrieval ACTUALLY returned for a sample of probes, for reading.

**Why this exists.** Recall@k answers one question — did the gold turn come
back — and it is blind in both directions that matter. A "hit" can return the
gold turn buried among junk that would not help a model answer. A "miss" can
return an equally good turn, which on this probe set is the common case:
**65% of the 592 probes have another turn matching the question at least as
well** (measured 2026-08-13 with the fixed ambiguity guard). A number computed
over that set is not wrong so much as unable to say what it is claiming.

So this writes the raw material for a human or an agent to read: the question,
the gold turn, and **every fragment ICE actually returned, in order, with its
leg and token count**. It computes no score and reaches no verdict on purpose —
the reading is the instrument, and every finding that survived scrutiny on
2026-08-12 came from someone reading output against green metrics.

Runs the production path (B2, reliance, real budget chain), same as the scorer.

Run:
  uv run python scripts/z1/harvest_probe_context.py --sample 25
  uv run python scripts/z1/harvest_probe_context.py --sample 25 --only-misses
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import func, text  # noqa: E402

from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402

PROBES = Path("experiments/curation_files/generated_probes.json")
OUT = Path("experiments/curation_files/probe_context")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--only-misses", action="store_true",
                    help="only probes whose gold turn did NOT come back")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="read")
    # ⚠ The default is the OLD 592-probe set, which score_typed.py does NOT
    # use — it reads typed_probes.json (420, carrying probe_type). Harvesting
    # one set and scoring the other means the read pass judges probes that were
    # never scored, on a set whose fixed ambiguity guard rejects 385 of 592.
    # Pass --probes experiments/curation_files/typed_probes.json to line them up.
    ap.add_argument("--probes", default=None,
                    help="probe file; default is the legacy untyped set")
    args = ap.parse_args()

    probe_path = Path(args.probes) if args.probes else PROBES
    probes = json.loads(probe_path.read_text())["probes"]
    print(f"probes: {len(probes)} from {probe_path}")
    random.Random(args.seed).shuffle(probes)

    db = SessionLocal()
    settings.codex_reinforce_increment = 0.0
    settings.decay_strengthen_amount = 0.0
    settings.retrieval_strengthen_writes = False

    # ⚑ TWO IDENTIFIER SPACES, AND THEY ARE NOT INTERCHANGEABLE. A fragment's
    # `source_batch_id` is the episodic ROW id, but `codex_edges.source_batch`
    # and `procedural_memory.source_batch_ids` hold the turn's BATCH id —
    # verified: 9,662 edges join on batch_id and 0 on row id. Crediting a
    # derived fragment therefore needs BOTH ids for the same turn, or codex and
    # procedural can never match a gold turn and recall silently scores the
    # episodic leg alone (G48).
    conv_of, gold_index, gold_text, gold_batch = {}, {}, {}, {}
    for rid, bid, cid, key, raw in db.execute(text(
            "select id, batch_id, conversation_id, idempotency_key, raw_text "
            "from episodic_memory where idempotency_key like 'z1seed-%'")):
        _, slug, turn = key.split("-", 2)
        conv_of[slug] = str(cid)
        gold_index[(slug, int(turn))] = str(rid)
        gold_batch[str(rid)] = str(bid)
        gold_text[str(rid)] = raw
    if not conv_of:
        print("store not seeded")
        return 1

    from src.api.context_ledger import effective_memory_budget
    from src.api.memory_decision import (decide_memory_retrieval,
                                         derive_total_budget,
                                         estimate_recent_window_tokens)
    from src.classifier.classifier import PyTorchClassifier
    from src.memory.embedder import get_embedder
    from src.memory.models import EpisodicMemory
    from src.memory.tokens import estimate_from_chars
    from src.model_registry.registry import find_best_model, get_model_context_window
    from src.model_registry.runtime_probe import serving_window
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator

    embedder = get_embedder()
    clf = PyTorchClassifier(model_path=settings.classifier_model_path,
                            schema_path=settings.label_schema_path)
    orch = HybridRetrievalOrchestrator(db, embedder)

    meta = {}
    for slug, cid in conv_of.items():
        tc = db.query(EpisodicMemory).filter_by(conversation_id=cid).count()
        ch = db.query(func.coalesce(
            func.sum(func.length(EpisodicMemory.raw_text)), 0)
        ).filter_by(conversation_id=cid).scalar() or 0
        meta[slug] = (tc, estimate_from_chars(ch))

    # Warm-up: the first retrieval of a process builds the relation-gloss cache
    # and differs from steady state (measured 2026-08-13). A sample must not
    # include the instrument's own warm-up.
    first = probes[0]
    if conv_of.get(first["conversation"]):
        c0 = clf.classify(first["question"][:2000])
        c0.context_reliance = "Long_Term_Memory"
        orch.set_budget_from_turn_count(10, total_tokens=1000, classification=c0)
        orch.retrieve(classification=c0,
                      conversation_id=conv_of[first["conversation"]],
                      prompt_embedding=embedder.encode(
                          first["question"], convert_to_tensor=False).tolist(),
                      scope=None)

    records, seen = [], 0
    for p in probes:
        if len(records) >= args.sample:
            break
        slug = p["conversation"]
        conv_id = conv_of.get(slug)
        # Two probe formats. The legacy set carries ONE `gold_turn`; the typed
        # set carries `gold_turns`, a LIST — a summary_synthesis probe can span
        # 17 turns, and reading only the first would judge retrieval against a
        # fraction of its own answer. Accept both, always work with the list.
        gold_turns = p.get("gold_turns")
        if gold_turns is None:
            gold_turns = [p["gold_turn"]] if p.get("gold_turn") is not None else []
        gold_ids = [gold_index.get((slug, t)) for t in gold_turns]
        gold_ids = [g for g in gold_ids if g]
        if not conv_id or not gold_ids:
            continue
        gold_set = {str(g) for g in gold_ids}
        # the same turns, in the other id space
        gold_set |= {gold_batch[str(g)] for g in gold_ids if str(g) in gold_batch}
        seen += 1
        q = p["question"]
        tc, tt = meta[slug]
        c = clf.classify(q[:2000])
        emb = embedder.encode(q, convert_to_tensor=False).tolist()
        model_name, _ = find_best_model(c.topic_tags, c.intent_tags)
        rw = get_model_context_window(model_name)
        win = serving_window(model_name, rw) if settings.context_use_serving_window else rw
        tb = effective_memory_budget(
            derive_total_budget(win, settings), q,
            generation_reserve=settings.context_generation_reserve,
            floor=settings.context_budget_floor)
        d = decide_memory_retrieval(
            c, turn_count=tc, total_tokens=tt, settings=settings,
            recent_window_tokens=estimate_recent_window_tokens(tc, tb),
            timescope_mode="current", coding_scope=False)
        if not d.retrieve:
            records.append({"question": q, "conversation": slug,
                            "probe_type": p.get("probe_type", "untyped"),
                            "gold_turns": gold_turns,
                            "b2_declined": True, "fragments": []})
            continue
        c.context_reliance = "Long_Term_Memory"
        orch.set_budget_from_turn_count(tc, total_tokens=tt, classification=c,
                                        total_budget=tb)
        frags = orch.retrieve(classification=c, conversation_id=conv_id,
                              prompt_embedding=emb, scope=None)
        # Rank of the FIRST gold turn to come back, and how many of them did.
        # For a multi-gold probe rank alone is misleading: a summary probe
        # spanning 17 turns can rank 1 and still have missed 16.
        # G48: credit a fragment whose ORIGIN turns include the gold turn, not
        # only one whose source_batch_id is it. Without this, codex, procedural
        # and timeline fragments can never be credited and recall silently
        # measures the episodic leg alone.
        def _hits(f):
            if f.source_batch_id and str(f.source_batch_id) in gold_set:
                return True
            return any(str(b) in gold_set
                       for b in (getattr(f, "origin_batch_ids", ()) or ()))
        rank = next((i for i, f in enumerate(frags, 1) if _hits(f)), None)
        returned = {str(f.source_batch_id) for f in frags if f.source_batch_id}
        for f in frags:
            returned.update(str(b) for b in (getattr(f, "origin_batch_ids", ()) or ()))
        covered = len(gold_set & returned)
        if args.only_misses and rank is not None:
            continue
        records.append({
            "question": q,
            "conversation": slug,
            "probe_type": p.get("probe_type", "untyped"),
            "gold_turns": gold_turns,
            "expected_answer": p.get("answer", ""),
            "gold_rank": rank,
            "gold_covered": covered,
            "gold_total": len(gold_set),
            "b2_declined": False,
            "gold_turn_text": "\n---\n".join(
                (gold_text.get(g) or "")[:1500] for g in gold_ids[:3]),
            "fragments": [{
                "position": i,
                "leg": f.source_type,
                "tokens": f.token_count,
                "score": round(float(f.score), 4),
                "is_gold": _hits(f),
                "text": (f.text or "")[:1200],
            } for i, f in enumerate(frags, 1)],
        })

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = OUT / f"{stamp}_{args.tag}.json"
    path.write_text(json.dumps({
        "utc": stamp, "sampled": len(records), "considered": seen,
        "only_misses": args.only_misses,
        "records": records,
    }, indent=1))

    legs = {}
    for r in records:
        for f in r["fragments"]:
            legs[f["leg"]] = legs.get(f["leg"], 0) + 1
    tot = sum(legs.values()) or 1
    print(f"wrote {path}  ({len(records)} probes)")
    print(f"  fragments by leg: "
          f"{ {k: f'{v} ({v/tot:.0%})' for k, v in sorted(legs.items(), key=lambda x: -x[1])} }")
    print(f"  gold returned: {sum(1 for r in records if r.get('gold_rank'))}/{len(records)}")
    print("\n  ⚑ THIS FILE IS FOR READING. It carries no score on purpose — the "
          "question it exists to answer ('would this context let a model "
          "answer?') is not one a metric on this probe set can reach.")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
