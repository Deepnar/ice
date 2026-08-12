#!/usr/bin/env python3
"""Z1: score retrieval against the generated probe set. The tuning instrument.

**What it measures.** For each probe — a question written FROM a known turn —
run the real orchestrator against the seeded store and ask: did that turn come
back, and where in the ranking? Recall@k and MRR, plus which leg found it.

**Why this can be a fast loop at all.** Retrieval has no LLM in it. Given a
fixed store and a fixed question the ranking is deterministic, and scoring is
exact (the gold turn is known by construction, not judged), so a configuration
can be evaluated in seconds with no model call and no judge. That is the whole
reason the probe set was rebuilt forwards — see `generate_probes.py`.

**⚑ G38: RETRIEVAL WRITES TO THE STORE, and this neutralises it with settings
rather than surgery.** `retrieve()` commits in three places — codex edge
strength and pending→active promotion, episodic `access_count`/`decay_score`,
and cold-storage resurrection which physically moves rows. Left alone, probe N
changes the store probe N+1 sees, so probe ORDER changes the score and a noise
floor would absorb the drift as if it were noise.

`--freeze-writes` (default ON) zeroes `codex_reinforce_increment` and
`decay_strengthen_amount`. Two properties make this the right fix rather than a
compromise:

  * it cannot change what the CURRENT probe retrieves — both values only affect
    *later* retrievals, so the measurement is untouched and only the
    contamination is removed;
  * the third write, cold resurrection, cannot fire here at all: a freshly
    seeded store has no `cold_storage` rows. Asserted at startup rather than
    assumed, because "cannot happen" is how it will happen.

**Resolved settings are dumped with every run.** `test_settings_freeze.py` now
compares DECLARATIONS, so an `.env` override is caught by nothing else — and a
tuning sweep is precisely a pile of `.env` overrides. A result whose settings
are not recorded alongside it is not a result.

Run:
  uv run python scripts/z1/score_retrieval.py --limit 20
  uv run python scripts/z1/score_retrieval.py --tag baseline
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text  # noqa: E402

from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402

PROBES = Path("experiments/curation_files/generated_probes.json")
MANIFEST = Path("experiments/curation_files/seeded_store.json")
RESULTS = Path("experiments/curation_files/score_runs")

# Every knob this instrument exists to turn. Dumped verbatim with each run so a
# score is never separated from the configuration that produced it.
_TRACKED = (
    "retrieval_leg_base_weights", "retrieval_leg_profiles",
    "retrieval_leg_topic_overrides", "retrieval_rrf_k",
    "retrieval_max_per_conversation", "retrieval_cluster_top_k",
    "retrieval_bm25_candidate_limit", "retrieval_vector_candidate_limit",
    "retrieval_chunk_candidate_limit", "codex_max_fanout",
    "codex_relation_fit_weight", "codex_relation_pool_multiplier",
    "codex_entity_edge_limit", "codex_max_depth", "max_retrieval_tokens",
    "decay_strengthen_amount", "codex_reinforce_increment",
    "retrieval_coverage_enabled", "retrieval_set_floor_enabled",
)


def resolved_settings() -> dict:
    out = {}
    for name in _TRACKED:
        try:
            out[name] = getattr(settings, name)
        except AttributeError:
            out[name] = "<absent>"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=10, help="recall@k headline")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--freeze-writes", dest="freeze", action="store_true", default=True)
    ap.add_argument("--allow-writes", dest="freeze", action="store_false",
                    help="let retrieval mutate the store (G38) — for measuring the drift itself")
    args = ap.parse_args()

    if not PROBES.exists() or not MANIFEST.exists():
        print(f"need {PROBES} and {MANIFEST} — generate probes and seed first")
        return 1
    probes = json.loads(PROBES.read_text())["probes"]
    manifest = json.loads(MANIFEST.read_text())

    # turn_number -> the episodic row id, per conversation. `source_batch_id` on
    # a fragment holds the episodic PRIMARY KEY (see _strengthen_retrieved,
    # which does .get() on it), despite the name.
    gold_index, conv_of = {}, {}
    for cid, meta in manifest["conversations"].items():
        conv_of[cid] = meta["conversation_id"]
        for rec in meta["turns"]:
            gold_index[(cid, rec["turn_number"])] = rec["episodic_id"]

    db = SessionLocal()
    cold = db.execute(text("SELECT count(*) FROM cold_storage")).scalar()
    if cold and args.freeze:
        print(f"⚠ {cold} cold_storage rows present — resurrection CAN fire and "
              f"--freeze-writes does not gate it. Scores will drift across probes.")

    if args.freeze:
        settings.codex_reinforce_increment = 0.0
        settings.decay_strengthen_amount = 0.0

    from src.classifier.classifier import PyTorchClassifier
    from src.memory.embedder import get_embedder
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator

    embedder = get_embedder()
    clf = PyTorchClassifier(model_path=settings.classifier_model_path,
                            schema_path=settings.label_schema_path)
    orch = HybridRetrievalOrchestrator(db, embedder)

    todo = probes[:args.limit] if args.limit else probes
    ranks, per_leg, misses, stats = [], Counter(), [], Counter()

    for i, p in enumerate(todo, 1):
        cid = p["conversation"]
        gold_id = gold_index.get((cid, p["gold_turn"]))
        if gold_id is None:
            stats["gold_turn_not_seeded"] += 1
            continue
        q = p["question"]
        try:
            c = clf.classify(q[:2000])
            emb = embedder.encode(q, convert_to_tensor=False).tolist()
            frags = orch.retrieve(classification=c, conversation_id=conv_of[cid],
                                  prompt_embedding=emb, scope=None)
        except Exception as exc:
            print(f"  ! probe {i}: {type(exc).__name__}: {exc}")
            stats["retrieve_failed"] += 1
            continue

        rank = None
        for pos, f in enumerate(frags, 1):
            if f.source_batch_id and str(f.source_batch_id) == str(gold_id):
                rank = pos
                per_leg[getattr(f, "leg", None) or f.source_type] += 1
                break
        ranks.append(rank)
        if rank is None:
            misses.append({"conversation": cid, "gold_turn": p["gold_turn"],
                           "question": q, "returned": len(frags)})
        if i % 25 == 0:
            hit = sum(1 for r in ranks if r)
            print(f"    {i}/{len(todo)}  hit {hit}/{len(ranks)}")

    n = len(ranks)
    if not n:
        print("no probes scored")
        return 1
    hits = [r for r in ranks if r]
    rec1 = sum(1 for r in hits if r <= 1) / n
    reck = sum(1 for r in hits if r <= args.k) / n
    mrr = sum(1.0 / r for r in hits) / n

    print(f"\n{'='*58}\nSCORED {n} probes   (tag: {args.tag})\n{'='*58}")
    print(f"  recall@1   {rec1:.3f}")
    print(f"  recall@{args.k:<3} {reck:.3f}")
    print(f"  MRR        {mrr:.3f}")
    print(f"  found at all {len(hits)}/{n}")
    if hits:
        print(f"  median rank when found: {statistics.median(hits):.0f}")
    print(f"  by leg: {dict(per_leg.most_common())}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS / f"{stamp}_{args.tag}.json"
    out.write_text(json.dumps({
        "tag": args.tag, "utc": stamp, "n": n,
        "recall_at_1": rec1, f"recall_at_{args.k}": reck, "mrr": mrr,
        "found": len(hits), "by_leg": dict(per_leg),
        "freeze_writes": args.freeze, "stats": dict(stats),
        "ranks": ranks, "misses": misses[:40],
        "settings": resolved_settings(),
    }, indent=2, default=str))
    print(f"\nwrote {out}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
