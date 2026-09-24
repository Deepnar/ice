#!/usr/bin/env python3
"""Z1: score retrieval against the generated probe set. The tuning instrument.

**What it measures.** For each probe — a question written FROM a known turn —
run the real orchestrator against the seeded store and ask: did that turn come
back, and where in the ranking? Recall@k and MRR, plus which legs found it.

**⚑ IT RUNS THE PRODUCTION PREAMBLE, and G46 is why (fixed 2026-08-12).** This
instrument used to call `retrieve()` directly with the raw classification. That
is not the path `main.py` takes, and three of its numbers were partly about the
harness: the orchestrator kept its `__init__` default of 5,000 tokens instead of
the 8,100-11,350 a real conversation derives, and the classifier's raw
`Zero_Shot` reached a defensive guard production never hits because `main.py`
sets `context_reliance = "Long_Term_Memory"` first. Correcting only those two
moved recall@10 from **0.250 to 0.55** on the same store and the same probes,
and took zero-fragment probes from 15 to 0. Nothing about ICE changed.

So the preamble below is not optional scaffolding — it IS the measurement:
route the model, derive the budget from its real window, ask B2 whether memory
is consulted at all, and only then retrieve. **A probe B2 declines is reported
separately; it is a decision, not a miss.**

**⚠ What this still cannot tell you.** Recall is a PRESENCE metric — it answers
"did the gold turn come back", never "was the retrieved context any good". And
the generator's ambiguity guard scored candidates on question + answer while
retrieval only sees the question, so **74% of these probes have another turn
matching the question at least as well** (G46): a "miss" may be ICE returning an
equally good turn. Both caps are on the probe set, not on retrieval.

**v3 repair boundary:** this scorer measures retrieval evidence presence, not
answer quality or semantic support. The local reranker is a model call. Existing
probe labels and final experiment design still require the end-stage audit.

`--freeze-writes` (default ON) disables episodic access/decay strengthening and
cold restoration with `retrieval_strengthen_writes=False`. Graph retrieval no
longer strengthens or promotes edges. The no-cold-row startup check is retained
as an explicit corpus assumption, although the write gate now covers restoration.

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
SEED_MARKER = "z1seed"

# Every knob this instrument exists to turn. Dumped verbatim with each run so a
# score is never separated from the configuration that produced it.
_TRACKED = (
    "retrieval_leg_base_weights", "retrieval_leg_profiles",
    "retrieval_leg_topic_overrides", "retrieval_rrf_k",
    "retrieval_max_per_conversation", "retrieval_cluster_top_k",
    "retrieval_bm25_candidate_limit", "retrieval_vector_candidate_limit",
    "retrieval_chunk_candidate_limit", "codex_max_fanout",
    "codex_relation_fit_weight", "codex_relation_pool_multiplier",
    "codex_entity_edge_limit", "codex_max_depth",
    # ⚠ NOT `max_retrieval_tokens` — that is an orchestrator instance attribute
    # (default 5000, overwritten per request from the computed budget at
    # orchestrator.py:402), not a setting. Tracking it recorded a null under a
    # name that does not exist. These are the knobs that actually set the window.
    "context_budget_min", "context_budget_max", "context_budget_fallback",
    "context_budget_floor",
    "decay_strengthen_amount",
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

    if not PROBES.exists():
        print(f"need {PROBES} — generate probes and seed first")
        return 1
    probes = json.loads(PROBES.read_text())["probes"]

    db = SessionLocal()

    # G46: turn_number -> the episodic row id, and slug -> conversation_id,
    # DERIVED FROM THE STORE via idempotency_key. Both used to come out of a
    # file, and the arm runs overwrote that file — the map then pointed at the
    # last arm's 20-turn conversations while the 293-turn store was loaded, and
    # "every leg returns 0" nearly shipped as *retrieval is completely broken*.
    # The store is the only thing that cannot disagree with itself.
    #
    # `source_batch_id` on a fragment holds the episodic PRIMARY KEY (see
    # _strengthen_retrieved, which does .get() on it), despite the name.
    gold_index, conv_of = {}, {}
    for row_id, conv_id, key in db.execute(text(
            "SELECT id, conversation_id, idempotency_key FROM episodic_memory "
            "WHERE idempotency_key LIKE :m"), {"m": f"{SEED_MARKER}-%"}):
        _, slug, turn = key.split("-", 2)
        conv_of[slug] = str(conv_id)
        gold_index[(slug, int(turn))] = str(row_id)
    if not conv_of:
        print(f"no '{SEED_MARKER}-' rows in the store — seed it first")
        return 1
    print(f"  map from store: {len(conv_of)} conversations, "
          f"{len(gold_index)} turns")
    cold = db.execute(text("SELECT count(*) FROM cold_storage")).scalar()
    if cold and args.freeze:
        print(f"⚠ {cold} cold_storage rows present — resurrection CAN fire and "
              f"--freeze-writes does not gate it. Scores will drift across probes.")

    if args.freeze:
        settings.decay_strengthen_amount = 0.0
        # The amount alone does not stop writes. Without this gate the
        # access_count increment still fires on every retrieval, and that alone
        # made two identical runs disagree on 14 of 40 probes — the freeze
        # looked like it held because decay_score stopped moving.
        settings.retrieval_strengthen_writes = False

    from sqlalchemy import func

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

    # Per-conversation constants for the budget, queried once.
    conv_meta = {}
    for slug, conv_id in conv_of.items():
        turns = db.query(EpisodicMemory).filter_by(conversation_id=conv_id).count()
        chars = db.query(func.coalesce(
            func.sum(func.length(EpisodicMemory.raw_text)), 0)
        ).filter_by(conversation_id=conv_id).scalar() or 0
        conv_meta[slug] = (turns, estimate_from_chars(chars))

    # G46: record EVERY leg that produced the gold fragment, not the first one
    # RRF happened to stamp. `_apply_rrf` keeps first-leg-wins and bm25 is first
    # in the legs dict, so a fragment both legs found reads as bm25 — which is
    # how *"the vector leg is dead"* got written down about a leg that returns
    # 87-89 candidates on its own. Measured here in the harness by wrapping the
    # leg methods; the production attribution is untouched.
    gold_legs: set[str] = set()
    _watch = {}

    def _wrap_leg(name, fn):
        def inner(*a, **kw):
            out = fn(*a, **kw)
            try:
                want = _watch.get("gold")
                if want and any(f.source_batch_id and str(f.source_batch_id) == want
                                for f in out):
                    gold_legs.add(name)
            except Exception:
                pass
            return out
        return inner

    for _leg in ("_bm25_episodic", "_vector_episodic", "_codex_graph",
                 "_procedural_lookup", "_batch_summary_lookup", "_cold_lookup"):
        if hasattr(orch, _leg):
            setattr(orch, _leg, _wrap_leg(_leg.strip("_").split("_")[0],
                                          getattr(orch, _leg)))

    todo = probes[:args.limit] if args.limit else probes
    ranks, per_leg, misses, stats = [], Counter(), [], Counter()
    # Recorded because a probe set that collapses into one intent would exercise
    # ONE row of a 13-row leg-weight table while looking like full coverage.
    intents, topics, frag_counts = Counter(), Counter(), []
    budgets, found_legs, declined = [], Counter(), []

    for i, p in enumerate(todo, 1):
        cid = p["conversation"]
        gold_id = gold_index.get((cid, p["gold_turn"]))
        if gold_id is None:
            stats["gold_turn_not_seeded"] += 1
            continue
        q = p["question"]
        turn_count, total_tokens = conv_meta[cid]
        try:
            # ── the production preamble main.py runs before retrieve() ──
            # ⚑ G54: this was the FOURTH local copy of it, and like the other
            # three it classified without the conversation, passed scope=None
            # and hardcoded timescope_mode="current". `production_parity` is now
            # the single reproduction of main.py; tests/test_harness_parity.py
            # fails if it drifts. The original note here is still the reason it
            # exists at all: without a preamble this instrument measured a
            # system nobody ships — the orchestrator sat at its __init__ default
            # of 5,000 tokens against real budgets of 8,100-11,350, and a raw
            # `Zero_Shot` reached a guard production never touches, returning 0
            # fragments for 15 probes and scoring them as retrieval misses.
            pre = pp.build(db, q, conv_of[cid], clf, embedder,
                           stats=(turn_count, total_tokens))
            c = pre.classification

            # B2 decides whether memory is consulted at all. A turn it declines
            # is NOT a retrieval failure and averaging the two together is how
            # a deliberate decision got counted as a miss.
            if not pre.retrieve:
                declined.append({"conversation": cid, "gold_turn": p["gold_turn"],
                                 "question": q})
                continue

            gold_legs.clear()
            _watch["gold"] = str(gold_id)
            frags = pp.retrieve(orch, pre)
        except Exception as exc:
            print(f"  ! probe {i}: {type(exc).__name__}: {exc}")
            stats["retrieve_failed"] += 1
            continue

        intents.update(c.intent_tags or ["<none>"])
        topics.update(c.topic_tags or ["<none>"])
        frag_counts.append(len(frags))
        budgets.append(orch.max_retrieval_tokens)

        # ⚑ G48b/TRAPS #32: credit BOTH provenance spaces, not just the one
        # only episodic fragments ever populate.
        #
        # This matched `source_batch_id` alone, and `source_batch_id` is set in
        # exactly three places in the orchestrator, all of them building
        # `episodic` fragments. Codex, procedural, timeline and batch-summary
        # fragments could therefore never register a hit — not rarely, never —
        # so every recall and MRR number this scorer has produced is an
        # EPISODIC-LEG score reported under the whole system's name. Measured on
        # one harvest: 4,124 episodic fragments earned 249 gold credits while
        # 474 codex, 400 procedural and 207 timeline fragments earned zero.
        #
        # score_typed, answer_probes and harvest_probe_context all received this
        # fix on 2026-08-16. This file did not, and nothing noticed because its
        # numbers still looked like recall.
        def _frag_ids(f):
            out = set()
            if f.source_batch_id:
                out.add(str(f.source_batch_id))
            out.update(str(b) for b in (getattr(f, "origin_batch_ids", ()) or ()))
            return out

        rank = None
        for pos, f in enumerate(frags, 1):
            if str(gold_id) in _frag_ids(f):
                rank = pos
                per_leg[getattr(f, "leg", None) or f.source_type] += 1
                break
        ranks.append(rank)
        if rank is not None:
            found_legs["+".join(sorted(gold_legs)) or "<none>"] += 1
        if rank is None:
            misses.append({"conversation": cid, "gold_turn": p["gold_turn"],
                           "question": q, "returned": len(frags),
                           "produced_by_legs": sorted(gold_legs)})
        if i % 25 == 0:
            hit = sum(1 for r in ranks if r)
            print(f"    {i}/{len(todo)}  hit {hit}/{len(ranks)}  "
                  f"declined {len(declined)}")

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
    print(f"  by leg (first-leg-wins, as RRF stamps it): "
          f"{dict(per_leg.most_common())}")
    print(f"  by ALL producing legs: {dict(found_legs.most_common(8))}")
    print(f"\n  B2 declined to retrieve for {len(declined)} probes — production "
          f"never calls retrieve() for those, so they are a memory-DECISION "
          f"result, not a retrieval failure. They are excluded from the {n} above.")
    # The old warning here claimed recall@k was capped by
    # retrieval_max_per_conversation. It is not, for these probes:
    # `_session_diversify` exempts the CURRENT conversation from the cap
    # entirely, and every probe asks about its own conversation. Measured
    # 2026-08-12: RRF fuses ~157-196, diversify keeps ~59-115, and what
    # actually bounds the returned set is the token budget below.
    if budgets:
        print(f"  retrieval token budget: median "
              f"{statistics.median(budgets):.0f} "
              f"(min {min(budgets)}, max {max(budgets)}) — derived per "
              f"conversation from turn count; NOT the 5,000 __init__ default.")
    if frag_counts:
        z = sum(1 for x in frag_counts if x == 0)
        print(f"  fragments returned: mean {statistics.mean(frag_counts):.1f}, "
              f"median {statistics.median(frag_counts):.0f}, "
              f"ZERO for {z}/{len(frag_counts)} probes")
    print(f"\n  intents:  {dict(intents.most_common(6))}")
    print(f"  topics:   {dict(topics.most_common(6))}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS / f"{stamp}_{args.tag}.json"
    out.write_text(json.dumps({
        "tag": args.tag, "utc": stamp, "n": n,
        "recall_at_1": rec1, f"recall_at_{args.k}": reck, "mrr": mrr,
        "found": len(hits), "by_leg": dict(per_leg),
        "by_all_producing_legs": dict(found_legs),
        "intents": dict(intents), "topics": dict(topics),
        "fragments_returned": frag_counts, "budgets": budgets,
        "declined": len(declined), "declined_probes": declined,
        "freeze_writes": args.freeze, "stats": dict(stats),
        "ranks": ranks, "misses": misses[:40],
        "settings": resolved_settings(),
    }, indent=2, default=str))
    print(f"\nwrote {out}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
