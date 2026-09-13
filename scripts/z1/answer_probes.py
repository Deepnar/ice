#!/usr/bin/env python3
"""Z2-mini: retrieve, assemble the REAL prompt, and let a model ANSWER.

**Why this exists.** Every Z1 number measures one half of the system — did the
gold turn come back. [Z2](../../docs/ROADMAP.md#z2)'s rule 0 is explicit that the
two failures are independent and only the pair is diagnostic: retrieval can
surface exactly the right memory and the model can still ignore it or contradict
it. Recall cannot see that, and neither can a coverage score.

It is also the only test that can rank two BACKGROUND models fairly. The
background model writes summaries, codex triplets and habit patterns; the
retrieval metrics see those only through a codex path with known defects
(relation sprawl, refused numeric entities, a closed property list), so
attributing a graph-shape difference to the model is unsafe. The answer does not
care which leg produced the context — it only cares whether the assembled
context was usable. Same probes, same answering model, one variable: which
background model built the store.

Writes question / gold turn / assembled context / answer, and **computes no
score** — same contract as harvest_probe_context.py. The reading is the
instrument.

Run:
  uv run python scripts/z1/answer_probes.py --n 30 --tag arm1
  uv run python scripts/z1/answer_probes.py --n 30 --tag arm2 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx  # noqa: E402
from sqlalchemy import func, text  # noqa: E402

from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402
from scripts.z1.run_meta import run_meta  # noqa: E402

PROBES = Path("experiments/curation_files/typed_probes.json")
OUT = Path("experiments/curation_files/probe_answers")


def stratified(probes, n, rng):
    """Even spread across probe_type — an unstratified sample is 55% episodic
    and would rank the arms on the one leg that is already known to dominate."""
    by_type = defaultdict(list)
    for p in probes:
        by_type[p.get("probe_type", "untyped")].append(p)
    for v in by_type.values():
        rng.shuffle(v)
    out, types = [], sorted(by_type)
    i = 0
    while len(out) < n and any(by_type[t] for t in types):
        t = types[i % len(types)]
        if by_type[t]:
            out.append(by_type[t].pop())
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="answers")
    ap.add_argument("--ablate", choices=["none", "fragments", "all"],
                    default="none",
                    help="none=full; fragments=drop codex fragments, "
                         "KEEP query expansion; all=disable the codex "
                         "leg entirely")
    ap.add_argument("--probes", default=None)
    ap.add_argument("--answer-model", default=None,
                    help="override the routed chat model (keep it FIXED across "
                         "arms — it is not the variable under test)")
    args = ap.parse_args()

    probe_path = Path(args.probes) if args.probes else PROBES
    probes = json.loads(probe_path.read_text())["probes"]
    rng = random.Random(args.seed)
    sample = stratified(probes, args.n, rng)
    print(f"probes: {len(sample)} of {len(probes)} from {probe_path}")

    db = SessionLocal()
    # Z1/G38: never let a measurement mutate the store it is measuring.
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
    from scripts.z1 import production_parity as pp
    from src.api.prompt_assembler import (_recent_turn_rows, assemble_prompt,
                                          conversation_summary_block)
    from src.classifier.classifier import PyTorchClassifier
    from src.memory.embedder import get_embedder
    from src.memory.models import EpisodicMemory
    from src.memory.tokens import estimate_from_chars
    from src.model_registry.registry import find_best_model, get_model_context_window
    from src.model_registry.runtime_probe import serving_window
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    from src.retrieval.configurable_orchestrator import ConfigurableOrchestrator

    embedder = get_embedder()
    clf = PyTorchClassifier(model_path=settings.classifier_model_path,
                            schema_path=settings.label_schema_path)
    # ⚑ Same two-condition split as `score_typed --ablate`, and for the same
    # reason: the codex graph reaches the prompt BOTH as fragments and as A4
    # query expansion (`orchestrator.py:558`). `fragments` withholds only the
    # fragments, so expansion still shapes the BM25 query; `all` disables the
    # leg, which kills expansion too because `_last_matched_entities` is never
    # populated. Retrieval-level stage 1 found the leg nets ~zero and expansion
    # contributes nothing — but `codex_multihop`'s retrieval metric is CIRCULAR
    # (its anchor half is defined as the anchor appearing in a codex fragment),
    # so only the ANSWER level can judge codex on its own ground.
    if args.ablate == "all":
        orch = ConfigurableOrchestrator(db, embedder, overrides={"codex": False})
    else:
        orch = HybridRetrievalOrchestrator(db, embedder)
    _drop_codex = (args.ablate == "fragments")

    meta = {}
    for slug, cid in conv_of.items():
        tc = db.query(EpisodicMemory).filter_by(conversation_id=cid).count()
        ch = db.query(func.coalesce(
            func.sum(func.length(EpisodicMemory.raw_text)), 0)
        ).filter_by(conversation_id=cid).scalar() or 0
        meta[slug] = (tc, estimate_from_chars(ch))

    # Warm-up: the first retrieval of a process builds the relation-gloss cache
    # and differs from steady state (measured 2026-08-13).
    first = sample[0]
    if conv_of.get(first["conversation"]):
        # G54: warm up through the SAME path, so the cache it primes is the one
        # the scored calls will use. It also stopped this block passing
        # `set_budget_from_turn_count(10, total_tokens=1000)` with no
        # total_budget, which silently fell back to context_total_budget_fallback.
        pp.retrieve(orch, pp.build(db, first["question"],
                                   conv_of[first["conversation"]], clf, embedder))

    records, _parity = [], []
    for idx, p in enumerate(sample, 1):
        slug = p["conversation"]
        conv_id = conv_of.get(slug)
        gold_turns = p.get("gold_turns") or []
        gold_ids = [g for g in (gold_index.get((slug, t)) for t in gold_turns) if g]
        if not conv_id or not gold_ids:
            continue
        gold_set = {str(g) for g in gold_ids}
        # the same turns, in the other id space
        gold_set |= {gold_batch[str(g)] for g in gold_ids if str(g) in gold_batch}
        q = p["question"]
        # G54: the shared reproduction of main.py, not a fourth local copy.
        # This block used to classify without the conversation, pass scope=None
        # and hardcode timescope_mode="current" — see production_parity's
        # module docstring for what each of those cost.
        pre = pp.build(db, q, conv_id, clf, embedder, stats=meta[slug])
        _parity.append(pre)
        c = pre.classification
        answer_model = args.answer_model or pre.model_name

        frags = []
        if pre.retrieve:
            frags = pp.retrieve(orch, pre)
            if _drop_codex:
                frags = [f for f in frags if f.source_type != "codex"]

        # The REAL assembler, not a hand-rolled concatenation — the shape of the
        # prompt is part of what is under test (Z2: "does the assembled context
        # read like something a model can use, or like concatenated fragments").
        #
        # ⚑ G56: and now with the arguments production actually passes. This
        # call omitted five of them, so the assembled prompt was not the one
        # main.py builds: `scope` (the cluster-name header and project slots
        # never rendered), `bookmarked_texts`, `session_start_text`,
        # `conversation_summary_text` (the C4 whole-conversation summary never
        # reached the answering model at all), and `max_recent_tokens` — which
        # silently defaulted to 4000 instead of the orchestrator's own
        # recent-turn budget.
        messages = assemble_prompt(
            memory_slots=[], retrieved_fragments=frags, user_message=q,
            db_session=db, conversation_id=conv_id, classification=c,
            scope=pre.scope,
            max_recent_tokens=getattr(orch, "recent_token_budget", None) or 4000,
            conversation_summary_text=conversation_summary_block(
                db, conv_id, pre.turn_count, pre.total_tokens,
                getattr(orch, "recent_token_budget", None) or 4000))

        answer, err = "", None
        try:
            r = httpx.post(f"{settings.ollama_base_url}/v1/chat/completions",
                           json={"model": answer_model, "messages": messages,
                                 "stream": False, "temperature": 0.0},
                           timeout=180.0)
            r.raise_for_status()
            answer = r.json()["choices"][0]["message"]["content"]
        except Exception as exc:                              # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:200]}"

        # G48: a fragment counts if the gold turn is its source OR one of the
        # turns it was derived from. Without the second clause only episodic
        # fragments can ever be credited — measured on one harvest, 474 codex,
        # 400 procedural and 207 timeline fragments earned ZERO gold credits
        # against 249 for episodic, so every recall number was an episodic score
        # wearing the system's name.
        def _frag_ids(f):
            """Every turn this fragment can be attributed to, in both id spaces."""
            out = set()
            if f.source_batch_id:
                out.add(str(f.source_batch_id))
            out.update(str(b) for b in (getattr(f, "origin_batch_ids", ()) or ()))
            return out

        def _hits(f):
            return bool(_frag_ids(f) & gold_set)
        rank = next((i for i, f in enumerate(frags, 1) if _hits(f)), None)

        # The turns the assembler will inject as "recent context", regardless of
        # retrieval. Taken from the assembler's own row selector so it cannot
        # drift; the budget trim inside get_recent_turns may drop a few, so this
        # slightly OVER-flags — the safe direction for a warning.
        recent_window_ids = set()
        try:
            for t in _recent_turn_rows(
                    db, conv_id,
                    getattr(settings, "recent_window_scope", "session"),
                    int(getattr(settings, "recent_window_bridge_turns", 1)),
                    int(getattr(settings, "recent_window_max_turns", 40))):
                recent_window_ids.add(str(getattr(t, "id", "")))
                recent_window_ids.add(str(getattr(t, "batch_id", "")))
        except Exception:                                     # noqa: BLE001
            pass
        records.append({
            "probe_type": p.get("probe_type", "untyped"),
            "question": q,
            "conversation": slug,
            "gold_turns": gold_turns,
            "gold_rank": rank,
            # ⚑ G56: `gold_covered` used only `source_batch_id` while `_hits`
            # above credits BOTH id spaces — two crediting rules on the same
            # record, disagreeing by exactly the non-episodic legs. Same rule
            # now.
            "gold_covered": len(gold_set & {i for f in frags
                                            for i in _frag_ids(f)}),
            # ⚑ G56: is this probe answerable WITHOUT retrieval? The assembler
            # always injects the conversation's recent turns, so a probe whose
            # gold sits inside that window can be answered from the window
            # alone and retrieval gets the credit. Measured across the 444-probe
            # set: 53 (11.9%) — 37 episodic_lookup, 10 summary_synthesis, 6
            # procedural. `gold_rank` cannot see this, because it is computed
            # over `frags` only. Recorded per probe so the judge's verdicts can
            # be split on it rather than averaged over it.
            "gold_in_recent_window": bool(recent_window_ids & gold_set),
            "gold_total": len(gold_set),
            "expected_answer": p.get("answer", ""),
            "gold_turn_text": "\n---\n".join(
                (gold_text.get(g) or "")[:1200] for g in gold_ids[:3]),
            "answer_model": answer_model,
            "b2_declined": not pre.retrieve,
            "fragment_count": len(frags),
            "legs": sorted({f.source_type for f in frags}),
            # ⚑ Per-fragment identity, not just a list of leg NAMES. The old
            # `legs` field could not support re-scoring at all, so a crediting
            # change meant re-running retrieval on the GPU (G48b).
            "fragments": [{
                "position": i, "leg": f.leg or f.source_type,
                "tokens": f.token_count,
                "source_batch_id": str(f.source_batch_id) if f.source_batch_id else None,
                "origin_batch_ids": [str(b) for b in (getattr(f, "origin_batch_ids", ()) or ())],
            } for i, f in enumerate(frags, 1)],
            "assembled_prompt": "\n\n".join(
                f"[{m['role']}]\n{m['content']}" for m in messages)[:12000],
            "answer": answer,
            "error": err,
        })
        print(f"  {idx}/{len(sample)}  {p.get('probe_type','?'):18s} "
              f"frags={len(frags):3d} rank={rank} "
              # ⚑ PRINT THE REASON, not just that it failed. Two probes
              # errored in the ablation's `fragments` condition with no way to
              # tell a timeout from a 400 from a refusal, which is the
              # difference between "drop these probes" and "the condition is
              # broken". Diagnostic only — no behavioural change.
              f"{('ERR ' + str(err)[:70]) if err else str(len(answer)) + ' chars'}")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = OUT / f"{stamp}_{args.tag}.json"
    path.write_text(json.dumps({
        "meta": run_meta(script=__file__, args=vars(args),
                         settings_keys=["codex_relation_canonical_threshold",
                                        "procedural_min_cited_turns",
                                        "retrieval_strengthen_writes",
                                        "decay_strengthen_amount"],
                         # ⚑ WHAT WAS ACTUALLY PASSED TO retrieve(). Every run
                         # recorded resolved settings and a git sha and NONE
                         # recorded this, which is precisely why a scope=None /
                         # no-conversation-id harness went nine days unnoticed.
                         extra={"parity": pp.provenance_fields(_parity[-1])} if _parity else None),
        "utc": stamp, "tag": args.tag, "seed": args.seed,
        "probe_file": str(probe_path), "records": records,
    }, indent=2))
    ok = sum(1 for r in records if not r["error"])
    print(f"\nwrote {path}  ({len(records)} probes, {ok} answered)")
    print("  ⚑ NO SCORE COMPUTED. Read both halves: what came back, and what "
          "the model did with it.")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
