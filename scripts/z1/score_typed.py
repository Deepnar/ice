#!/usr/bin/env python3
"""Z1: score TYPED probes — a different metric per probe class.

**Why a second scorer.** `score_retrieval.py` computes recall@k against a single
gold turn. That is the right metric for exactly one of the five probe classes.
Applying it to the rest is not merely imprecise, it is backwards: a
`summary_synthesis` probe has no single gold turn, so recall@k scores it as a
failure whenever retrieval does the right thing and returns a summary.

The consequence of having only that metric was measured (2026-08-13): of 377
hits on the 592-probe set, **`bm25+vector` produced 376 and `vector` 1 — codex,
procedural, batch-summary and timeline scored zero, never**. Those legs still
spend the token budget, so on that metric they are pure cost and a leg-weight
sweep would drive them to zero and call it an improvement (G48).

**Per class, what counts as success:**

| type | metric | rationale |
|---|---|---|
| `episodic_lookup`   | recall@k on the gold turn | one fact, one turn |
| `codex_multihop`    | **entity coverage** — did the anchor entity and the required turns' content reach the prompt, by any leg | no single gold turn exists |
| `procedural`        | **pattern presence** — did any procedural fragment come back at all | the answer is a stored habit, not a turn |
| `summary_synthesis` | **turn-set coverage** — what fraction of the required turn set is represented, directly or via a summary covering it | one turn cannot answer it |
| `temporal`          | recall@k **plus** the superseding turn not outranking it | returning only the current value is the failure mode |

Every number is reported PER TYPE and never averaged into one figure — a single
mean over five incommensurable metrics is a number about nothing.

Run:
  uv run python scripts/z1/score_typed.py --limit 40
  uv run python scripts/z1/score_typed.py --tag post-reseed
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import func, text  # noqa: E402

from scripts.z1 import production_parity as pp  # noqa: E402
from scripts.z1.run_meta import file_digest, run_meta  # noqa: E402
from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402

PROBES = Path("experiments/curation_files/typed_probes.json")
RESULTS = Path("experiments/curation_files/score_runs")
SEED_MARKER = "z1seed"

_TRACKED = (
    "retrieval_leg_base_weights", "retrieval_rrf_k",
    "retrieval_max_per_conversation", "retrieval_cluster_top_k",
    "codex_max_fanout", "codex_max_depth", "codex_relation_open_vocabulary",
    "codex_relation_canonical_threshold", "codex_node_promotion",
    "procedural_min_session_turns", "procedural_similarity_threshold",
    "context_growth_cap_ladder", "retrieval_strengthen_writes",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probes", default=None,
                    help="alternate probe file (used to exercise every "
                         "scoring branch, which --limit alone cannot)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default="typed")
    # ⚑ `choices` from the class itself, not a list typed here. Without it an
    # unrecognised name set an override key nothing reads, so the run was
    # byte-identical to the full system and was written to score_runs/ labelled
    # as an ablation. The two failure modes it hides are indistinguishable in
    # the output: "this leg is worth nothing" and "I misspelled the leg".
    from src.retrieval.configurable_orchestrator import SUPPORTED_FLAGS
    ap.add_argument("--ablate-leg", default=None,
                    choices=sorted(SUPPORTED_FLAGS),
                    help="disable ONE named leg/feature via "
                         "ConfigurableOrchestrator. Choices come from the "
                         "class's own SUPPORTED_FLAGS, so a name it does not "
                         "implement is rejected rather than silently ignored.")
    ap.add_argument("--ablate", choices=["none", "fragments", "all"],
                    default="none",
                    help="none=full system; fragments=drop codex "
                         "fragments but KEEP query expansion; "
                         "all=disable the codex leg entirely "
                         "(expansion dies with it)")
    args = ap.parse_args()

    probe_path = Path(args.probes) if args.probes else PROBES
    if not probe_path.exists():
        print(f"need {probe_path} — run generate_typed_probes.py first")
        return 1
    payload = json.loads(probe_path.read_text())
    probes = payload["probes"]
    if args.limit:
        probes = probes[:args.limit]

    db = SessionLocal()
    settings.codex_reinforce_increment = 0.0
    settings.decay_strengthen_amount = 0.0
    settings.retrieval_strengthen_writes = False

    # ⚑ TWO ID SPACES (G48b). A fragment's source_batch_id is the episodic ROW
    # id, but codex_edges.source_batch and procedural_memory.source_batch_ids
    # hold the turn's BATCH id — 9,662 edges join on batch_id and 0 on row id.
    # Without carrying both, codex/procedural/timeline fragments can never be
    # credited and this scorer measures the episodic leg alone.
    conv_of, gold_index, turn_text_by_id, gold_batch = {}, {}, {}, {}
    for rid, bid, cid, key, raw in db.execute(text(
            "select id, batch_id, conversation_id, idempotency_key, raw_text "
            "from episodic_memory where idempotency_key like :m"),
            {"m": f"{SEED_MARKER}-%"}):
        _, slug, turn = key.split("-", 2)
        conv_of[slug] = str(cid)
        gold_index[(slug, int(turn))] = str(rid)
        gold_batch[str(rid)] = str(bid)
        turn_text_by_id[str(rid)] = raw or ""

    def _frag_ids(f):
        """Every turn this fragment can be attributed to, in both id spaces."""
        out = set()
        if f.source_batch_id:
            out.add(str(f.source_batch_id))
        out.update(str(b) for b in (getattr(f, "origin_batch_ids", ()) or ()))
        return out

    def _ids_for(gid):
        return {str(gid)} | ({gold_batch[str(gid)]} if str(gid) in gold_batch else set())
    if not conv_of:
        print("store not seeded — nothing to score against")
        return 1

    # ⚑ COVERAGE METRICS INFLATE ON A SMALL STORE, and silently.
    # `codex_multihop` and `summary_synthesis` score how much of a required turn
    # set came back. If the store holds only those turns, everything comes back
    # and both score 1.000 — measured on a 2-turn fragment during branch
    # testing, where a system doing nothing useful looked perfect. The number is
    # not wrong so much as unearned, which is worse: it reads as a result.
    store_turns = sum(1 for _ in gold_index)
    expected = max((max(p.get("gold_turns") or [0]) for p in probes), default=0)
    if store_turns < expected or store_turns < 50:
        print(f"\n  ⚠⚠ STORE HAS {store_turns} TURNS; the probe set references "
              f"turns up to {expected}.")
        print("  Coverage-based scores (codex_multihop, summary_synthesis) are "
              "TRIVIALLY SATISFIED at this size and must not be reported as "
              "results. Re-seed before believing anything below.\n")

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
    from src.retrieval.configurable_orchestrator import ConfigurableOrchestrator

    embedder = get_embedder()
    clf = PyTorchClassifier(model_path=settings.classifier_model_path,
                            schema_path=settings.label_schema_path)
    # ⚑ THE CODEX ABLATION HAS TWO DISTINCT CONDITIONS, and collapsing them
    # would answer neither question. The codex graph reaches the prompt by TWO
    # paths: as fragments, and as A4 grounded query expansion, which appends
    # matched entity names to the BM25 search prompt (`orchestrator.py:558`).
    #   `none`      - full system.
    #   `fragments` - codex fragments dropped AFTER retrieval, expansion KEPT.
    #                 Isolates what the codex leg contributes to the prompt.
    #   `all`       - `ConfigurableOrchestrator(codex=False)`, which skips
    #                 `_codex_graph` entirely, so `_last_matched_entities` is
    #                 never populated and expansion dies with it.
    # The GAP between `fragments` and `all` IS the expansion contribution — the
    # leading explanation for arm A's episodic win, and currently untested.
    if args.ablate == "all":
        orch = ConfigurableOrchestrator(db, embedder, overrides={"codex": False})
    elif args.ablate_leg:
        # ⚑ THE GENERAL QUESTION, not just codex: what is EACH leg worth?
        # `procedural` scores a constant 1.000 because its metric asks only
        # whether any procedural fragment came back (TRAPS #37), so its
        # contribution has never been measured at all; `batch_summary` sits at
        # 0.303 and IMPROVED when codex was removed. Turning one leg off at a
        # time against the same 444 probes is the only thing that separates a
        # leg that works from a leg nobody has checked.
        orch = ConfigurableOrchestrator(
            db, embedder, overrides={args.ablate_leg: False})
    else:
        orch = HybridRetrievalOrchestrator(db, embedder)
    _drop_codex_fragments = (args.ablate == "fragments")

    meta_conv = {}
    for slug, cid in conv_of.items():
        tc = db.query(EpisodicMemory).filter_by(conversation_id=cid).count()
        ch = db.query(func.coalesce(
            func.sum(func.length(EpisodicMemory.raw_text)), 0)
        ).filter_by(conversation_id=cid).scalar() or 0
        meta_conv[slug] = (tc, estimate_from_chars(ch))

    _parity = []

    def retrieve_for(question, slug):
        """The production preamble, then retrieval.

        ⚑ G54: this used to be a LOCAL copy of the preamble, one of four, and
        all four had drifted the same way — `classify(question[:2000])` with no
        conversation id, `scope=None`, and `timescope_mode="current"` hardcoded.
        The classification gap alone changed the RRF blend weights on 65% of
        probes. The copy is gone; `production_parity` is the one reproduction of
        `main.py` and `tests/test_harness_parity.py` fails if it drifts again.
        """
        pre = pp.build(db, question, conv_of[slug], clf, embedder,
                       stats=meta_conv[slug])
        _parity.append(pre)
        if not pre.retrieve:
            return None, pre.classification
        _frags = pp.retrieve(orch, pre)
        if _drop_codex_fragments:
            # Retrieval ran in full — expansion already shaped the BM25 query —
            # and only the codex FRAGMENTS are withheld. That is the isolation
            # the condition needs.
            _frags = [f for f in _frags if f.source_type != "codex"]
        return _frags, pre.classification

    # Warm-up (the first retrieval of a process differs — relation gloss cache).
    for p in probes:
        if p["conversation"] in conv_of:
            retrieve_for(p["question"], p["conversation"])
            break

    per_type = defaultdict(lambda: {"n": 0, "scores": [], "declined": 0,
                                    "legs": Counter(), "detail": []})
    for p in probes:
        slug = p["conversation"]
        if slug not in conv_of:
            continue
        ptype = p["probe_type"]
        frags, _c = retrieve_for(p["question"], slug)
        bucket = per_type[ptype]
        bucket["n"] += 1
        if frags is None:
            bucket["declined"] += 1
            continue
        for f in frags:
            bucket["legs"][f.source_type] += 1

        gold_ids = [gold_index.get((slug, tn)) for tn in (p.get("gold_turns") or [])]
        gold_ids = [g for g in gold_ids if g]
        returned_ids = [i for f in frags for i in _frag_ids(f)]
        blob = " ".join((f.text or "") for f in frags).lower()

        if ptype == "episodic_lookup":
            gid = gold_ids[0] if gold_ids else None
            want = _ids_for(gid) if gid else set()
            # ⚑ A SPAN IS NOT A TURN (2026-08-22, caught by an ablation).
            #
            # This asks "did THIS turn come back, and where in the ranking".
            # Stamping origin_batch_ids onto batch-summary fragments earlier the
            # same day made a summary creditable — correct for the coverage
            # metrics below, and WRONG here: one batch summary spans up to **33
            # turns** on this store, so returning it scored a hit for any gold
            # turn inside that range. 25% of all turns sit inside some summary's
            # span, and the inflation measured **0.634 -> 0.754** — a leg
            # ablation exposed it, because turning the summary leg OFF should
            # not improve a recall score and it "did".
            #
            # Rule: a fragment credits a turn when it IS that turn or was
            # derived FROM it specifically. A codex or procedural fragment
            # extracted from the gold turn still counts — that is G48b and it
            # stands. A range that merely CONTAINS the turn does not.
            specific = [f for f in frags
                        if f.source_type not in ("batch_summary", "summary")]
            rank = next((i for i, f in enumerate(specific, 1)
                         if _frag_ids(f) & want), None)
            score = 1.0 if (rank and rank <= args.k) else 0.0
            # Kept so the span-credit effect stays visible instead of silently
            # disappearing: what the old, over-generous rule would have scored.
            span_rank = next((i for i, f in enumerate(frags, 1)
                              if _frag_ids(f) & want), None)
            bucket["detail"].append({
                "rank": rank,
                "rank_incl_span_credit": span_rank,
                "span_credited_only": bool(span_rank and not rank)})

        elif ptype == "codex_multihop":
            # No single gold turn, so success is: did the material needed to
            # join the hop reach the prompt?
            #
            # ⚑ THE ANCHOR MUST ARRIVE VIA THE GRAPH, NOT ANY LEG. The first
            # version gave half credit whenever the anchor entity appeared
            # anywhere in the returned text — including inside an episodic
            # fragment bm25 happened to return. That scores the knowledge graph
            # as working while it contributes nothing, which is precisely the
            # blindness G48 exists to remove, reintroduced one level up. Anchor
            # credit now requires a CODEX fragment carrying it.
            anchor = (p.get("anchor_entity") or "").strip().lower()
            codex_blob = " ".join((f.text or "") for f in frags
                                  if f.source_type in ("codex", "timeline")).lower()
            anchor_via_graph = bool(anchor and anchor in codex_blob)
            anchor_anywhere = bool(anchor and anchor in blob)
            covered = sum(1 for g in gold_ids if g in returned_ids)
            frac = covered / len(gold_ids) if gold_ids else 0.0
            # ⚑ A MISSING FIELD IS NOT A GRAPH FAILURE (2026-08-17).
            # The TRAPS #35 salvage recovered `question` and `gold_turns` but not
            # `anchor_entity`, which only the generator ever wrote — so 29 of 41
            # codex probes had no anchor, scored the anchor half False BY
            # CONSTRUCTION, and capped at 0.5. The mean then read as the graph
            # regressing 0.538 → 0.317 when on the 12 scoreable probes it had
            # IMPROVED, 68.5% → 75%. An unanchored probe is now scored on turn
            # coverage alone and counted separately; the two populations are
            # never averaged into one figure.
            if anchor:
                score = (0.5 if anchor_via_graph else 0.0) + 0.5 * frac
            else:
                score = frac
            bucket["detail"].append({"anchored": bool(anchor),
                                     "anchor_via_graph": anchor_via_graph,
                                     # kept for contrast: a large gap between
                                     # these two is the graph being carried by
                                     # the episodic legs.
                                     "anchor_anywhere": anchor_anywhere,
                                     "turn_coverage": round(frac, 3),
                                     "codex_frags": sum(
                                         1 for f in frags if f.source_type == "codex")})

        elif ptype == "procedural":
            # ⚑ G55. `1.0 if n_proc else 0.0` is a presence test, and the
            # procedural leg returns its LIMIT on every query: 200 fragments
            # over 40 probes, 1,160 over 232, 140 over 28 — exactly 5 each,
            # every time, in every recorded run. So the score was 1.000 by
            # construction and carried no information about whether the right
            # habit came back. TRAPS #37 called this out when the candidate
            # pool was one row; the pool is now 30 (G49 shipped a second
            # activation path) and the metric is vacuous for a different
            # reason — the leg is not selective, so presence is guaranteed.
            #
            # Scored on gold provenance instead: a procedural pattern carries
            # `origin_batch_ids` for the turns it was extracted from, so a
            # pattern built from this probe's gold turns is a hit and one built
            # from elsewhere is not. `presence` stays in the detail as a
            # diagnostic — it is what the old score was — so the two runs
            # remain comparable and the change is visible rather than silent.
            n_proc = sum(1 for f in frags if f.source_type == "procedural")
            hit = any(g in _frag_ids(f) for g in gold_ids
                      for f in frags if f.source_type == "procedural")
            score = 1.0 if hit else 0.0
            bucket["detail"].append({"procedural_frags": n_proc,
                                     "presence_only": 1.0 if n_proc else 0.0,
                                     "gold_grounded": bool(hit)})

        elif ptype == "summary_synthesis":
            # ⚑ G55. This used to be `max(frac, 1.0 if n_sum else 0.0)`: ANY
            # batch-summary fragment scored the probe a perfect 1.0, with no
            # check that the summary covered the gold at all. It was harmless
            # only by accident — the summary leg was switched off in every
            # recorded run (G53), so `n_sum` was always 0 and the clause never
            # fired. The moment G53 made the leg fire, it returned a fragment on
            # 12 of 12 probes, and this metric would have read 0.303 -> 1.000
            # and been reported as the fix working. That is TRAPS #21/#37: a
            # presence test wearing a coverage score's name.
            #
            # A summary covering the span IS the right answer here, and that
            # intent is preserved — but it is now EARNED rather than assumed.
            # `origin_batch_ids` on a batch_summary fragment carries the turns
            # it compresses (from `episodic_memory.batch_summary_id`), so
            # `_frag_ids` already folds them into `returned_ids` and a summary
            # that genuinely covers a gold turn scores that turn. One that does
            # not, scores nothing.
            direct = sum(1 for g in gold_ids if g in returned_ids)
            n_sum = sum(1 for f in frags
                        if f.source_type in ("batch_summary", "summary"))
            # How much of the gold this probe's summaries actually reach, kept
            # separately so "the leg fired" and "the leg helped" stay distinct.
            via_summary = sum(
                1 for g in gold_ids
                if any(g in _frag_ids(f) for f in frags
                       if f.source_type in ("batch_summary", "summary")))
            score = direct / len(gold_ids) if gold_ids else 0.0
            bucket["detail"].append({"turn_coverage": round(score, 3),
                                     "summary_frags": n_sum,
                                     "gold_via_summary": via_summary})

        elif ptype == "temporal":
            # Same rule as episodic_lookup: this is a "did THIS turn come back,
            # and did the newer one outrank it" question, so a span that merely
            # contains either turn credits neither. Worse here than there — one
            # summary can span BOTH the old value and the turn that superseded
            # it, which would make the ordering test unanswerable while looking
            # like a pass.
            specific = [f for f in frags
                        if f.source_type not in ("batch_summary", "summary")]
            gid = gold_ids[0] if gold_ids else None
            rank = next((i for i, f in enumerate(specific, 1)
                         if _frag_ids(f) & _ids_for(gid)), None)
            sup_id = gold_index.get((slug, p.get("superseded_by")))
            sup_rank = next((i for i, f in enumerate(specific, 1)
                             if _frag_ids(f) & _ids_for(sup_id)),
                            None)
            hit = bool(rank and rank <= args.k)
            # Returning ONLY the current value is the failure this class exists
            # to catch, so the superseding turn must not outrank the old one.
            beaten = bool(hit and sup_rank and sup_rank < rank)
            score = 1.0 if (hit and not beaten) else 0.0
            bucket["detail"].append({"rank": rank, "superseding_rank": sup_rank,
                                     "outranked_by_newer": beaten})
        else:
            continue
        bucket["scores"].append(score)

    print(f"\n{'='*66}\nTYPED SCORE   (tag: {args.tag})\n{'='*66}")
    summary = {}
    for ptype in sorted(per_type):
        b = per_type[ptype]
        sc = b["scores"]
        mean = statistics.mean(sc) if sc else 0.0
        # ⚑ EVERY TYPE CARRIES ITS STANDARD ERROR. `temporal` runs at n=28,
        # where the arm-to-arm gap that was read as a result (0.500 vs 0.643)
        # is ~1.9 SE — indistinguishable from noise. A mean printed without its
        # spread invites exactly that reading, and got it.
        se = (round(statistics.pstdev(sc) / (len(sc) ** 0.5), 3)
              if len(sc) > 1 else None)
        summary[ptype] = {
            "n": b["n"], "scored": len(sc), "declined": b["declined"],
            "mean_score": round(mean, 3),
            "se": se,
            "ci95": (round(1.96 * se, 3) if se is not None else None),
            "legs_seen": dict(b["legs"].most_common()),
        }
        print(f"  {ptype:<20} n={b['n']:<4} scored={len(sc):<4} "
              f"score={mean:.3f}{f' ±{se}' if se is not None else ''}   "
              f"declined={b['declined']}")
        print(f"      legs returned: {dict(b['legs'].most_common(6))}")
        # Split the codex population by whether the metric could see the anchor
        # at all. One mean over both answers nothing (see the branch above).
        if ptype == "codex_multihop":
            anch = [d for d in b["detail"] if d.get("anchored")]
            unanch = [d for d in b["detail"] if not d.get("anchored")]
            summary[ptype]["anchored"] = len(anch)
            summary[ptype]["unanchored"] = len(unanch)
            # ⚑ THE TWO POPULATIONS GET THEIR OWN MEANS, because they are
            # scored by different formulas. An anchored probe is
            # 0.5*anchor + 0.5*coverage; an unanchored one is coverage alone.
            # The comment on the scoring branch says they "are never averaged
            # into one figure" — and `mean_score` above averaged them anyway,
            # which is how 0.531 hid anchored 0.603 and unanchored 0.345 on
            # arm B. Counting them was never the problem; reporting one number
            # over them was.
            def _mean(ds):
                vals = [(0.5 if d["anchor_via_graph"] else 0.0) + 0.5 * d["turn_coverage"]
                        if d.get("anchored") else d["turn_coverage"] for d in ds]
                return round(statistics.mean(vals), 3) if vals else None

            def _se(ds):
                vals = [(0.5 if d["anchor_via_graph"] else 0.0) + 0.5 * d["turn_coverage"]
                        if d.get("anchored") else d["turn_coverage"] for d in ds]
                if len(vals) < 2:
                    return None
                return round(statistics.pstdev(vals) / (len(vals) ** 0.5), 3)

            summary[ptype]["anchored_mean"] = _mean(anch)
            summary[ptype]["anchored_se"] = _se(anch)
            summary[ptype]["unanchored_mean"] = _mean(unanch)
            summary[ptype]["unanchored_se"] = _se(unanch)
            if anch:
                via = sum(1 for d in anch if d["anchor_via_graph"])
                anywhere = sum(1 for d in anch if d.get("anchor_anywhere"))
                summary[ptype]["anchor_via_graph"] = f"{via}/{len(anch)}"
                summary[ptype]["anchor_anywhere"] = f"{anywhere}/{len(anch)}"
                print(f"      anchored   n={len(anch):<4} score={_mean(anch)} "
                      f"±{_se(anch)}   anchor via GRAPH {via}/{len(anch)}, "
                      f"anywhere {anywhere}/{len(anch)}")
                if anywhere > via:
                    print(f"      ⚠ {anywhere - via} anchors arrived WITHOUT the graph — "
                          f"the episodic legs are carrying them")
            if unanch:
                print(f"      unanchored n={len(unanch):<4} score={_mean(unanch)} "
                      f"±{_se(unanch)}   (turn coverage alone)")
                print(f"      ⚠ {len(unanch)} probes carry NO anchor_entity — scored on "
                      f"turn coverage alone, NOT as an anchor failure")
    print("\n  ⚑ These are FIVE DIFFERENT METRICS. They are not averaged, and a "
          "single headline number over them would describe nothing.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS / f"{stamp}_{args.tag}.json"
    out.write_text(json.dumps({
        "meta": run_meta(script=__file__, args=vars(args),
                         settings_keys=list(_TRACKED),
                         inputs=[file_digest(probe_path)],
                         extra={"probe_set_meta": payload.get("meta", {}),
                                "k": args.k,
                                # ⚑ What was actually passed to retrieve() —
                                # the field set whose absence hid G52/G54.
                                "parity": pp.provenance_fields(_parity[-1])
                                if _parity else None}),
        "per_type": summary,
        "detail": {t: per_type[t]["detail"] for t in per_type},
    }, indent=1, default=str))
    print(f"\nwrote {out}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
