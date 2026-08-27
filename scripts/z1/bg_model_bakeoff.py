#!/usr/bin/env python3
"""Z1: choose the GENERAL background model, across every job it actually does.

**Why this exists, and why it is not the comparison the roadmap describes.**
[ROADMAP.md:377](../../docs/ROADMAP.md) tells the next session to pick the
generation model by comparing `summary_coverage`. That is now a *defensible*
screen — `strip_generated_index` landed and coverage scores the summary PROSE,
not the model's own `Key terms:` index, so it is no longer the receipt
[TRAPS #45](../../docs/TRAPS.md) caught. But it is still a **presence** metric:
it answers *"are the must-terms in there"* and never *"is this summary any good,
did it invent anything"*. Presence metrics are the failure CLAUDE.md names
first. So coverage is used here as a cheap screen and nothing more.

**And summarising is one of NINE jobs.** `get_bg_model_name()` is read by the
turn summary, batch summaries, conversation summary, cluster naming, reflection,
the maintenance agent, procedural extraction, document-kind, the raw slicer and
the codex conflict reconciler. Picking a model on summaries alone picks it for
one job in nine. This runs every job that can be exercised without a seeded
store, and scores the ones that have an OBJECTIVE answer objectively.

**⚑ IT CALLS THE PRODUCTION FUNCTIONS.** Not copies of their prompts — the real
`generate_summary`, the real `_generate_cluster_name`, the real
`_extract_slice_turns`, the real `make_llm_decider` / `make_llm_reconciler`.
A harness that re-implements the prompt measures the harness. Models are
swapped by setting `settings.background_model_name`, which every one of those
functions resolves through `get_bg_model_name()` at call time — the same
mechanism `seed_store.py --bg-model` uses.

**⚑ THE GOLD IS BUILT FROM THE CORPUS, NOT HAND-WRITTEN**, wherever it can be.
The slicer test concatenates N real turns with their speaker labels stripped:
the true boundaries are known because we built the blob, so turn-count error,
role accuracy and invented-content are all exactly measurable. Hand-written
fixtures agree with whoever wrote them.

  uv run python scripts/z1/bg_model_bakeoff.py --list
  uv run python scripts/z1/bg_model_bakeoff.py --turns 30
  uv run python scripts/z1/bg_model_bakeoff.py --models qwen3:4b-instruct,granite4.2:3b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.config import settings  # noqa: E402

CORPUS = "data/simulation/simulation_full.jsonl"
OUT_DIR = "experiments/curation_files/bakeoff"

# Small models only — G58's standing decision: ICE will NEVER use a model that
# large for background work, because background runs on a laptop card beside
# whatever chat model the user is talking to. The 26B and granite4:small-h (19G)
# are therefore not candidates, however they might score.
DEFAULT_MODELS = [
    "granite4:micro",          # 2.1 G
    "granite4.2:3b",           # 2.2 G  — newer; IBM lists summarisation +
                               #          classification + native JSON as targets
    "qwen3:4b-instruct",       # 2.5 G  — the incumbent, and what every prior
                               #          measurement actually ran on
    "nemotron-mini:4b",        # 2.7 G
    "qwen3.5:4b",              # 3.4 G
    "granite4:tiny-h",         # 4.2 G
    "granite4.2:8b",           # 5.3 G
    "ministral-3:8b",          # 6.0 G
    "lfm2.5:8b",               # 6-ish  — edge model, built for tool calling
    "mistral-nemo:latest",     # 7.1 G
    "gemma4:e4b",              # 9.6 G
]

# ── the reconciler gold ─────────────────────────────────────────────────────
# ⚠ HAND-WRITTEN, n=9, and that is stated rather than hidden. It separates a
# model that understands the task from one that does not; it does NOT separate
# 78% from 89%. Same honesty as calibrate_judge.py's n=15 note. Built as three
# clean cases per label so a model cannot score well by always guessing one.
RECONCILE_GOLD = [
    # the new fact replaces the old one
    ({"subject": "i", "old_relation": "uses", "old_object": "postgres",
      "relation": "uses", "object": "sqlite",
      "turn": "I switched the prototype from Postgres to SQLite last week; "
              "Postgres is gone from the project now."}, "expire_old"),
    ({"subject": "the api", "old_relation": "runs_on", "old_object": "port 5432",
      "relation": "runs_on", "object": "port 5433",
      "turn": "I moved the API off 5432 because of the clash. It runs on 5433 now."},
     "expire_old"),
    ({"subject": "i", "old_relation": "lives_in", "old_object": "pune",
      "relation": "lives_in", "object": "mumbai",
      "turn": "I moved out of Pune in March. Been living in Mumbai since then."},
     "expire_old"),
    # both hold at once
    ({"subject": "i", "old_relation": "knows", "old_object": "python",
      "relation": "knows", "object": "rust",
      "turn": "I've written Python for years and I picked up Rust this spring. "
              "I use both depending on the job."}, "keep_both"),
    ({"subject": "the repo", "old_relation": "has", "old_object": "tests",
      "relation": "has", "object": "docs",
      "turn": "The repo has a test suite and a docs folder, both maintained."},
     "keep_both"),
    ({"subject": "she", "old_relation": "works_at", "old_object": "the lab",
      "relation": "teaches_at", "object": "the college",
      "turn": "She works at the lab on weekdays and teaches at the college on "
              "Saturdays."}, "keep_both"),
    # the new fact is not actually asserted / is wrong
    ({"subject": "i", "old_relation": "uses", "old_object": "vim",
      "relation": "hates", "object": "vim",
      "turn": "I use vim for everything. Someone in the thread said they hate it, "
              "which I thought was funny."}, "reject_new"),
    ({"subject": "the model", "old_relation": "has_size", "old_object": "4b",
      "relation": "has_size", "object": "70b",
      "turn": "The model is 4B. I was reading about a 70B one earlier but that is "
              "a different project entirely."}, "reject_new"),
    ({"subject": "we", "old_relation": "shipped", "old_object": "the feature",
      "relation": "cancelled", "object": "the feature",
      "turn": "We shipped the feature on Tuesday. The one we cancelled was the "
              "other proposal, not this."}, "reject_new"),
]

AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["merge", "expire", "keep", "unsure"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}


def load_turns(n: int, seed: int = 20260826) -> list[dict]:
    """N dense turns, sampled deterministically from the true corpus source.

    `simulation_full.jsonl` — not the curated EC checkpoints — because it is the
    only file carrying both sides of every turn (see
    experiments/curation_files/README.md).
    """
    rows = [json.loads(l) for l in open(CORPUS)]
    dense = [r for r in rows
             if len((r.get("prompt") or "") + (r.get("response") or "")) >= 400]
    rng = random.Random(seed)
    return rng.sample(dense, min(n, len(dense)))


# ── jobs ────────────────────────────────────────────────────────────────────

def job_summary(turns, embedder) -> dict:
    """The turn summary — the highest-stakes background job there is.

    Its coverage decides `inject_raw`, i.e. whether the summary REPLACES the raw
    turn in the prompt the answer is generated from. A bad summariser does not
    degrade a side feature, it rewrites the evidence.
    """
    from src.workers.post_flight import generate_summary
    from src.workers.turn_density import extract_key_terms, strip_generated_index

    cov, empty, no_prose, no_abstract, lens = [], 0, 0, 0, []
    for t in turns:
        kt = extract_key_terms((t.get("prompt") or "") + "\n" + (t.get("response") or ""),
                               embedder)
        summary, coverage, abstract = generate_summary(
            t.get("prompt") or "", t.get("response") or "", kt)
        if not (summary or "").strip():
            empty += 1
            continue
        prose = strip_generated_index(summary)
        # A summary that is ONLY the `Key terms:` index is the TRAPS #21 failure
        # (granite4:tiny-h scored 1.000 on 12 of 12 doing exactly this).
        if len(prose.strip()) < 40:
            no_prose += 1
        if not abstract:
            no_abstract += 1
        cov.append(coverage)
        lens.append(len(prose))
    return {"n": len(turns), "empty": empty, "no_prose": no_prose,
            "no_abstract": no_abstract,
            "coverage_mean": round(statistics.mean(cov), 4) if cov else 0.0,
            "coverage_median": round(statistics.median(cov), 4) if cov else 0.0,
            "prose_chars_median": int(statistics.median(lens)) if lens else 0}


def job_cluster_name(turns) -> dict:
    """Cluster naming, scored on the prompt's OWN rules.

    The prompt forbids turn numbers, chapter numbers and dates, and asks for
    3-5 words. Those are objective — no judge needed to see a model ignore them.
    """
    from src.workers.clustering import _generate_cluster_name

    groups = [turns[i:i + 5] for i in range(0, min(len(turns), 25), 5)]
    bad, has_digit, wrong_len, names = 0, 0, 0, []
    for g in groups:
        shim = [type("T", (), {"raw_text": f"User: {t.get('prompt','')}\n\n"
                                           f"Assistant: {t.get('response','')}"})()
                for g_ in [g] for t in g_]
        name = _generate_cluster_name(shim)
        names.append(name)
        if not name or name == "Unnamed Cluster":
            bad += 1
            continue
        if any(ch.isdigit() for ch in name):
            has_digit += 1
        if not (3 <= len(name.split()) <= 5):
            wrong_len += 1
    return {"n": len(groups), "failed": bad, "contains_digit": has_digit,
            "wrong_word_count": wrong_len, "samples": names[:3]}


def job_raw_slice(turns) -> dict:
    """⚑ THE ONE WITH FREE, REAL GOLD.

    Take 4 consecutive real turns, strip the speaker labels, concatenate. We
    BUILT the blob, so we know exactly how many turns are in it and which role
    each carries. Scored on: did it recover the right NUMBER of turns, did it
    get the roles right, and — the one that matters — did it INVENT text that
    is not in the source.
    """
    from src.ingestion.raw_slicer import _default_llm, _extract_slice_turns

    # ⚠ BLOBS ARE KEPT SMALL ON PURPOSE. The slicer has to ECHO the whole slice
    # back as JSON, so the output is longer than the input — and its budget is
    # max_tokens=1500. A 4-turn blob with full responses truncated qwen at 5,949
    # chars, `_parse_turns` fell back to one-turn-holding-everything, and the
    # naive scorer read that as role_accuracy 1.000. Sizing the blob so a
    # CORRECT answer fits is what makes this measure segmentation rather than
    # output budget. (That truncation is a real production finding — recorded
    # separately, not smuggled in as this model's score.)
    blocks, gold = [], []
    for i in range(0, min(len(turns), 24), 3):
        chunk = turns[i:i + 3]
        if len(chunk) < 3:
            break
        parts, roles = [], []
        for t in chunk:
            parts.append((t.get("prompt") or "").strip()[:300])
            roles.append("user")
            parts.append((t.get("response") or "").strip()[:400])
            roles.append("assistant")
        blocks.append("\n\n".join(parts))
        gold.append(roles)

    n_err, role_acc, invented = [], [], []
    failed = fell_back = 0
    for blob, roles in zip(blocks, gold):
        try:
            got = _extract_slice_turns(blob, _default_llm)
        except Exception:
            failed += 1
            continue
        if not got:
            failed += 1
            continue
        # ⚑ DETECT THE FALLBACK. `_parse_turns` never drops content: on any
        # parse failure it returns ONE user turn holding the entire slice. That
        # is a total failure of the job, and it must not be scored as a
        # perfectly-roled single turn.
        if len(got) == 1 and got[0]["role"] == "user" and \
                len(got[0]["text"]) > 0.8 * len(blob.strip()):
            fell_back += 1
            continue
        n_err.append(abs(len(got) - len(roles)))
        # Scored over the TRUE turn count, not over zip's shorter side — a model
        # that emits 2 of 8 turns must not score 1.000 on the 2 it managed.
        role_acc.append(sum(1 for g, r in zip(got, roles) if g.get("role") == r)
                        / len(roles))
        # invented content: how much of the emitted text is absent from the source
        src = "".join(blob.lower().split())
        miss = sum(1 for g in got
                   if "".join((g.get("text") or "").lower().split())[:60] not in src)
        invented.append(miss / len(got))
    return {"n": len(blocks), "failed": failed, "fell_back_to_one_turn": fell_back,
            "turn_count_err_mean": round(statistics.mean(n_err), 2) if n_err else None,
            "role_accuracy": round(statistics.mean(role_acc), 3) if role_acc else None,
            "invented_frac": round(statistics.mean(invented), 3) if invented else None}


def job_reconcile() -> dict:
    """The codex conflict reconciler: one word out of three, against gold.

    ⚠ This runs on the GENERAL model on purpose — it is the job inside
    codex_extractor.py that must NOT follow the extraction specialist, because
    it reasons rather than extracts.
    """
    from src.workers.codex_extractor import make_llm_reconciler

    rec = make_llm_reconciler()
    ok, got_counts = 0, Counter()
    for ctx, want in RECONCILE_GOLD:
        try:
            got = rec(ctx)
        except Exception:
            got = "ERROR"
        got_counts[got] += 1
        ok += (got == want)
    return {"n": len(RECONCILE_GOLD), "correct": ok,
            "accuracy": round(ok / len(RECONCILE_GOLD), 3),
            "answers": dict(got_counts)}


def job_agent_decide() -> dict:
    """The maintenance agent's bounded JSON decision — schema conformance.

    G67: the agent has never run against real data, so nothing is known about
    whether a small model can hold its contract. Conformance is objective:
    a dict comes back, or it does not.
    """
    from src.workers.maintenance_agent import make_llm_decider

    decide = make_llm_decider()
    cases = [
        "Two codex entities look like the same thing: 'manhattan 5 lb book' and "
        "'manhattan 5lb book'. Should they be merged? Answer with action and reason.",
        "A stored fact says the user lives in Pune. A newer turn says they moved to "
        "Mumbai in March. What should happen to the old fact?",
        "An entity 'thing' has one edge and no description. Keep it or expire it?",
    ]
    ok, shape_ok = 0, 0
    for c in cases:
        got = decide(c, max_tokens=200, schema=AGENT_SCHEMA)
        if isinstance(got, dict):
            ok += 1
            if "action" in got and got.get("action") in \
                    AGENT_SCHEMA["properties"]["action"]["enum"]:
                shape_ok += 1
    return {"n": len(cases), "returned_dict": ok, "valid_action": shape_ok}


def job_doc_kind(turns) -> dict:
    """DOCUMENT vs TRANSCRIPT, against gold built from real material.

    ⚑ BOTH CLASSES ARE REAL, neither is written for the test. Transcripts are
    blobs of actual corpus turns; documents are chunks of this repo's own
    markdown. A hand-written "document" would be written to look like one.

    ⚠ The asymmetry matters more than the accuracy: `detect_blob_kind` defaults
    to DOCUMENT on failure *on purpose*, because a document mis-run through the
    slicer gets turns and roles INVENTED into it and stored as memory. So
    transcript→document is under-reading; document→transcript corrupts.
    """
    import glob

    from src.ingestion.documents.kind import detect_blob_kind

    cases = []
    for i in range(0, min(len(turns), 15), 3):
        chunk = turns[i:i + 3]
        if len(chunk) < 3:
            break
        blob = "\n\n".join(((t.get("prompt") or "").strip()[:300] + "\n\n"
                            + (t.get("response") or "").strip()[:400]) for t in chunk)
        cases.append((blob, "transcript"))
    for path in sorted(glob.glob("docs/*.md"))[:5]:
        try:
            cases.append((open(path).read()[2000:6000], "document"))
        except Exception:
            continue

    # ⚑ THE MODEL'S FAILURES MUST BE COUNTED, NOT CREDITED. Found by pointing
    # the whole harness at a nonexistent model: every other job died, and this
    # one scored **0.833**. `detect_blob_kind` swallows any model error and
    # returns DOCUMENT by design (a document mangled by the slicer is worse
    # than a transcript read flat), so a model that answers NOTHING collects
    # every document in the gold set for free. Unwrapped, this metric rewards
    # total failure — and it scored higher than a model that genuinely tried.
    from src.ingestion.documents.kind import _default_llm

    failures = {"n": 0}

    def counting_llm(prompt, system):
        try:
            return _default_llm(prompt, system)
        except Exception:
            failures["n"] += 1
            raise

    ok = 0
    confusion = Counter()
    for text, want in cases:
        got = detect_blob_kind(text, llm=counting_llm)
        confusion[f"{want}->{got}"] += 1
        ok += (got == want)
    # The dangerous direction, called out separately: a document read as a
    # transcript is the one that corrupts.
    out = {"n": len(cases), "correct": ok,
           "accuracy": round(ok / max(1, len(cases)), 3),
           "llm_failures": failures["n"],
           "document_read_as_transcript": confusion.get("document->transcript", 0),
           "confusion": dict(confusion)}
    if failures["n"]:
        # Say it in the number's own words rather than letting it read clean.
        out["accuracy"] = None
        out["VOID"] = (f"{failures['n']}/{len(cases)} model calls failed; the "
                       f"DOCUMENT default supplied the answers")
    return out


def job_conv_fold(turns, embedder) -> dict:
    """The conversation summary's rolling FOLD — a different task from the turn
    summary, and the one nobody has ever measured.

    `_summarize_chunk` carries an existing summary forward while folding in new
    turns. The failure it can hide is silent forgetting: the fold keeps getting
    shorter and the early conversation quietly disappears.
    """
    from src.workers.conversation_summary import _default_llm, _summarize_chunk

    existing, lens = "", []
    for i in range(0, min(len(turns), 12), 3):
        chunk = turns[i:i + 3]
        chunk_text = "\n\n".join(
            f"User: {(t.get('prompt') or '')[:300]}\n"
            f"Assistant: {(t.get('response') or '')[:400]}" for t in chunk)
        try:
            existing = _summarize_chunk(existing, chunk_text, _default_llm, embedder)
        except Exception as exc:                                   # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
        lens.append(len(existing or ""))
    # A fold that SHRINKS as it ingests more is dropping the early conversation.
    shrank = sum(1 for a, b in zip(lens, lens[1:]) if b < a)
    return {"folds": len(lens), "chars_after_each": lens,
            "shrank_steps": shrank,
            "final_chars": lens[-1] if lens else 0,
            "empty": 1 if not (existing or "").strip() else 0}


def job_procedural(limit: int = 8) -> dict:
    """Habit extraction, run against the EXISTING store.

    **⚑ WHY THE OLD STORE IS FINE HERE (maintainer, 2026-08-26).** The pre-reseed
    stores are dead data *because their codex graph was built by the 15%-correct
    extractor*. `procedural_extractor.py` contains **zero** codex references —
    verified by grep — so it reads episodic turns nobody has disputed. Same for
    `batch_summarizer.py`. Waiting for the reseed to measure these would have
    been waiting for something they do not depend on.

    **The scoring is self-validating.** The prompt demands
    `PATTERN: … | EVIDENCE: <message numbers>` and the caller REJECTS any reply
    whose evidence set is smaller than two. So "rows written / batches tried" is
    exactly the rate at which the model obeyed a rule it can be held to, with no
    judge in the loop.
    """
    from sqlalchemy import text as _t

    from src.api.db import SessionLocal
    from src.workers.procedural_extractor import extract_procedural

    db = SessionLocal()
    try:
        # ⚑ RESET TO EMPTY FIRST, EVERY MODEL. `extract_procedural` counts
        # SESSIONS, not extractions — re-reading a session it already processed
        # is neither an insert nor a reinforcement. Measured on the untouched
        # store: 43 rows before, 43 after, 0 reinforced, on every model. That is
        # not a model result, it is the job correctly declining to redo itself.
        # Clearing here is what makes each arm start from identical state.
        # Backed up to bakeoff/PRE_RESET_BACKUP.json; store restorable from
        # snapshots/dir-false-run1.sql.
        db.execute(_t("DELETE FROM procedural_memory"))
        # ⚑ AND THE IDEMPOTENCY KEYS, or the reset does nothing. Deleting the
        # rows alone left every session still marked done, and
        # `extract_procedural` returns **silently** on that check — no log line
        # at any level. Diagnosed only by running it at DEBUG and seeing NO
        # output at all. Keys are hashed, so they are recomputed here rather
        # than pattern-matched, which also keeps every other job's keys intact.
        from src.api.config import settings as _s
        from src.workers.idempotency import job_key
        sessions = db.execute(_t(
            "SELECT session_id, count(*) FROM episodic_memory "
            "WHERE session_id IS NOT NULL GROUP BY session_id")).all()
        keys = [job_key("procedural", f"{sid}:{n // max(1, _s.procedural_session_step)}")
                for sid, n in sessions]
        if keys:
            db.execute(_t("DELETE FROM idempotency_keys WHERE key = ANY(:k)"),
                       {"k": keys})
        db.commit()
        before = set()
        batches = [str(r[0]) for r in db.execute(_t(
            "SELECT DISTINCT batch_id FROM episodic_memory "
            "WHERE batch_id IS NOT NULL ORDER BY batch_id LIMIT :n"),
            {"n": limit}).all()]
        errors = 0
        for b in batches:
            try:
                extract_procedural(b)
            except Exception:
                errors += 1
        after = db.execute(_t(
            "SELECT id, pattern_name FROM procedural_memory")).all()
        new = [r for r in after if r[0] not in before]
        lens = [len(r[1] or "") for r in new]
        # ⚑ RESET, or the next model inherits this one's rows and the comparison
        # is between different starting states rather than between models.
        if new:
            db.execute(_t("DELETE FROM procedural_memory WHERE id = ANY(:i)"),
                       {"i": [r[0] for r in new]})
            db.commit()
        return {"batches_tried": len(batches), "errors": errors,
                "patterns_accepted": len(new),
                "accept_rate": round(len(new) / max(1, len(batches)), 3),
                "pattern_chars_median": int(statistics.median(lens)) if lens else 0}
    finally:
        db.close()


def job_batch_summary() -> dict:
    """Batch summaries over the existing store, then reset.

    ⚠ `batch_summarize()` selects turns whose `batch_summary_id IS NULL` and
    STAMPS them. Without the reset below, the first model summarises everything
    eligible and every later model correctly reports zero — which would read as
    a model failure and is actually the harness eating its own input.
    """
    from sqlalchemy import text as _t

    from src.api.db import SessionLocal
    from src.workers.batch_summarizer import batch_summarize

    db = SessionLocal()
    try:
        # ⚑ RESET TO UNSUMMARISED FIRST, EVERY MODEL. Selection is
        # `batch_summary_id IS NULL`, and the job STAMPS what it summarises — so
        # on the untouched store the only group left was 3 turns and it logged
        # `batch_summary_skipped_small`, writing 0 for every model. Clearing the
        # stamps hands each arm the same 180 turns of unsummarised work.
        db.execute(_t("UPDATE episodic_memory SET batch_summary_id = NULL"))
        db.execute(_t("DELETE FROM batch_summaries"))
        db.commit()
        before = set()
        err = None
        try:
            batch_summarize()
        except Exception as exc:                                    # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:150]}"
        rows = db.execute(_t("SELECT id, summary_text FROM batch_summaries")).all()
        new = [r for r in rows if r[0] not in before]
        lens = [len(r[1] or "") for r in new]
        empty = sum(1 for r in new if not (r[1] or "").strip())
        if new:
            ids = [r[0] for r in new]
            db.execute(_t("UPDATE episodic_memory SET batch_summary_id = NULL "
                          "WHERE batch_summary_id = ANY(:i)"), {"i": ids})
            db.execute(_t("DELETE FROM batch_summaries WHERE id = ANY(:i)"), {"i": ids})
            db.commit()
        out = {"summaries_written": len(new), "empty": empty,
               "chars_median": int(statistics.median(lens)) if lens else 0}
        if err:
            out["error"] = err
        return out
    finally:
        db.close()


def job_reflection(limit: int = 40) -> dict:
    """Reflection's FOUR codex-free sub-calls, against the existing store.

    **⚑ The split is clean, and it decides what is measurable here.**
    `run_reflection()` runs `_synthesize_session`, `_crystallize_patterns`,
    `_evolve_memory_slots`, `_detect_motifs` — none of which reference the codex
    graph — and then `_enrich_codex_entities`, which is entirely about it. Only
    the first four are run below, so the input is episodic turns nobody has
    disputed rather than the 15%-correct graph.

    **⚑ AND NOTHING IS WRITTEN.** The sub-calls only `db.add(...)`; the single
    `db.commit()` lives in `run_reflection` itself. Calling them directly and
    rolling back leaves the store byte-identical — no reset logic, no residue,
    which is the TRAPS #6/#15/#16 failure this project keeps paying for.

    ⛔ **The maintenance agent is NOT here on purpose.** Unlike reflection it is
    codex all the way down — pending `codex_reconciliation` items, `codex_edges`,
    entity merge pairs. Running it on the old store would score models on how
    well they reason about entities the dead extractor invented. Its *contract*
    is already covered by `agent_decide`, which is store-free.
    """
    from src.api.db import SessionLocal
    from src.memory.models import EpisodicMemory
    from src.workers.reflection import (
        _crystallize_patterns,
        _detect_motifs,
        _evolve_memory_slots,
        _synthesize_session,
    )

    db = SessionLocal()
    out: dict = {}
    try:
        recent = (db.query(EpisodicMemory)
                  .order_by(EpisodicMemory.timestamp.desc()).limit(limit).all())
        out["turns_in"] = len(recent)
        for name, fn in (("synthesize_session", _synthesize_session),
                         ("crystallize_patterns", _crystallize_patterns),
                         ("evolve_memory_slots", _evolve_memory_slots),
                         ("detect_motifs", _detect_motifs)):
            t0 = time.time()
            try:
                fn(db, recent)
                # `db.new` is what the sub-call staged. It is the only visible
                # product before the commit that is never going to happen.
                out[name] = {"staged": len(db.new), "s": round(time.time() - t0, 1)}
            except Exception as exc:                                # noqa: BLE001
                out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:110]}",
                             "s": round(time.time() - t0, 1)}
                db.rollback()
        out["produced_anything"] = sum(
            1 for k, v in out.items()
            if isinstance(v, dict) and v.get("staged", 0) > 0)
        return out
    finally:
        # Never commit. The store must come out of this exactly as it went in.
        db.rollback()
        db.close()


JOBS = {
    "summary": True,        # needs the embedder
    "cluster_name": False,
    "raw_slice": False,
    "reconcile": False,
    "agent_decide": False,
    "doc_kind": False,
    "conv_fold": True,      # needs the embedder
    # ── these read the EXISTING store. None touches the codex graph, which is
    #    the only reason the pre-reseed stores were declared dead data.
    "procedural": False,
    "batch_summary": False,
    "reflection": False,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated; default the full list")
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--jobs", default=",".join(JOBS), help="comma-separated subset")
    ap.add_argument("--list", action="store_true", help="print the candidate list and exit")
    args = ap.parse_args()

    models = [m.strip() for m in (args.models.split(",") if args.models else DEFAULT_MODELS)]
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip() in JOBS]
    if args.list:
        print("candidates:")
        for m in models:
            print(f"  {m}")
        print(f"\njobs: {', '.join(jobs)}")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    turns = load_turns(args.turns)
    print(f"corpus: {len(turns)} dense turns from {CORPUS}\n")

    embedder = None
    if any(JOBS[j] for j in jobs):
        from src.memory.embedder import get_embedder
        embedder = get_embedder()

    original = settings.background_model_name
    results: dict[str, dict] = defaultdict(dict)
    try:
        for model in models:
            print(f"══ {model}")
            # One arm = one model. Every job below resolves through
            # get_bg_model_name() at call time, so this one line repoints all
            # of them — the same mechanism seed_store.py --bg-model uses.
            settings.background_model_name = model

            # ⚑ WARM FIRST. Measured 2026-08-23: the first call after Ollama
            # loads a model returns different text from every call after it, on
            # identical input at temperature 0. An unwarmed arm is comparing a
            # cold draw against warm ones.
            t0 = time.time()
            try:
                from src.workers.bg_client_factory import bg_timeout, get_bg_client
                get_bg_client().chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "Say: ready"}],
                    max_tokens=8, temperature=0.0, timeout=bg_timeout(8))
                print(f"   warmed in {time.time()-t0:.0f}s")
            except Exception as exc:
                print(f"   ⛔ UNAVAILABLE ({type(exc).__name__}: {str(exc)[:90]}) — skipped")
                results[model] = {"_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                continue

            for job in jobs:
                t1 = time.time()
                try:
                    if job == "summary":
                        r = job_summary(turns, embedder)
                    elif job == "cluster_name":
                        r = job_cluster_name(turns)
                    elif job == "raw_slice":
                        r = job_raw_slice(turns)
                    elif job == "reconcile":
                        r = job_reconcile()
                    elif job == "doc_kind":
                        r = job_doc_kind(turns)
                    elif job == "conv_fold":
                        r = job_conv_fold(turns, embedder)
                    elif job == "procedural":
                        r = job_procedural()
                    elif job == "batch_summary":
                        r = job_batch_summary()
                    elif job == "reflection":
                        r = job_reflection()
                    else:
                        r = job_agent_decide()
                    r["_seconds"] = round(time.time() - t1, 1)
                except Exception as exc:                       # noqa: BLE001
                    # Loud, never silent: a job that blew up is a RESULT about
                    # this model, not a gap in the table.
                    r = {"_error": f"{type(exc).__name__}: {str(exc)[:200]}",
                         "_seconds": round(time.time() - t1, 1)}
                results[model][job] = r
                print(f"   {job:14} {json.dumps({k: v for k, v in r.items() if k != 'samples'})}")
    finally:
        settings.background_model_name = original

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"{OUT_DIR}/bakeoff_{stamp}.json"
    with open(path, "w") as fh:
        json.dump({"utc": stamp, "corpus": CORPUS, "turns": len(turns),
                   "jobs": jobs, "results": results}, fh, indent=2)
    print(f"\n→ {path}")
    print("⚠ experiments/curation_files is GITIGNORED — copy anything that "
          "matters into docs/PROVENANCE.md this session (TRAPS #40).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
