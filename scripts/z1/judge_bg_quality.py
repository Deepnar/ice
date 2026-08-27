#!/usr/bin/env python3
"""Z1: the QUALITY half for the background jobs that only ever had structure checks.

**The gap.** `bg_model_bakeoff.py` scored nine jobs, and for five of them every
number is presence, format or count:

  | job | what was measured | what was NEVER asked |
  |---|---|---|
  | batch summary | how many written, chars | is it faithful to the turns |
  | conversation fold | did it shrink | is the fold accurate |
  | cluster naming | digits / word count (the prompt's own rules) | is the name apt |
  | procedural | did it cite ≥2 messages | is the habit real |
  | reflection | did it stage an object | is the pattern evidenced |

Only the turn summary ever got a faithfulness judge. This adds one for the rest.

**⚑ EVERY JOB IS GATED BEFORE IT IS SCORED, with a defect of its own kind.**
`judge_summaries.py` established the pattern and it caught a real problem: a
judge that flags everything scores 100% detection and is worthless, so each job
here carries BOTH a planted defect (must be caught) and an untouched control
(must be passed). A job whose judge fails either arm is reported VOID rather
than ranked — see `judge_summaries.py` for the run where that mattered.

  * summarisation-shaped jobs (batch summary, fold) — splice in a fabricated
    sentence; control is the real output.
  * cluster naming — swap in the name generated for a DIFFERENT cluster. It is
    a fluent, well-formed name that is simply wrong about this content, which
    is exactly the failure rule-checking cannot see.
  * procedural — re-point the pattern's citations at messages that do not
    support it. The job's own output format makes this checkable.

⚠ **No caps on the evidence.** `judge_one` sends whole sources; the one time
this project truncated a judge's input it killed the result twice
([TRAPS #41](../../docs/TRAPS.md)).

  uv run python scripts/z1/judge_bg_quality.py --models qwen3:4b-instruct,gemma4:e4b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx  # noqa: E402

from src.api.config import settings  # noqa: E402

OUT_DIR = "experiments/curation_files/bakeoff"

SUPPORT_SYS = """You judge whether a piece of DERIVED TEXT is supported by the SOURCE it was derived from.

Return JSON: {"verdicts":[{"label":"...","why":"..."}]}

Labels, first that applies:
  fabricated  - states something the source does not support: a claim, number,
                name, relationship or outcome that is not there.
  incomplete  - invents nothing, but omits something central to the source.
  faithful    - everything stated is supported, and the substance survives.

Judge ONLY against the source. Do not reward fluency, length or structure."""

NAME_SYS = """You judge whether a NAME fits the conversation excerpts it was generated for.

Return JSON: {"verdicts":[{"label":"...","why":"..."}]}

Labels, first that applies:
  wrong    - the name describes subject matter these excerpts are not about.
  generic  - technically not wrong, but so broad it would fit almost any
             conversation ("General Discussion", "Various Topics").
  apt      - names the recurring subject these excerpts actually share.

A fluent, well-formed name that is about the wrong topic is `wrong`."""

FABRICATIONS = [
    " The user confirmed they had already deployed this to production on Tuesday.",
    " They mentioned their manager approved a budget of $40,000 for it.",
    " The assistant recommended switching to PostgreSQL 16 instead.",
    " This was the third time the same error had occurred that week.",
]


def _judge(client, model, system, source, derived, retries=3):
    from scripts.z1.judge_summaries import judge_one  # shares the no-cap rule
    return judge_one(client, model, source, derived, retries=retries,
                     system=system)


# ── producers: each returns [(source, derived)] via the PRODUCTION path ──────

def produce_batch_summary(model, turns):
    from src.workers.batch_summarizer import batch_summarize
    from sqlalchemy import text as _t
    from src.api.db import SessionLocal
    settings.background_model_name = model
    db = SessionLocal()
    try:
        db.execute(_t("UPDATE episodic_memory SET batch_summary_id = NULL"))
        db.execute(_t("DELETE FROM batch_summaries"))
        db.commit()
        batch_summarize()
        rows = db.execute(_t("SELECT id, summary_text FROM batch_summaries")).all()
        out = []
        for bid, summ in rows:
            src = db.execute(_t(
                "SELECT raw_text FROM episodic_memory WHERE batch_summary_id = :b "
                "ORDER BY timestamp"), {"b": bid}).all()
            if src and (summ or "").strip():
                out.append(("\n\n".join(r[0] or "" for r in src), summ))
        db.execute(_t("UPDATE episodic_memory SET batch_summary_id = NULL"))
        db.execute(_t("DELETE FROM batch_summaries"))
        db.commit()
        return out
    finally:
        db.close()


def produce_conv_fold(model, turns):
    from src.memory.embedder import get_embedder
    from src.workers.conversation_summary import _default_llm, _summarize_chunk
    settings.background_model_name = model
    emb = get_embedder()
    # ⚑ SEVERAL INDEPENDENT FOLDS, not one. The first version folded 12 turns
    # into a single result, so the job reported n=1 and was VOID on sample size
    # alone — a measurement that could never have said anything, whatever the
    # models did. Each conversation-sized slice below is its own fold.
    out = []
    per_fold = 9
    for start in range(0, len(turns) - per_fold + 1, per_fold):
        existing, seen = "", []
        for i in range(start, start + per_fold, 3):
            chunk = turns[i:i + 3]
            if len(chunk) < 3:
                break
            text_ = "\n\n".join(
                f"User: {t.get('prompt') or ''}\nAssistant: {t.get('response') or ''}"
                for t in chunk)
            seen.append(text_)
            existing = _summarize_chunk(existing, text_, _default_llm, emb)
        # ⚑ The fold is judged against EVERYTHING it folded, not the last chunk —
        # the failure it hides is losing the early conversation, and a judge
        # shown only the final chunk could never see that.
        if (existing or "").strip() and seen:
            out.append(("\n\n".join(seen), existing))
    return out


# ⚑ THE CURRENT NAMING PROMPT CONTRADICTS ITSELF — found 2026-08-27 while
# writing this arm. It says "NEVER be too general; be specific with the naming"
# and then, four lines later, "The name must be generic enough that a similar
# future turn about the SAME topic would also be assigned to this cluster."
# Be specific / be generic, in one prompt. That is a plausible cause of the
# 62-76% `wrong` rate, and arm B simply removes the contradiction rather than
# rewriting the task.
NAMING_HARD = "NEVER be too general; be specific with the naming. \n"
NAMING_SOFT = ("Name the SUBJECT these turns keep returning to. Prefer the most "
               "specific wording that would still fit a future turn on the same "
               "subject.\n")


class _NamingRewrite:
    def __init__(self, inner, counter):
        self._inner, self._c = inner, counter

    def create(self, *args, **kwargs):
        for m in kwargs.get("messages") or []:
            c = m.get("content") or ""
            if NAMING_HARD in c:
                m["content"] = c.replace(NAMING_HARD, NAMING_SOFT)
                self._c["rewritten"] += 1
            elif "NEVER be too general" in c:
                raise SystemExit("⛔ naming anchor drifted — arm B would be arm A.")
        return self._inner.create(*args, **kwargs)

    def __getattr__(self, k):
        return getattr(self._inner, k)


def produce_cluster_name(model, turns, soften=False):
    from src.workers import clustering as cl
    from src.workers.clustering import _generate_cluster_name
    settings.background_model_name = model
    counter = Counter()
    original = cl.bg_client
    if soften:
        class _C:
            def __init__(self, inner):
                self.completions = _NamingRewrite(inner.chat.completions, counter)
        class _B:
            def __init__(self, inner):
                self.chat = _C(inner)
        cl.bg_client = _B(original)
    out = []
    # ⚑ NO CAP. This was `min(len(turns), 25)` — 5 groups — so the job could
    # only ever report n=5 however many turns it was given. The sample was
    # limited by the harness, not by the store.
    for i in range(0, len(turns), 5):
        g = turns[i:i + 5]
        if len(g) < 5:
            break
        shim = [type("T", (), {"raw_text": f"User: {t.get('prompt','')}\n\n"
                                           f"Assistant: {t.get('response','')}"})()
                for t in g]
        name = _generate_cluster_name(shim)
        if name and name != "Unnamed Cluster":
            out.append(("\n\n".join(s.raw_text for s in shim), name))
    cl.bg_client = original
    if soften and not counter["rewritten"]:
        raise SystemExit("⛔ arm B never rewrote the naming prompt — it IS arm A.")
    return out


def produce_procedural(model, turns, limit=8):
    """Patterns with the messages they CITE — the citation makes it checkable."""
    from sqlalchemy import text as _t
    from src.api.db import SessionLocal
    from src.workers.procedural_extractor import extract_procedural
    from src.workers.idempotency import job_key
    settings.background_model_name = model
    db = SessionLocal()
    try:
        db.execute(_t("DELETE FROM procedural_memory"))
        sessions = db.execute(_t(
            "SELECT session_id, count(*) FROM episodic_memory "
            "WHERE session_id IS NOT NULL GROUP BY session_id")).all()
        keys = [job_key("procedural",
                        f"{sid}:{n // max(1, settings.procedural_session_step)}")
                for sid, n in sessions]
        if keys:
            db.execute(_t("DELETE FROM idempotency_keys WHERE key = ANY(:k)"),
                       {"k": keys})
        db.commit()
        batches = [str(r[0]) for r in db.execute(_t(
            "SELECT DISTINCT batch_id FROM episodic_memory WHERE batch_id IS NOT NULL "
            "ORDER BY batch_id LIMIT :n"), {"n": limit}).all()]
        for b in batches:
            try:
                extract_procedural(b)
            except Exception:
                pass
        out = []
        for pid, pname, sbatch in db.execute(_t(
                "SELECT id, pattern_name, source_batch_ids FROM procedural_memory")).all():
            src = db.execute(_t(
                "SELECT raw_text FROM episodic_memory WHERE batch_id = ANY(:b) "
                "ORDER BY timestamp"), {"b": list(sbatch or [])}).all()
            if src and (pname or "").strip():
                out.append(("\n\n".join(r[0] or "" for r in src), pname))
        db.execute(_t("DELETE FROM procedural_memory"))
        db.commit()
        return out
    finally:
        db.close()


JOBS = {
    "batch_summary": (produce_batch_summary, SUPPORT_SYS, "splice"),
    "conv_fold":     (produce_conv_fold, SUPPORT_SYS, "splice"),
    "cluster_name":  (produce_cluster_name, NAME_SYS, "swap"),
    "procedural":    (produce_procedural, SUPPORT_SYS, "swap"),
}
GOOD = {"batch_summary": "faithful", "conv_fold": "faithful",
        "cluster_name": "apt", "procedural": "faithful"}
BAD = {"batch_summary": "fabricated", "conv_fold": "fabricated",
       "cluster_name": "wrong", "procedural": "fabricated"}


def gate_and_score(client, judge, job, pairs, rng):
    """Plant a defect, prove the judge finds it, then score the real output."""
    good, bad = GOOD[job], BAD[job]
    system = JOBS[job][1]
    kind = JOBS[job][2]

    planted = []
    for i, (src, derived) in enumerate(pairs):
        if kind == "splice":
            planted.append((src, derived.rstrip() + rng.choice(FABRICATIONS)))
        else:
            # swap: a well-formed output belonging to DIFFERENT source material
            other = pairs[(i + 1) % len(pairs)][1] if len(pairs) > 1 else None
            if other and other != derived:
                planted.append((src, other))

    caught = sum(1 for s, d in planted
                 if _judge(client, judge, system, s, d).get("label") == bad)
    detect = caught / max(1, len(planted))
    real = [_judge(client, judge, system, s, d).get("label", "FAILED")
            for s, d in pairs]
    c = Counter(real)
    res = {"n": len(pairs), "planted_n": len(planted),
           "planted_defect_caught": round(detect, 3),
           "counts": dict(c),
           f"{good}_rate": round(c.get(good, 0) / max(1, len(pairs)), 3)}
    # ⚑ TWO DIFFERENT VOIDS, and the first version reported both with the same
    # sentence — so `conv_fold` on gemma4 printed "caught only 1/1 planted
    # defects", which reads as a judge failure and was actually 100% detection
    # on a sample of one. Saying why a number is void is the whole value of
    # voiding it.
    if len(pairs) < 3 or len(planted) < 3:
        res["VOID"] = (f"SAMPLE TOO SMALL — {len(pairs)} real unit(s), "
                       f"{len(planted)} planted. The gate is uninformative at "
                       f"this size (detection {detect:.0%}); the store simply "
                       f"does not contain enough material for this job.")
        res[f"{good}_rate"] = None
    elif detect < 0.6:
        res["VOID"] = (f"JUDGE CANNOT SEE THIS FAILURE MODE — caught "
                       f"{caught}/{len(planted)} defects that were PLANTED. "
                       f"The rates above are not evidence.")
        res[f"{good}_rate"] = None
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3:4b-instruct,gemma4:e4b")
    ap.add_argument("--jobs", default=",".join(JOBS))
    ap.add_argument("--judge", default=None)
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--naming-ab", action="store_true",
                    help="run cluster naming as an A/B: current prompt vs the "
                         "version with its self-contradiction removed")
    args = ap.parse_args()

    from scripts.z1.bg_model_bakeoff import load_turns
    judge = args.judge or settings.probe_model
    turns = load_turns(args.turns)
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip() in JOBS]
    rng = random.Random(20260826)
    client = httpx.Client(base_url=settings.probe_api_base_url,
                          headers={"Authorization": f"Bearer {settings.probe_api_key}"})

    out: dict = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n══ {model}")
        out[model] = {}
        for job in jobs:
            produce = JOBS[job][0]
            try:
                if job == "cluster_name" and args.naming_ab:
                    # Two arms, one changed substring. Reported side by side so
                    # the current prompt is never absent from the comparison.
                    a = produce(model, turns, soften=False)
                    b = produce(model, turns, soften=True)
                    ra = gate_and_score(client, judge, job, a, rng)
                    rb = gate_and_score(client, judge, job, b, rng)
                    out[model][job] = {"A_current": ra, "B_no_contradiction": rb}
                    print(f"   {job:15} A(current)     {json.dumps(ra)}")
                    print(f"   {job:15} B(uncontradicted) {json.dumps(rb)}")
                    continue
                pairs = produce(model, turns)
            except Exception as exc:                                # noqa: BLE001
                out[model][job] = {"_error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                print(f"   {job:15} ⛔ {out[model][job]['_error']}")
                continue
            if not pairs:
                out[model][job] = {"_error": "produced nothing to judge"}
                print(f"   {job:15} ⛔ produced nothing")
                continue
            res = gate_and_score(client, judge, job, pairs, rng)
            out[model][job] = res
            print(f"   {job:15} {json.dumps(res)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"{OUT_DIR}/bg_quality_{stamp}.json"
    json.dump({"judge": judge, "results": out}, open(path, "w"), indent=1)
    print(f"\n→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
