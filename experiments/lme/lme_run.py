#!/usr/bin/env python3
"""LongMemEval on ICE v2 (tag `v2-paper-eval`) -- resumable, unattended runner.

⚑ RUN THIS FROM THE WORKTREE so `uv` resolves v2's lockfile and `src.*` imports
resolve to the frozen tree. v2 embeds at truncate_dim=384; `main` embeds at 1024.
Running under main's venv produces silently wrong vectors.

    cd /home/deepnar/Programs/ice-worktrees/v2-paper-eval
    uv run python /home/deepnar/Programs/ice/experiments/lme/lme_run.py --phase oracle

RESUMABILITY -- the design constraint. One answer file per instance, written
atomically (tmp + os.replace). **Presence of the file IS the state**; there is no
progress file to fall out of sync with reality. Kill this at any time -- Ctrl-C,
lid close, power cut. The next invocation skips finished instances and redoes any
interrupted one from a full store wipe, which is what correctness demands anyway.

WHAT THIS DOES NOT DO: simulate decay days. LSREP ages memory between checkpoints
because it is measuring accumulation. LongMemEval asks once, at the end, so there
is nothing to age between; simulating decay here would model something the corpus
does not contain. Clustering runs once after ingestion so cluster-scoped retrieval
has something to scope to. Recorded as a deliberate deviation, not an oversight.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- repo layout -------------------------------------------------------------
HARNESS_DIR = Path(__file__).resolve().parent
DATA_DIR = HARNESS_DIR / "data"
DEFAULT_OUT = HARNESS_DIR / "runs" / "v2-paper-eval"

_stop_requested = False


def _on_signal(signum, _frame):
    """Stop AFTER the current instance, never mid-write."""
    global _stop_requested
    _stop_requested = True
    print(f"\n  [signal {signum}] finishing the current instance, then stopping. "
          f"Re-run the same command to resume.", flush=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write via tmp + os.replace. POSIX rename is atomic: a reader never sees
    a half-written answer, and a crash mid-write leaves the old file or none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _fmt(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# --- phases ------------------------------------------------------------------
# Sized against the measured corpus: median 492 turns/instance. See README.
PHASES = {
    # Control. Evidence sessions only -- ingestion is a rounding error. Proves the
    # adapter end-to-end before anything expensive. If ICE cannot answer from the
    # evidence sessions alone, the defect is HERE, not in ICE's memory.
    "oracle": {"corpus": "longmemeval_oracle", "select": "all"},
    # All 30 `_abs` instances. A complete, non-arbitrary unit, and the sharpest
    # test of a curation-first system: decline, or confabulate from retrieved noise?
    "abstention": {"corpus": "longmemeval_s", "select": "abstention"},
    # Stratified widening. Only after oracle and abstention look sane.
    "stratified": {"corpus": "longmemeval_s", "select": "stratified"},
}


def select_instances(corpus: list[dict], how: str, limit: int | None, seed: int) -> list[dict]:
    import random
    if how == "abstention":
        chosen = [x for x in corpus if x["question_id"].endswith("_abs")]
    elif how == "stratified":
        by_type: dict[str, list] = {}
        for x in corpus:
            by_type.setdefault(x["question_type"], []).append(x)
        rng = random.Random(seed)
        per = max(1, (limit or len(corpus)) // max(1, len(by_type)))
        chosen = []
        for _t, xs in sorted(by_type.items()):
            chosen.extend(rng.sample(xs, min(per, len(xs))))
    else:
        chosen = list(corpus)
    chosen.sort(key=lambda x: x["question_id"])  # deterministic order
    return chosen[:limit] if limit else chosen


# --- environment capture -----------------------------------------------------
def capture_env(db_url: str) -> dict:
    """Record what actually ran. PROVENANCE.md's standing rule: a run that cannot
    say what produced it is not a result."""
    def _sh(cmd, cwd=None):
        try:
            return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                  timeout=15).stdout.strip()
        except Exception:
            return "unavailable"

    env = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "system_under_test": "ICE v2 @ tag v2-paper-eval",
        "worktree_head": _sh(["git", "rev-parse", "HEAD"], cwd=os.getcwd()),
        "worktree_describe": _sh(["git", "describe", "--tags", "--always"], cwd=os.getcwd()),
        "harness_head": _sh(["git", "rev-parse", "HEAD"], cwd=str(HARNESS_DIR)),
        "database_url": db_url.rsplit("@", 1)[-1],  # host/db only, never credentials
    }
    for mod in ("torch", "transformers", "sentence_transformers", "pgvector"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "unavailable"

    manifest = DATA_DIR / "MANIFEST.json"
    if manifest.exists():
        env["corpus"] = json.loads(manifest.read_text())
    return env


# --- store lifecycle ---------------------------------------------------------
def wipe_store(SessionLocal, Base) -> None:
    """Every LongMemEval instance needs its OWN memory state -- haystacks cannot
    be shared. Truncating all mapped tables is derived from ORM metadata rather
    than a hand-listed set, so a schema change cannot leave a table un-wiped and
    silently leak evidence between instances."""
    from sqlalchemy import text
    db = SessionLocal()
    try:
        names = [t.name for t in Base.metadata.sorted_tables if t.name != "alembic_version"]
        if not names:
            raise RuntimeError("no mapped tables found -- refusing to run with an unknown schema")
        db.execute(text(f"TRUNCATE TABLE {', '.join(names)} RESTART IDENTITY CASCADE"))
        db.commit()
    finally:
        db.close()


def ingest_instance(inst, cid, classifier, embedder, SessionLocal, models) -> int:
    """Replay one haystack, mirroring the sequence that produced the paper's
    numbers (experiments/mature/run_mature_experiment.py).

    ⚑ Assistant text is stored VERBATIM. LongMemEval's fixed assistant replies
    carry evidence -- `single-session-assistant` is 56 instances -- so letting ICE
    regenerate them would destroy the haystack.
    """
    from src.workers.post_flight import is_lossless, generate_summary
    from src.workers.codex_extractor import extract_triplets, handle_triplet
    EpisodicMemory = models["EpisodicMemory"]

    sessions = inst["haystack_sessions"]
    dates = inst.get("haystack_dates") or []
    sids = inst.get("haystack_session_ids") or []
    stored = 0
    db = SessionLocal()
    try:
        for s_idx, session in enumerate(sessions):
            sid = sids[s_idx] if s_idx < len(sids) else f"s{s_idx}"
            base_ts = _parse_lme_date(dates[s_idx] if s_idx < len(dates) else None)

            # Sessions carry ONE date but many turns; nudge within the session so
            # ordering is preserved and the deterministic batch_id stays unique.
            pairs = _to_pairs(session)
            for t_idx, (prompt, response) in enumerate(pairs):
                ts = base_ts + timedelta(seconds=t_idx)
                batch_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{cid}:{sid}:{t_idx}")
                if db.query(EpisodicMemory).filter_by(batch_id=batch_id).first():
                    continue  # idempotent: a partial replay resumes cleanly

                classification = classifier.classify(prompt)
                emb = embedder.encode(prompt, convert_to_tensor=False).tolist()
                turn = EpisodicMemory(
                    conversation_id=cid,
                    batch_id=batch_id,
                    timestamp=ts,
                    topic_tags=classification.topic_tags,
                    intent_tags=classification.intent_tags,
                    context_reliance=classification.context_reliance,
                    raw_text=f"User: {prompt}\n\nAssistant: {response}",
                    embedding=emb,
                    idempotency_key=str(uuid.uuid4()),
                )
                db.add(turn)
                db.flush()

                lossless = is_lossless(response)
                force_lossless = bool(
                    turn.topic_tags and ("Creative_&_Media" in turn.topic_tags
                                         or "Emotional_Processing" in (turn.intent_tags or []))
                )
                if force_lossless:
                    lossless, inject_raw, summary = True, True, None
                else:
                    full_text = f"User: {prompt}\nAssistant: {response}"
                    word_count = len(full_text.split())
                    has_code = "```" in response
                    inject_raw = True
                    if lossless and word_count > 500 and not has_code:
                        summary, inject_raw = generate_summary(prompt, response), False
                    elif not lossless:
                        summary, inject_raw = generate_summary(prompt, response), False
                    else:
                        summary = None
                turn.lossless_flag = lossless
                turn.summary_text = summary
                turn.inject_raw = inject_raw

                if lossless:
                    for triplet in extract_triplets(turn.raw_text, topic_tags=turn.topic_tags):
                        if not isinstance(triplet, dict):
                            continue
                        s, r, o = (triplet.get("subject"), triplet.get("relation"),
                                   triplet.get("object"))
                        if all(isinstance(v, str) and v.strip() for v in (s, r, o)):
                            handle_triplet(db, s.strip(), r.strip(), o.strip(), str(batch_id))
                db.commit()
                stored += 1
    finally:
        db.close()
    return stored


def _to_pairs(session: list[dict]) -> list[tuple[str, str]]:
    """Fold a session's [{role, content}, ...] into (user, assistant) pairs.
    An unpaired trailing user turn is kept with an empty response rather than
    dropped -- dropping it would silently delete evidence."""
    pairs, pending = [], None
    for turn in session:
        role, content = turn.get("role"), turn.get("content") or ""
        if role == "user":
            if pending is not None:
                pairs.append((pending, ""))
            pending = content
        elif role == "assistant":
            pairs.append((pending if pending is not None else "", content))
            pending = None
    if pending is not None:
        pairs.append((pending, ""))
    return pairs


def _parse_lme_date(raw: str | None) -> datetime:
    """LongMemEval dates look like '2023/05/20 (Sat) 02:21'."""
    if not raw:
        return datetime.now(timezone.utc)
    cleaned = " ".join(p for p in raw.split() if not p.startswith("("))
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plan-only", action="store_true",
                    help="print what would run and exit -- touches nothing")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    spec = PHASES[args.phase]
    corpus_path = DATA_DIR / spec["corpus"]
    if not corpus_path.exists():
        print(f"⛔ corpus missing: {corpus_path}\n   run: uv run python "
              f"{HARNESS_DIR / 'fetch_dataset.py'}", file=sys.stderr)
        return 1

    corpus = json.loads(corpus_path.read_text())
    instances = select_instances(corpus, spec["select"], args.limit, args.seed)
    out_dir = args.out / args.phase
    answers_dir = out_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    done = {p.stem for p in answers_dir.glob("*.json")}
    todo = [x for x in instances if x["question_id"] not in done]
    turns_todo = sum(sum(len(s) for s in x["haystack_sessions"]) for x in todo)

    print(f"phase={args.phase}  corpus={spec['corpus']}")
    print(f"  selected {len(instances)}  done {len(done)}  remaining {len(todo)}")
    print(f"  ~{turns_todo:,} turns to ingest")
    if args.plan_only:
        for x in todo[:10]:
            n = sum(len(s) for s in x["haystack_sessions"])
            print(f"    {x['question_id']:12} {x['question_type']:26} {n:4d} turns")
        if len(todo) > 10:
            print(f"    ... and {len(todo) - 10} more")
        return 0
    if not todo:
        print("  nothing to do -- all selected instances already have answers.")
        return 0

    # Imports deferred until after --plan-only so the planner needs no DB or GPU.
    from src.api.config import settings
    from src.api.db import SessionLocal
    from src.memory.models import Base, Conversation, EpisodicMemory
    from src.classifier.classifier import PyTorchClassifier

    env = capture_env(settings.database_url)
    _atomic_write_json(out_dir / "MANIFEST.json", {"phase": args.phase,
                                                   "selection": spec["select"],
                                                   "seed": args.seed,
                                                   "n_selected": len(instances),
                                                   "environment": env})
    classifier = PyTorchClassifier()
    embedder = classifier.embedder
    dim = len(embedder.encode("dimension probe", convert_to_tensor=False))
    if dim != 384:
        print(f"⛔ embedder returned {dim} dims, expected 384.\n"
              f"   v2 stores Vector(384); `main` embeds at 1024 with the SAME model\n"
              f"   name. You are almost certainly running under main's venv.\n"
              f"   Run this from the worktree so uv resolves v2's lockfile.",
              file=sys.stderr)
        return 2
    print(f"  embedder OK: {dim} dims\n")

    models = {"EpisodicMemory": EpisodicMemory}
    started = time.time()
    durations: list[float] = []

    for i, inst in enumerate(todo, 1):
        if _stop_requested:
            print("  stopped by request; progress is on disk.")
            break
        qid = inst["question_id"]
        n_turns = sum(len(s) for s in inst["haystack_sessions"])
        t0 = time.time()
        eta = (sum(durations) / len(durations) * (len(todo) - i + 1)) if durations else None
        print(f"[{i:4d}/{len(todo)}] {qid:12} {inst['question_type']:26} "
              f"{n_turns:4d} turns  elapsed {_fmt(time.time() - started)}"
              f"{'  eta ' + _fmt(eta) if eta else ''}", flush=True)

        try:
            # Fresh state per instance. Wipe FIRST, so an instance interrupted last
            # run cannot leak its rows into this one.
            wipe_store(SessionLocal, Base)
            cid = uuid.uuid5(uuid.NAMESPACE_DNS, f"lme:{args.phase}:{qid}")
            db = SessionLocal()
            try:
                db.add(Conversation(id=cid))
                db.commit()
            finally:
                db.close()

            stored = ingest_instance(inst, cid, classifier, embedder, SessionLocal, models)

            # Clustering once, so cluster-scoped retrieval has something to scope
            # to. No decay simulation -- see the module docstring.
            from src.workers.clustering import cluster_turns, merge_similar_clusters
            cluster_turns()
            merge_similar_clusters()

            _atomic_write_json(answers_dir / f"{qid}.json", {
                "question_id": qid,
                "question_type": inst["question_type"],
                "is_abstention": qid.endswith("_abs"),
                "question": inst["question"],
                "question_date": inst.get("question_date"),
                "reference_answer": inst.get("answer"),
                "conversation_id": str(cid),
                "turns_stored": stored,
                "turns_in_haystack": n_turns,
                "ingest_seconds": round(time.time() - t0, 1),
                "answers": {},          # filled by the answer pass
                "status": "ingested",
            })
            durations.append(time.time() - t0)
            print(f"           ingested {stored}/{n_turns} in {_fmt(time.time() - t0)}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 -- one bad instance must not end the run
            # No answer file is written, so this instance is simply retried next
            # run. Recorded so a systematic failure is visible rather than silent.
            print(f"           ⛔ FAILED: {type(exc).__name__}: {exc}", flush=True)
            with (out_dir / "failures.log").open("a") as fh:
                fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{qid}\t"
                         f"{type(exc).__name__}\t{exc}\n")

    remaining = len([x for x in instances
                     if x["question_id"] not in {p.stem for p in answers_dir.glob('*.json')}])
    print(f"\ndone this pass. {remaining} instance(s) still outstanding.")
    print(f"artifacts: {out_dir}")
    if remaining:
        print("re-run the same command to continue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
