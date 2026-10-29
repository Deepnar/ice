"""Z2 S12: the LongMemEval oracle on the current system (spec D12, V3_POST_RESEED_REPAIR §2.6).

The pinned public oracle file (500 questions; evidence sessions only — no distractor
sessions, unlike LongMemEval-S) is replayed one question at a time:

1. the scratch store is wiped (each question's history is independent);
2. its sessions, sorted by date (the file does not order them), each become their own
   ICE conversation — the shape v2's corrected adapter established; flattening them
   into one conversation measures within-chat memory instead — and are replayed through
   `seed_v3.run` as in Z2's semi-LSREP: every turn through the whole pipeline with its
   writes kept, the recorded reply stored and post-flighted, maintenance on its cadence,
   live decay. Turns inside a session are TURN_GAP apart from the session's date;
3. at the question date (overdue maintenance run then, on that clock) the question is
   read from a fresh, empty conversation — no recent window, so the answer must come
   from memory — under full ICE, warm vector, recent-only and bare vector, plus the
   all-evidence control: every supplied session given straight to the same answerer
   (an answer the answerer cannot fit is recorded, never truncated);
4. what decay did to the evidence turns by question time (decay score, archived, cold)
   is recorded per question, so a miss caused by decay is visible.

Units use read_passes' format, so `answers.py` answers them; `grade_oracle` grades them
with LongMemEval's own judge prompts (experiments/lme/score.py, verbatim) by Muse —
within-study, non-official (the official GPT-4o judge is not used) — plus the
qualified Muse grader against the answer and the turns that hold it (`has_answer`).
Every question is a durable unit: a crash re-runs only the question in flight.

    uv run --no-sync python experiments/z2/oracle.py run --database ice_z2_main_lme --out logs/z2-run-main/s12
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-lme-oracle-1"
DATA = ROOT / "experiments/lme/data/longmemeval_oracle"
MANIFEST = ROOT / "experiments/lme/data/MANIFEST.json"
# FINAL (Z3 D3) runs the same machinery on LongMemEval-S's pinned 100-instance slice.
DATASETS = {"oracle": ("longmemeval_oracle", "ice-z2-lme-oracle-1"),
            "s": ("longmemeval_s", "ice-final-lme-s-1")}
DATASET = "oracle"


def select_dataset(name):
    """Point the module at one pinned LongMemEval file (and its policy tag)."""
    global DATA, POLICY, DATASET
    key, POLICY = DATASETS[name]
    DATA, DATASET = ROOT / "experiments/lme/data" / key, name
TURN_GAP = timedelta(minutes=2)
CONDITIONS = ("full", "warm_vector", "recent_only", "bare_vector", "all_evidence")
ARMS = {"full": "full", "warm_vector": "vector_only", "recent_only": "recent_only"}
# FINAL's LongMemEval-S slice (2026-10-28): run the way LongMemEval's own field runs it — ICE,
# plain vector search, full context (every haystack session, ~115K tokens) and LongMemEval's own
# oracle setting (only the answer sessions). recent_only is read only as the others' prompt frame;
# FINAL does not answer it.
CONDITIONS_S = ("full", "recent_only", "bare_vector", "all_evidence", "full_context")
ARMS_S = {"full": "full", "recent_only": "recent_only"}


def conditions_now():
    return CONDITIONS_S if DATASET == "s" else CONDITIONS


# ── the dataset ─────────────────────────────────────────────────────────────

def load_instances(path=None, manifest=MANIFEST):
    """The pinned LongMemEval file, refused if its bytes are not the manifest's."""
    path, key = path or DATA, DATASETS[DATASET][0]
    expected = json.loads(Path(manifest).read_text())["files"][key]["sha256"]
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{key} differs from its pinned sha256")
    data = json.loads(raw)
    if len(data) != 500:
        raise ValueError(f"expected 500 instances, found {len(data)}")
    return data


def parse_date(raw):
    """LongMemEval dates look like '2023/05/20 (Sat) 02:21'."""
    cleaned = " ".join(p for p in (raw or "").split() if not p.startswith("("))
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unparseable LongMemEval date {raw!r}")


def pairs(session):
    """(user, assistant, has_answer) pairs; an unpaired turn is kept with an empty side."""
    out, pending = [], None
    for turn in session:
        role, content, marked = turn.get("role"), turn.get("content") or "", bool(turn.get("has_answer"))
        if role == "user":
            if pending is not None:
                out.append((pending[0], "", pending[1]))
            pending = (content, marked)
        elif role == "assistant":
            user, user_marked = pending if pending is not None else ("", False)
            out.append((user, content, user_marked or marked))
            pending = None
    if pending is not None:
        out.append((pending[0], "", pending[1]))
    return out


def layout(inst):
    """Sessions sorted by date as seed_v3 conversations, plus the question time."""
    sessions, answer_sessions = [], set(inst.get("answer_session_ids") or [])
    for index, messages in enumerate(inst["haystack_sessions"]):
        sessions.append(dict(index=index, session_id=inst["haystack_session_ids"][index],
                             start=parse_date(inst["haystack_dates"][index]), pairs=pairs(messages)))
    sessions.sort(key=lambda s: (s["start"], s["index"]))
    conversations, evidence, previous_end = {}, [], None
    for k, s in enumerate(sessions):
        slug = f"lme{k:02d}"
        start = s["start"] if previous_end is None or s["start"] > previous_end else previous_end + TURN_GAP
        rows = []
        for n, (user, assistant, marked) in enumerate(s["pairs"], 1):
            rows.append(dict(turn_number=n, timestamp=start + (n - 1) * TURN_GAP, ts_provenance="original",
                             prompt=user, response=assistant))
            if marked:
                evidence.append(dict(slug=slug, turn=n, session_id=s["session_id"]))
        conversations[slug] = rows
        previous_end = rows[-1]["timestamp"] if rows else previous_end
    asked = parse_date(inst["question_date"])
    return dict(conversations=conversations, evidence=evidence, question_time=asked,
                sessions=[dict(slug=f"lme{k:02d}", session_id=s["session_id"], start=s["start"].isoformat(),
                               evidence=s["session_id"] in answer_sessions)
                          for k, s in enumerate(sessions)],
                eligible=previous_end is None or asked > previous_end)


def ask_of(inst):
    return dict(id=inst["question_id"], question=inst["question"], kind="question", type=inst["question_type"],
                checkpoint_turn=None, answer=inst["answer"], abstention=inst["question_id"].endswith("_abs"),
                question_date=inst["question_date"])


# ── one question ────────────────────────────────────────────────────────────

def wipe(engine):
    from sqlalchemy import text
    from src.memory.models import Base
    if not engine.url.database.startswith("ice_z2_"):
        raise RuntimeError("wipe refused outside an ice_z2_ scratch database")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE " + ", ".join(f'"{t}"' for t in Base.metadata.tables) + " RESTART IDENTITY CASCADE"))


def replay(inst, plan, directory):
    """Every session through seed_v3.run; overdue maintenance at the question time."""
    from scripts.z1 import seed_v3
    from experiments.z2.replay import boundary_clock
    convs = plan["conversations"]
    last_slug = list(convs)[-1]
    last_turn = convs[last_slug][-1]["turn_number"]
    real = seed_v3.due_maintenance

    def maintenance(db, sink, source_time, conversation_id, last_run, recovery=None, context=None):
        result = real(db, sink, source_time, conversation_id, last_run, recovery, context)
        context = context or {}
        if context.get("conversation") == last_slug and context.get("turn") == last_turn:
            with boundary_clock(plan["question_time"], db.get_bind()):
                real(db, sink, plan["question_time"], conversation_id, last_run, recovery,
                     dict(context, stage="question_time"))
        return result

    args = SimpleNamespace(out=str((directory / "seed.jsonl").relative_to(ROOT)), resume=False, checkpoint_every=100_000,
                           checkpoint_dir=str((directory / "seed.recovery").relative_to(ROOT)),
                           worker_failure_policy="continue", development_repeat_review=None, limit=0,
                           conversation=None, no_maintenance=False, no_probe_controls=True, probe_panel="immediate")
    with patch.object(seed_v3, "due_maintenance", maintenance), \
            patch.dict(seed_v3.SOURCE_TIMESTAMP_PROVENANCE, {slug: "original" for slug in convs}):
        code = seed_v3.run(args, convs, {}, expected_turns={s: len(r) for s, r in convs.items()},
                           input_files=[DATA], selected_plan=dict(policy=POLICY, question_id=inst["question_id"]))
    if code:
        raise RuntimeError(f"replay exited {code}")


def evidence_state(db, plan):
    """Decay state of the turns holding the answer, at question time."""
    from sqlalchemy import text
    keys = {f"z1seed-{e['slug']}-{e['turn']}" for e in plan["evidence"]}
    live = {r.idempotency_key: dict(decay_score=r.decay_score, archived=bool(r.is_archived)) for r in db.execute(text(
        "SELECT idempotency_key, decay_score, is_archived FROM episodic_memory WHERE idempotency_key = ANY(:k)"),
        {"k": list(keys)})}
    cold = {r[0] for r in db.execute(text("SELECT idempotency_key FROM cold_storage WHERE idempotency_key = ANY(:k)"),
                                     {"k": list(keys)})}
    return dict(turns=len(keys), live=len(live), archived=sum(v["archived"] for v in live.values()), cold=len(cold),
                missing=len(keys - set(live) - cold),
                min_decay=min((v["decay_score"] for v in live.values() if v["decay_score"] is not None), default=None))


def all_evidence_messages(recent_stage, plan, every=False):
    """recent_only's envelope with the evidence sessions as the evidence pair (in the
    oracle file every session is evidence; in LongMemEval-S only answer_session_ids).
    *every*: all sessions in date order — LongMemEval-S's full-context condition."""
    from experiments.z2.read_passes import CONTEXT_ACK, CONTEXT_HEADER
    parts = []
    for s in [s for s in plan["sessions"] if every or s.get("evidence", True)]:
        lines = [f"[Session {s['session_id']}, {s['start'][:16]}]"]
        for row in plan["conversations"][s["slug"]]:
            lines.append(f"User: {row['prompt']}\nAssistant: {row['response']}")
        parts.append("\n\n".join(lines))
    messages = [dict(m) for m in recent_stage["prompt_messages"]]
    return messages[:-1] + [{"role": "user", "content": f"{CONTEXT_HEADER}\n" + "\n\n".join(parts)},
                            {"role": "assistant", "content": CONTEXT_ACK}] + messages[-1:]


def read(inst, plan, out, reader):
    """The five conditions from a fresh, empty conversation at the question date."""
    import uuid
    from experiments.v3_repair.bare_vector import Chunk, Index, select
    from experiments.z2 import read_passes as R
    from src.api.db import SessionLocal
    from src.memory import tokens
    from src.memory.models import Conversation
    ask = ask_of(inst)
    now = plan["question_time"]
    with SessionLocal() as db:
        query = Conversation(id=uuid.uuid4(), memory_scope_type="auto", created_at=now)
        db.add(query)
        db.commit()
        qid = str(query.id)
    stages = {}
    for name, arm in (ARMS_S if DATASET == "s" else ARMS).items():
        with SessionLocal() as db:
            stages[name] = reader.read(db, qid, ask, arm, now)
    with SessionLocal() as db:
        decay = evidence_state(db, plan)
    # bare vector: every original message of every session, numbered across sessions
    chunks, number = [], 0
    for s in plan["sessions"]:
        for row in plan["conversations"][s["slug"]]:
            number += 1
            for role, text in (("user", row["prompt"]), ("assistant", row["response"])):
                if text.strip():
                    chunks.append(Chunk(number, role, 0, len(text), text))
    encode = lambda texts: reader.embedder.encode(texts, convert_to_tensor=False, batch_size=8, show_progress_bar=False)
    budget = stages["full"].get("retrieval_budget_if_retrieved")
    chosen = select(Index(encode), ask["question"], chunks, budget, None, tokens.count)
    stages["bare_vector"] = dict(prompt_messages=R.bare_messages(stages["recent_only"], chosen, ask["question"]),
                                 retrieval_budget=budget, selected_tokens=sum(tokens.count(c.render()) for c, _ in chosen),
                                 selected=[dict(turn=c.turn, role=c.role, score=sc) for c, sc in chosen])
    evidence = all_evidence_messages(stages["recent_only"], plan)
    stages["all_evidence"] = dict(prompt_messages=evidence, selected_tokens=None,
                                  prompt_tokens=tokens.count(json.dumps(evidence, ensure_ascii=False)))
    if DATASET == "s":
        everything = all_evidence_messages(stages["recent_only"], plan, every=True)
        stages["full_context"] = dict(prompt_messages=everything, selected_tokens=None,
                                      prompt_tokens=tokens.count(json.dumps(everything, ensure_ascii=False)))
    for name in conditions_now():
        R.write_unit(R.unit_path(out, name, ask["id"]), dict(
            policy=POLICY, history="lme-oracle", condition=name, ask=ask, checkpoint=None,
            question_time=now, decay_overlay="live", stage=stages[name], evidence_decay=decay))
    return decay


def run(out, *, limit=None, only=None, shard=None):
    """Child side: every pending question, one durable unit each; with *shard* (i, n) only the
    questions at positions i, i+n, ... — each worker its own scratch database (every question
    wipes and replays its own sessions, so they are independent)."""
    from sqlalchemy import create_engine
    from src.api.config import settings
    from experiments.z2.read_passes import Reader
    instances = load_instances()
    if only:
        instances = [i for i in instances if i["question_id"] in only]
    if limit:
        instances = instances[:limit]
    if shard:
        instances = [inst for k, inst in enumerate(instances) if k % shard[1] == shard[0]]
    engine = create_engine(settings.database_url)
    reader = None
    failures, skipped = [], []
    for inst in instances:
        directory = out / "instances" / inst["question_id"]
        done = directory / "done.json"
        if done.exists():
            continue
        started = time.time()
        plan = layout(inst)
        if not plan["eligible"]:
            # never replayable: LongMemEval dates it inside its own history — skipped, not a failure
            # (a failure made the runner relaunch S12 forever; 44 of 500, 2026-10-29)
            skipped.append(dict(question_id=inst["question_id"], reason="question date is not after the history"))
            continue
        if directory.exists():
            shutil.rmtree(directory)                 # an unfinished attempt: start the question afresh
        directory.mkdir(parents=True)
        try:
            wipe(engine)
            replay(inst, plan, directory)
            reader = reader or Reader()
            decay = read(inst, plan, out, reader)
        except Exception as exc:                     # this question only; the next run retries it
            failures.append(dict(question_id=inst["question_id"], reason=f"{type(exc).__name__}: {exc}"[:500]))
            continue
        done.write_text(json.dumps(dict(policy=POLICY, question_id=inst["question_id"],
                                        turns=sum(len(r) for r in plan["conversations"].values()),
                                        sessions=len(plan["conversations"]), evidence_decay=decay,
                                        seconds=round(time.time() - started)), default=str) + "\n")
        print(json.dumps(dict(question=inst["question_id"], seconds=round(time.time() - started),
                              evidence_decay=decay), default=str), flush=True)
    suffix = f"-{shard[0]}" if shard else ""
    (out / f"failures{suffix}.json").write_text(json.dumps(failures, indent=1) + "\n")
    (out / f"skipped{suffix}.json").write_text(json.dumps(skipped, indent=1) + "\n")
    return failures


def progress(out, total=500):
    """(questions done, total) for the runner's meter, from files only; skipped questions
    (never replayable) leave the total."""
    skipped = {s["question_id"] for p in Path(out).glob("skipped*.json") for s in json.loads(p.read_text())}
    return len(list((Path(out) / "instances").glob("*/done.json"))), total - len(skipped)


# ── grading ─────────────────────────────────────────────────────────────────

def grade_oracle(units, answers, out, *, ask, ask_text, upstream_only=False):
    """Upstream LongMemEval yes/no (its prompts, verbatim) and the qualified Muse grade.
    *upstream_only* (FINAL's LongMemEval-S, 2026-10-28): LongMemEval's own judge only, as the
    field reports it."""
    sys.path.insert(0, str(ROOT / "experiments/lme"))
    from experiments.lme.score import get_anscheck_prompt
    from experiments.z2 import grade as G
    contexts = []
    index = json.loads((Path(answers) / "index.json").read_text())["index"]
    rows = {json.loads(l)["prompt_sha256"]: json.loads(l)
            for l in (Path(answers) / "answers.jsonl").read_text().splitlines() if l.strip()}
    instances = {i["question_id"]: i for i in load_instances()}
    for path in sorted(Path(units).glob("units/*/*.json")):
        unit = json.loads(path.read_text())
        answer = rows.get(index.get(f"{unit['condition']}|{unit['ask']['id']}"))
        if answer:
            contexts.append((unit, answer))
    def one(item):
        unit, answer = item
        key = f"{unit['condition']}|{unit['ask']['id']}"
        if answer["finish"] != "complete":
            return key, dict(status="ungraded", reason=f"answer {answer['finish']}")
        a, inst = unit["ask"], instances[unit["ask"]["id"]]
        upstream = ask_text(get_anscheck_prompt(a["type"], a["question"], a["answer"], answer["text"],
                                                abstention=a["abstention"]))
        marked = [dict(turn=None, role=t["role"], text=t["content"]) for s in inst["haystack_sessions"] for t in s
                  if t.get("has_answer")]
        body = dict(QUESTION=a["question"], REQUIRED=[str(a["answer"])], SOURCES=marked, SUPERSEDED=[],
                    EXPECT_ABSTENTION=a["abstention"], ANSWER=answer["text"])
        rich = {} if upstream_only else ask(G.GRADE_SYSTEM, json.dumps(body, ensure_ascii=False)) or {}
        return key, dict(status="graded", upstream=_yes(upstream), grade=rich.get("grade"), score=rich.get("score"),
                         type=a["type"], abstention=a["abstention"], evidence_decay=unit.get("evidence_decay"))

    # answers side by side (one at a time, S12's ~2,500 grades were ~10 hours); every call is durable
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=G.CLAIM_WORKERS) as pool:
        results = dict(pool.map(one, contexts))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "oracle-grades.json").write_text(json.dumps(results, indent=1) + "\n")
    (Path(out) / "oracle-report.json").write_text(json.dumps(report(results), indent=1) + "\n")
    return results


def _yes(text):
    """The upstream prompt asks for a bare yes/no: the first yes/no word decides; None if neither."""
    import re
    found = re.search(r"\b(yes|no)\b", (text or "").strip().lower())
    return None if not found else found.group(1) == "yes"


def text_caller(calls, generate, meta, session):
    """A durable plain-text judge call (the upstream prompt is not JSON)."""
    import uuid as _uuid
    from experiments.z2 import ledger as L
    from experiments.z2 import ledger_merge as M

    def ask_text(prompt):
        identity = L.sha256(json.dumps(dict(policy=M.CALL_POLICY, kind="upstream", meta=meta,
                                            prompt=L.sha256(prompt)), sort_keys=True))

        def run():
            result = generate([{"role": "user", "content": prompt}], temperature=0, max_output_tokens=4000,
                              session_id=session)
            return result.text, result.usage
        return calls.get(identity, run)["text"]
    return ask_text


def report(results):
    import collections
    table = collections.defaultdict(lambda: collections.Counter())
    for key, r in results.items():
        cond = key.split("|", 1)[0]
        bucket = "abstention" if r.get("abstention") else r.get("type")
        c = table[(cond, bucket)]
        c["n"] += 1
        if r.get("status") != "graded":
            c["ungraded"] += 1
            continue
        c["upstream_yes"] += r["upstream"] is True
        c["upstream_spoken"] += r["upstream"] is not None
        c[f"grade_{r.get('grade')}"] += 1
    return {f"{c}|{b}": dict(v, upstream_accuracy=round(v["upstream_yes"] / v["upstream_spoken"], 3)
                             if v["upstream_spoken"] else None) for (c, b), v in sorted(table.items())}


def worker_argv(argv, k, n):
    """Worker k of n's command line: the same run, database <database>_k, --shard k/n."""
    out = []
    skip = False
    for j, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == "--workers":
            skip = True
            continue
        out.append(f"{a}_{k}" if j and argv[j - 1] == "--database" else a)
    return out + ["--shard", f"{k}/{n}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--database", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--only", nargs="*")
    r.add_argument("--dataset", choices=sorted(DATASETS), default="oracle")
    r.add_argument("--slice", type=Path, help="a pinned slice (experiments/lme/final_slice.json): its question_ids")
    r.add_argument("--workers", type=int, default=1,
                   help="questions side by side, each worker its own database <database>_<i> (2026-10-28: "
                        "one at a time, S12's background calls never overlapped and the GPU sat idle)")
    r.add_argument("--shard", help="i/n: this worker's share (set by --workers)")
    g = sub.add_parser("grade")
    g.add_argument("--dataset", choices=sorted(DATASETS), default="oracle")
    g.add_argument("--judge", default="muse", help="judge role")
    g.add_argument("--units", type=Path, required=True)
    g.add_argument("--answers", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--upstream-only", action="store_true", help="LongMemEval's own judge only (FINAL)")
    args = parser.parse_args()
    select_dataset(args.dataset)
    if getattr(args, "slice", None):
        args.only = json.loads(args.slice.read_text())["question_ids"]
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "logs"):
        raise ValueError("oracle outputs are private: keep them under logs/")
    if args.command == "grade":
        from experiments.lme.cloud_provider import load_selected_env
        from experiments.z2 import ledger_merge as M
        from experiments.z2.grade import judge_caller
        from experiments.z2.profiles import generator, metadata
        load_selected_env()
        out.mkdir(parents=True, exist_ok=True)
        calls = M.Calls(out / "oracle-grade-raw.jsonl")
        ask = judge_caller(calls, generator(args.judge).generate, metadata(args.judge), "lme-oracle", role=args.judge)
        ask_text = text_caller(calls, generator(args.judge).generate, metadata(args.judge), "ice-z2-lme-oracle-upstream")
        grade_oracle(args.units, args.answers, out, ask=ask, ask_text=ask_text, upstream_only=args.upstream_only)
        return 0
    if args.workers > 1 and not args.shard:
        import signal
        import subprocess
        children = [subprocess.Popen([sys.executable, *worker_argv(sys.argv, k, args.workers)])
                    for k in range(args.workers)]

        def stop(signum, _frame):
            # a stop reaches every worker (the runner stops this process only; 2026-10-28 the
            # workers ran on after it); each leaves its unfinished question to be redone
            for c in children:
                if c.poll() is None:
                    c.send_signal(signum)
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        return max(c.wait() for c in children)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    if os.environ.get("ICE_TEST_DATABASE") != args.database:
        from experiments.z2.profile_write_path import create_database
        from sqlalchemy.engine import make_url
        from src.api.config import settings as live
        try:
            url = create_database(args.database)
        except RuntimeError:                         # exists: reuse (each question wipes it)
            url = make_url(live.database_url).set(database=args.database).render_as_string(hide_password=False)
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  dict(os.environ, DATABASE_URL=url, ICE_TEST_DATABASE=args.database))
    out.mkdir(parents=True, exist_ok=True)
    failures = run(out, limit=args.limit, only=args.only, shard=shard)
    print(json.dumps(dict(failures=len(failures))))
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
