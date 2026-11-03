"""FINAL BEAM grading: every answered context scored BEAM's own way (its prompts and metrics
verbatim — experiments/beam/adapter.score: nugget mean; event ordering by normalised tau-b, as
BEAM's report counts it), non-official judge, labelled.

Durable: one row per context in <out>/beam-grades.jsonl and one cached call per judge message in
<out>/beam-grade-raw.jsonl; a rerun scores only what is missing. Incomplete answers are ungraded,
never counted wrong (Z2 D11); a reply BEAM's parser cannot read leaves its answer for the next run.
Writes <out>/beam-report.json: mean BEAM score per condition × BEAM ability.

    uv run --no-sync python experiments/beam/grade.py --units <read-pass dir> \
        --answers <answers dir> --asks logs/final-beam/1M/12/asks.json --out <dir> [--judge muse]
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.beam.adapter import score  # noqa: E402

WORKERS = 4                             # answers graded side by side, like the other graders


def contexts(units_dir, answers_dir):
    index = json.loads((Path(answers_dir) / "index.json").read_text())["index"]
    rows = {}
    for line in (Path(answers_dir) / "answers.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["prompt_sha256"]] = r
    for path in sorted(Path(units_dir).glob("units/*/*.json")):
        unit = json.loads(path.read_text())
        answer = rows.get(index.get(f"{unit['condition']}|{unit['ask']['id']}"))
        if answer is not None:
            yield unit, answer


POLICY = "ice-final-beam-grade-2"       # 2: BEAM's own prompts and metrics (was a paraphrase; Z2 F27)


def run(units_dir, answers_dir, asks_path, out, judge, *, say=print):
    """judge(messages) -> reply text (a durable call)."""
    asks = {a["id"]: a for a in json.loads(Path(asks_path).read_text())}
    out.mkdir(parents=True, exist_ok=True)
    store = out / "beam-grades.jsonl"
    done = {}
    if store.exists():
        for line in store.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    todo = [(f"{unit['condition']}|{unit['ask']['id']}", unit, answer, asks.get(unit["ask"]["id"]))
            for unit, answer in contexts(units_dir, answers_dir)]

    def one(job):
        key, unit, answer, ask = job
        if answer.get("finish") != "complete":
            return dict(key=key, condition=unit["condition"], type=ask["type"], status="ungraded",
                        reason=answer.get("finish"))
        try:
            return dict(key=key, condition=unit["condition"], type=ask["type"], status="graded",
                        **score(ask, answer["text"], judge))
        except (ValueError, KeyError, TypeError) as exc:              # unreadable reply: retried next run
            return key, f"{type(exc).__name__}: {exc}"[:200]

    (out / "meter.json").write_text(json.dumps(dict(answers=len([j for j in todo if j[3] is not None])), indent=1) + "\n")
    # answers side by side (one at a time, FINAL's ~1,600 BEAM answers were hours); one writer
    unread = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for row in pool.map(one, [j for j in todo if j[0] not in done and j[3] is not None]):
            if isinstance(row, tuple):
                unread += 1
                say(f"WARNING: BEAM judge reply unreadable for {row[0]} ({row[1]}); left for the next run")
                continue
            with store.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[row["key"]] = row
    report = {}
    for row in done.values():
        cell = report.setdefault(row["condition"], {}).setdefault(row["type"], dict(n=0, graded=0, sum=0.0))
        cell["n"] += 1
        if row["status"] == "graded":
            cell["graded"] += 1
            cell["sum"] += row["beam_score"]
    for cond in report.values():
        for cell in cond.values():
            cell["mean_score"] = round(cell.pop("sum") / cell["graded"], 4) if cell["graded"] else None
    (out / "beam-report.json").write_text(json.dumps(dict(policy=POLICY, label="BEAM's own protocol, non-official judge",
                                                          by_condition_and_type=report), indent=1) + "\n")
    (out / "status.json").write_text(json.dumps(dict(pending=unread, finished=not unread)) + "\n")
    return unread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--asks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge", default="muse")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("grades stay under logs/")
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2 import ledger_merge as M
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    args.out.mkdir(parents=True, exist_ok=True)
    generate, meta = generator(args.judge).generate, metadata(args.judge)
    calls = M.Calls(args.out / "beam-grade-raw.jsonl")
    session = f"ice-final-beam-{hashlib.sha256(str(args.out.resolve()).encode()).hexdigest()[:16]}"
    judge = message_judge(calls, generate, meta, session)
    unread = run(args.units, args.answers, args.asks, args.out, judge)
    print(json.dumps(dict(pending=unread)))
    return 3 if unread else 0


def message_judge(calls, generate, meta, session):
    """judge(messages) -> reply text: BEAM's messages sent as they are (no system prompt of ours),
    temperature 0, one durable call per distinct message list."""
    def judge(messages):
        identity = hashlib.sha256(json.dumps(dict(policy=POLICY, meta=meta, messages=messages),
                                             sort_keys=True).encode()).hexdigest()

        def call():
            result = generate(messages, temperature=0, max_output_tokens=2000, session_id=session)
            return result.text, result.usage
        return calls.get(identity, call)["text"]
    return judge


if __name__ == "__main__":
    raise SystemExit(main())
