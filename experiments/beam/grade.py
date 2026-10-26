"""FINAL BEAM grading: every answered context scored BEAM's way (nugget mean, Kendall tau-b
for event ordering — experiments/beam/adapter.score), non-official judge, labelled.

Durable: one row per context in <out>/beam-grades.jsonl; a rerun scores only what is
missing. Incomplete answers are ungraded, never counted wrong (Z2 D11). Writes
<out>/beam-report.json: mean nugget score per condition × BEAM ability.

    uv run --no-sync python experiments/beam/grade.py --units <read-pass dir> \
        --answers <answers dir> --asks logs/final-beam/1M/12/asks.json --out <dir> [--judge muse]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.beam.adapter import score  # noqa: E402


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


def run(units_dir, answers_dir, asks_path, out, judge):
    asks = {a["id"]: a for a in json.loads(Path(asks_path).read_text())}
    out.mkdir(parents=True, exist_ok=True)
    store = out / "beam-grades.jsonl"
    done = {}
    if store.exists():
        for line in store.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    for unit, answer in contexts(units_dir, answers_dir):
        key = f"{unit['condition']}|{unit['ask']['id']}"
        ask = asks.get(unit["ask"]["id"])
        if key in done or ask is None:
            continue
        if answer.get("finish") != "complete":
            row = dict(key=key, condition=unit["condition"], type=ask["type"], status="ungraded",
                       reason=answer.get("finish"))
        else:
            row = dict(key=key, condition=unit["condition"], type=ask["type"], status="graded",
                       **score(ask, answer["text"], judge))
        with store.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[key] = row
    report = {}
    for row in done.values():
        cell = report.setdefault(row["condition"], {}).setdefault(row["type"], dict(n=0, graded=0, sum=0.0))
        cell["n"] += 1
        if row["status"] == "graded":
            cell["graded"] += 1
            cell["sum"] += row["nugget_mean"]
    for cond in report.values():
        for cell in cond.values():
            cell["mean_nugget"] = round(cell.pop("sum") / cell["graded"], 4) if cell["graded"] else None
    (out / "beam-report.json").write_text(json.dumps(dict(label="BEAM nugget protocol, non-official judge",
                                                          by_condition_and_type=report), indent=1) + "\n")
    return report


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
    from experiments.z2.grade import judge_caller
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    args.out.mkdir(parents=True, exist_ok=True)
    judge = judge_caller(M.Calls(args.out / "beam-grade-raw.jsonl"), generator(args.judge).generate,
                         metadata(args.judge), f"beam|{args.asks}", role=args.judge)
    print(json.dumps(run(args.units, args.answers, args.asks, args.out, judge))[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
