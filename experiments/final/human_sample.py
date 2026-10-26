"""FINAL judge validity against a person (Z3 D11 item 2).

`sheet`: ~100 graded answers drawn across histories and judge verdicts by a fixed hash, written
as a blind sheet (question, reference, answer — no condition name, no judge verdict) to
<run>/human-sample/sheet.json and sheet.md. The maintainer fills `human_grade` in sheet.json
with correct | partial | incorrect. `agree`: agreement with the judge — exact rate, Cohen's
kappa, and the confusion table — into agreement.json.

    uv run --no-sync python experiments/final/human_sample.py sheet --run logs/final-run-main
    uv run --no-sync python experiments/final/human_sample.py agree --run logs/final-run-main
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GRADES = ("correct", "partial", "incorrect")
POLICY = "ice-final-human-sample-1"


def collect(run):
    """[(history, key, judge grade, question, reference, answer)] from every grade-<history> output."""
    rows = []
    for grades_path in sorted(Path(run).glob("grade-*/grades.json")):
        history = grades_path.parent.name.removeprefix("grade-")
        g = json.loads(grades_path.read_text())
        read, ans = Path(run) / f"local-read-{history}", Path(run) / f"cloud-answers-{history}"
        if not (ans / "index.json").exists():
            continue
        index = json.loads((ans / "index.json").read_text())["index"]
        texts = {}
        for line in (ans / "answers.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                texts[r["prompt_sha256"]] = r.get("text", "")
        for key, r in g.get("graded", {}).items():
            if r.get("status") != "graded" or r.get("grade") not in GRADES:
                continue
            cond, ask_id = key.split("|", 1)
            unit = read / "units" / cond / f"{ask_id}.json"
            if not unit.exists():
                continue
            u = json.loads(unit.read_text())
            rows.append(dict(history=history, key=key, judge=r["grade"], question=u["ask"].get("question"),
                             reference=r.get("reference"), answer=texts.get(index.get(key), "")))
    return rows


def draw(rows, n=100):
    """Stratified by judge verdict, then by a fixed hash — the same rows on every rerun."""
    by = {g: sorted([r for r in rows if r["judge"] == g],
                    key=lambda r: hashlib.sha256(f"{POLICY}|{r['history']}|{r['key']}".encode()).hexdigest())
          for g in GRADES}
    out, i = [], 0
    while len(out) < n and any(by.values()):
        g = GRADES[i % 3]
        if by[g]:
            out.append(by[g].pop(0))
        i += 1
    return out


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == g for a, _ in pairs) / n) * (sum(b == g for _, b in pairs) / n) for g in GRADES)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("sheet", "agree"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    out = args.run / "human-sample"
    if args.command == "sheet":
        sample = draw(collect(args.run), args.n)
        out.mkdir(parents=True, exist_ok=True)
        blind = [dict(item=i, question=r["question"], reference=r["reference"], answer=r["answer"], human_grade=None)
                 for i, r in enumerate(sample)]
        (out / "key.json").write_text(json.dumps(sample, indent=1, ensure_ascii=False) + "\n")
        (out / "sheet.json").write_text(json.dumps(blind, indent=1, ensure_ascii=False) + "\n")
        (out / "sheet.md").write_text("\n\n".join(
            f"## {b['item']}\n**Question:** {b['question']}\n\n**Reference:** {json.dumps(b['reference'], ensure_ascii=False)}"
            f"\n\n**Answer:** {b['answer']}\n\n**Your grade (correct / partial / incorrect):**" for b in blind) + "\n")
        print(f"{len(blind)} items -> {out / 'sheet.md'}; fill human_grade in sheet.json")
        return 0
    key = json.loads((out / "key.json").read_text())
    filled = {b["item"]: b["human_grade"] for b in json.loads((out / "sheet.json").read_text())}
    pairs = [(r["judge"], filled[i]) for i, r in enumerate(key) if filled.get(i) in GRADES]
    table = {j: {h: sum(1 for a, b in pairs if a == j and b == h) for h in GRADES} for j in GRADES}
    result = dict(n=len(pairs), exact=round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
                  cohen_kappa=kappa(pairs), judge_by_human=table)
    (out / "agreement.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
