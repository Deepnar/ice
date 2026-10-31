"""R10 (Z2 F4): what the reranker rescue does on LongMemEval questions the main decision refused —
a code check on saved read units (oracle.py --rescue-read), no model and no answers.

Per question the gate refused in this run: does the answer reach the prompt under the shipped rescue
("full") and under the reranker rescue at each bar? The rescue read ran with its bar off and records
its best score, so a bar b admits exactly the questions whose best score is >= b. "Reached" = the
evidence turn's own words or the reference answer's text is in the prompt (ICE often carries the
answer as a fact or a summary, not the original words). Abstention questions are counted apart: their
"answer" is that the thing was never said.

    uv run --no-sync python experiments/z2/r10_report.py --units logs/z2-run-main/s12-r10/units
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BARS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def reached(stage, snippets, answer):
    text = "\n".join(m["content"] for m in stage.get("prompt_messages") or []).lower()
    return any(s.lower() in text for s in snippets) or bool(answer and answer.lower() in text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--dataset", default="oracle")
    args = parser.parse_args()
    from experiments.z2 import oracle as O
    O.select_dataset(args.dataset)
    data = {i["question_id"]: i for i in O.load_instances()}
    rows = []
    for path in sorted((args.units / O.RESCUE_CONDITION[0]).glob("*.json")):
        rescue = json.loads(path.read_text())
        full_path = args.units / "full" / path.name
        if not full_path.exists():
            continue
        full = json.loads(full_path.read_text())["stage"]
        q = rescue["ask"]["id"]
        inst, gate = data[q], rescue["stage"].get("gate") or {}
        plan = O.layout(inst)
        turns = [r for e in plan["evidence"] for r in plan["conversations"][e["slug"]] if r["turn_number"] == e["turn"]]
        snippets = [s[:80] for r in turns for s in (r["prompt"], r["response"]) if len(s) > 30]
        answer = None if q.endswith("_abs") else str(inst["answer"]).strip()
        answer = answer if answer and len(answer) >= 3 else None
        rows.append(dict(q=q, type=inst["question_type"], abstention=q.endswith("_abs"),
                         refused=gate.get("b2_retrieve") is False,
                         score=gate.get("rescue_best_score"), shipped=reached(full, snippets, answer),
                         rescue=reached(rescue["stage"], snippets, answer), marked=bool(snippets or answer)))
    refused = [r for r in rows if r["refused"] and not r["abstention"] and r["marked"]]
    abst = [r for r in rows if r["refused"] and r["abstention"]]
    print(f"{len(rows)} questions read; the gate refused {sum(r['refused'] for r in rows)} "
          f"({len(refused)} answerable with a marked answer, {len(abst)} abstention)")
    if not refused:
        return 0
    unscored = sum(r["score"] is None for r in refused)
    print(f"answer in the prompt, of {len(refused)} answerable refused questions:")
    print(f"   shipped rescue (proof)          {sum(r['shipped'] for r in refused):4}")
    print(f"   reranker rescue, bar off        {sum(r['rescue'] for r in refused):4}   (best score missing on {unscored})")
    for bar in BARS:
        admitted = [r for r in refused if r["score"] is not None and r["score"] >= bar]
        ab = sum(r["score"] is not None and r["score"] >= bar for r in abst)
        print(f"   reranker rescue, bar {bar:+.0f}        {sum(r['rescue'] for r in admitted):4}   "
              f"admitted {len(admitted):3}/{len(refused)}   abstention admitted {ab}/{len(abst)}")
    print("by question type, at bar +0 (answer in the prompt: shipped / reranker, of refused):")
    for kind in sorted({r["type"] for r in refused}):
        group = [r for r in refused if r["type"] == kind]
        new = sum(r["rescue"] and r["score"] is not None and r["score"] >= 0.0 for r in group)
        print(f"   {kind:28} {sum(r['shipped'] for r in group):3} / {new:3}   of {len(group)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
