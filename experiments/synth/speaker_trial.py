"""Which of the two writer models writes the simulated USER more like a person (maintainer,
2026-10-28: DeepSeek V4.1 Flash or MiMo-V2.6-Flash only; the other writes the assistant).

Both pairings generate the first TURNS turns of one plan per style family (terse, rambling,
formal, casual), in parallel, under logs/final-synth-trial/<user>__<assistant>/<plan>/. The
report counts, per pairing: turns verified on the first try, cost and seconds per call from the
provider's usage, and marks of AI writing in the user messages — em dashes, stock assistant
phrases, markdown structure — next to their mean length. The texts are for reading: the counts
only point at where to look.

    uv run --no-sync python experiments/synth/speaker_trial.py [--turns 6]
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/final-synth-trial"
PLANS = ("S1", "S2", "S3", "S5")            # terse-lowercase-typos, rambling-runon, formal, casual
PAIRINGS = (("cloud:deepseek_chat", "cloud:mimo_chat"), ("cloud:mimo_chat", "cloud:deepseek_chat"))
STOCK = re.compile(r"\b(certainly|great question|i hope this helps|absolutely|delve|thank you so much)\b", re.I)
MARKDOWN = re.compile(r"^\s*(#{1,6} |[-*] |\d+\. )|\*\*", re.M)


def folder(user, assistant, plan):
    name = f"{user.split(':')[1]}__{assistant.split(':')[1]}"
    return OUT / name / plan


def generate(user, assistant, plan, turns):
    out = folder(user, assistant, plan)
    cmd = [sys.executable, "experiments/synth/generate.py", "--plan", plan, "--out", str(out),
           "--user", user, "--assistant", assistant, "--limit", str(turns)]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return plan, done.returncode, done.stderr[-400:]


def marks(text):
    return dict(em_dash=text.count("—"), stock=len(STOCK.findall(text)), markdown=len(MARKDOWN.findall(text)))


def report(turns):
    out = {}
    for user, assistant in PAIRINGS:
        rows, usage = [], []
        for plan in PLANS:
            d = folder(user, assistant, plan)
            if (d / "turns.jsonl").exists():
                rows += [dict(json.loads(line), plan=plan) for line in (d / "turns.jsonl").read_text().splitlines()
                         if line.strip()]
            if (d / "usage.jsonl").exists():
                usage += [json.loads(line) for line in (d / "usage.jsonl").read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["turn"] <= turns]
        cost = [((u["usage"] or {}).get("opencode") or {}).get("allowance") or 0 for u in usage]
        user_marks = [marks(r["user"]) for r in rows]
        out[f"user={user} assistant={assistant}"] = dict(
            turns=len(rows), first_try=sum(r["attempts"] == 1 and r["verified"] for r in rows),
            unverified=sum(not r["verified"] for r in rows), calls=len(usage), cost=round(sum(cost), 5),
            seconds_per_call=round(statistics.mean(u["seconds"] for u in usage), 1) if usage else None,
            user_em_dashes=sum(m["em_dash"] for m in user_marks), user_stock_phrases=sum(m["stock"] for m in user_marks),
            user_markdown=sum(m["markdown"] for m in user_marks),
            user_words_by_plan={p: round(statistics.mean(len(r["user"].split()) for r in rows if r["plan"] == p))
                                for p in PLANS if any(r["plan"] == p for r in rows)})
    (OUT / "report.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--user", help="only the pairing with this user side, e.g. cloud:mimo_chat")
    args = parser.parse_args()
    jobs = [(u, a, p) for u, a in PAIRINGS for p in PLANS if not args.user or u == args.user]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for plan, code, err in pool.map(lambda j: generate(*j, args.turns), jobs):
            if code not in (0, 3):
                print(f"{plan}: exit {code}: {err}", file=sys.stderr)
    print(json.dumps(report(args.turns), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
