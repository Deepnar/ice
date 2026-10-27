"""Readable review files for generated synthetic conversations (the maintainer's quality gate):
one Markdown file per plan, each turn with what the plan planted, how many tries it took and why
every rejected try was rejected (attempts.jsonl), the user's message and the assistant's reply;
a summary on top — turns, first-try rate, stops, cost and how much of the input came from the
provider's cache.

    uv run --no-sync python experiments/synth/review_sample.py logs/final-synth-batch6 logs/review-batch6
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.synth.generate import due, session_of  # noqa: E402
from experiments.synth.plans import PLANS  # noqa: E402


def lines_of(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summary(run):
    rows, usage = lines_of(run / "turns.jsonl"), lines_of(run / "usage.jsonl")
    tokens = [u["usage"] or {} for u in usage]
    seen = sum(t.get("prompt_tokens") or t.get("input_tokens") or 0 for t in tokens)
    cached = sum(((t.get("prompt_tokens_details") or t.get("input_tokens_details") or {}).get("cached_tokens") or 0) for t in tokens)
    return dict(turns=len(rows), first_try=sum(r["attempts"] == 1 for r in rows),
                repaired=sum(bool(r.get("repaired")) for r in rows), stopped=(run / "stopped.json").exists(),
                calls=len(usage), cost=round(sum(((t.get("opencode") or {}).get("allowance") or 0) for t in tokens), 4),
                cached_share=round(cached / seen, 2) if seen else None)


def write(run, out):
    plan = json.loads((PLANS / f"{run.name}.json").read_text())
    schedule, when = due(plan), session_of(plan)
    rejected = defaultdict(list)
    for a in lines_of(run / "attempts.jsonl"):
        rejected[a["turn"]].append(a)
    s = summary(run)
    text = [f"# {run.name} — {s['turns']} turns", "",
            f"First try {s['first_try']}/{s['turns']} · stopped: {s['stopped']} · repaired by hand: {s['repaired']} · "
            f"{s['calls']} calls, ${s['cost']} · input from cache: {s['cached_share']}", "",
            f"Persona: {plan['persona'].get('name')} — {plan['persona'].get('situation', '')}", ""]
    for r in lines_of(run / "turns.jsonl"):
        session, _ = when[r["turn"]]
        items = schedule.get(r["turn"], {})
        planted = [f"{i['about']} = {i['value']} ({i['change']})" for i in items.get("user", [])]
        suggest = [f"{i['about']} = {i['value']}" for i in items.get("assistant", [])]
        text += [f"## Turn {r['turn']} — session {session['index']}, style: {session['style']}",
                 f"*Planted:* {'; '.join(planted) or 'nothing'}" + (f" · *assistant must suggest:* {'; '.join(suggest)}" if suggest else ""),
                 f"*Tries:* {r['attempts']}" + (" (repaired by hand)" if r.get("repaired") else "")]
        for a in rejected.get(r["turn"], []):
            text.append(f"  - try {a['attempt']} rejected at {a['stage']}: {'; '.join(a['problems'])[:300]}")
        text += ["", "**User:**", "", r["user"], "", "**Assistant:**", "", r["assistant"], "", "---", ""]
    if (run / "stopped.json").exists():
        stop = json.loads((run / "stopped.json").read_text())
        text += [f"## STOPPED at turn {stop['turn']} — {stop['problems']}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run.name}.md").write_text("\n".join(text) + "\n")
    return s


def main() -> int:
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    total = {}
    for run in sorted(d for d in runs.iterdir() if d.is_dir() and (d / "turns.jsonl").exists()):
        total[run.name] = write(run, out)
    print(json.dumps(total, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
