"""FINAL synthetic set: the user's question text for each planned ask, and the final asks.

A phrasing file <ID>.questions.json (written by a fast model from <ID>.occurrences.json and the
plan) holds one question per occurrence, in the conversation's writing style, plus 8-12
NEVER-SAID questions: plausible things this person might ask about that the plan never
mentions — certain abstentions, since the plan lists everything that is said.

    {"questions": {"<family>@<checkpoint>#<repeat>": "text", ...},
     "negatives": [{"question": "text", "about": "short subject"}, ...]}

check() names every problem; asks() merges plan keys and text into Z2-format asks
(<ID>.asks.json) for the read passes. Negatives are asked at the last checkpoint.

    uv run --no-sync python experiments/synth/phrase.py           # check all, write asks
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.synth.keys import checkpoints  # noqa: E402
from experiments.synth.plans import PLANS  # noqa: E402

MAX_WORDS, NEGATIVES = (3, 60), (8, 12)


def key(o):
    return f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}"


def norm(text):
    return re.sub(r"[\s\-_,.'\"]+", " ", (text or "").lower()).strip()


def check(plan, occurrences, phrasing):
    p, qs = [], phrasing.get("questions") or {}
    want = {key(o): o for o in occurrences}
    missing, extra = set(want) - set(qs), set(qs) - set(want)
    if missing:
        p.append(f"{len(missing)} asks without a question, e.g. {sorted(missing)[:3]}")
    if extra:
        p.append(f"{len(extra)} questions for unknown asks, e.g. {sorted(extra)[:3]}")
    for k, text in qs.items():
        o = want.get(k)
        n = len(str(text).split())
        if not MAX_WORDS[0] <= n <= MAX_WORDS[1]:
            p.append(f"{k}: {n} words")
        if o:
            for value in o["reference"]["required"]:
                if len(norm(value)) >= 3 and norm(value) in norm(text):
                    p.append(f"{k}: the question contains its answer ({value!r})")
    negs = phrasing.get("negatives") or []
    if not NEGATIVES[0] <= len(negs) <= NEGATIVES[1]:
        p.append(f"{len(negs)} never-said questions, want {NEGATIVES[0]}-{NEGATIVES[1]}")
    said = norm(json.dumps([f["about"] for f in plan["facts"]] + [f["value"] for f in plan["facts"] for f in f["values"]]
                           + [e["what"] for e in plan.get("events") or []], ensure_ascii=False))
    for i, n in enumerate(negs):
        if not str(n.get("question", "")).strip() or not str(n.get("about", "")).strip():
            p.append(f"negative {i}: empty")
        elif norm(n["about"]) in said:
            p.append(f"negative {i}: {n['about']!r} is something the plan does say")
    return p


def asks(plan, occurrences, phrasing):
    out = []
    for o in occurrences:
        out.append(dict(id=key(o), kind="question", type=o["type"], question=phrasing["questions"][key(o)],
                        checkpoint_turn=o["checkpoint_turn"], family=o["family"], component=o["ask"].get("component"),
                        expected_abstention=o["expected_abstention"]))
    last = checkpoints(plan)[-1]
    for i, n in enumerate(phrasing.get("negatives") or []):
        out.append(dict(id=f"{plan['id']}:NEG{i:02d}@{last}#0", kind="question", type="negative",
                        question=n["question"], checkpoint_turn=last, family=f"{plan['id']}:NEG{i:02d}",
                        component=None, expected_abstention=True))
    return out


def final_occurrences(plan, occurrences, phrasing):
    """Occurrences with their question text, plus the never-said negatives — the
    grader's reference file for this conversation (grade.py --occurrences)."""
    out = [dict(o, question=phrasing["questions"][key(o)]) for o in occurrences]
    last = checkpoints(plan)[-1]
    for i, n in enumerate(phrasing.get("negatives") or []):
        out.append(dict(family=f"{plan['id']}:NEG{i:02d}", type="negative", checkpoint_turn=last, repeat_index=0,
                        question=n["question"], ask=dict(about=n["about"]),
                        reference=dict(required=[], superseded=[], citations=[]), expected_abstention=True))
    return out


def plan_ledger(plan):
    """The plan as a Z2-shaped ledger (one entry per planted value, each change superseding
    the previous) so the grader's invented-content check works on synthetic answers."""
    entries = []
    for f in plan["facts"]:
        prev = None
        for i, v in enumerate(f["values"]):
            eid = f"{plan['id']}:{f['id']}.{i}"
            entries.append(dict(id=eid, members=[eid], supersedes_raw=[prev] if prev else [],
                                value=f"{f['about']}: {v['value']}", speaker=f["speaker"], kind=f["status"],
                                valid_from_turn=v["turn"], usable_for_questions=True,
                                citations=[dict(turn=v["turn"], role=f["speaker"])]))
            prev = eid
    for e in plan.get("events") or []:
        eid = f"{plan['id']}:{e['id']}"
        entries.append(dict(id=eid, members=[eid], supersedes_raw=[], value=f"{e['what']} on {e['date']}",
                            speaker="user", kind="event", valid_from_turn=e["told_turn"], usable_for_questions=True,
                            citations=[dict(turn=e["told_turn"], role="user")]))
    return dict(policy="ice-final-synth-plan-ledger-1", history=plan["id"], entries=entries)


def main() -> int:
    bad = 0
    for path in sorted(PLANS.glob("S*.questions.json")):
        pid = path.name.split(".")[0]
        plan = json.loads((PLANS / f"{pid}.json").read_text())
        occ = json.loads((PLANS / f"{pid}.occurrences.json").read_text())
        phrasing = json.loads(path.read_text())
        problems = check(plan, occ, phrasing)
        bad += bool(problems)
        print(f"{pid}: {'OK' if not problems else f'{len(problems)} problems'}")
        for x in problems[:20]:
            print("   -", x)
        if not problems:
            (PLANS / f"{pid}.asks.json").write_text(json.dumps(asks(plan, occ, phrasing), indent=1, ensure_ascii=False) + "\n")
            (PLANS / f"{pid}.final-occurrences.json").write_text(
                json.dumps(final_occurrences(plan, occ, phrasing), indent=1, ensure_ascii=False) + "\n")
            (PLANS / f"{pid}.ledger.json").write_text(json.dumps(plan_ledger(plan), indent=1, ensure_ascii=False) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
