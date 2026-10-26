"""FINAL synthetic set: questions-to-be and their answer keys, computed from a plan.

Every occurrence is in Z2's format (experiments/z2/questions.py) so the read passes and
the grader take it unchanged; the question text is phrased later by Z2's question
machinery from `ask` (what to ask about). Scheduling follows Z2 D4: first ask once the
relevant turn has left the recent window (RECENT_TURNS); a changing fact is asked before
AND after its change; a repeat >= REPEAT_GAP later; S4 asks at ~100/300/600/1,000 turns
after planting. A few asks fall BEFORE a fact is first said: the correct answer is that it
has not been mentioned (expected_abstention). Caps keep each conversation's count near its
brief; the sample is a deterministic hash, chosen before any result.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.synth.plans import BRIEFS, PLANS  # noqa: E402

RECENT_TURNS, REPEAT_GAP = 40, 150
RETENTION = (100, 300, 600, 1000)
CAPS = {"S1": 120, "S2": 90, "S3": 60, "S4": 150, "S5": 100, "S6": 80, "S7": 120, "S8": 120, "S9": 260}
EARLY_ABSTAIN_SHARE = 0.10
POLICY = "ice-final-synth-keys-1"


def checkpoints(plan):
    out, turn = [], 0
    for s in plan["sessions"]:
        turn += s["pairs"]
        out.append(turn)
    return out


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, (POLICY,) + parts)).encode()).hexdigest(), 16)


def kind_of(fact):
    if fact["status"] in ("suggested", "rejected"):
        return "attribution"
    if fact["status"] == "hypothetical":
        return "hypothetical"
    if fact["status"] in ("preference", "instruction"):
        return "preference"
    return "update" if len(fact["values"]) > 1 else "fact_recall"


def value_at(fact, turn):
    said = [v for v in fact["values"] if v["turn"] <= turn]
    return (said[-1] if said else None), [v["value"] for v in said[:-1]]


def first_after(turn, points, gap=RECENT_TURNS):
    return next((c for c in points if c - turn >= gap), None)


def ask_points(fact, points, plan_id):
    intro = fact["values"][0]["turn"]
    if plan_id == "S4" and len(fact["values"]) == 1 and not fact.get("mentions"):
        asks = []
        for d in RETENTION:
            c = next((c for c in points if c >= intro + d), None)
            if c is not None and c not in asks:
                asks.append(c)
        return asks
    first = first_after(intro, points)
    if first is None:
        return []
    asks = [first]
    for change in fact["values"][1:]:
        before = [c for c in points if asks[-1] < c < change["turn"]]
        after = first_after(change["turn"], points)
        if before and before[-1] not in asks:
            asks.append(before[-1])
        if after is not None and after not in asks:
            asks.append(after)
    later = [c for c in points if c - asks[-1] >= REPEAT_GAP]
    if later:
        asks.append(later[0])
    return sorted(set(asks))


def reference(fact, checkpoint):
    now, old = value_at(fact, checkpoint)
    if now is None:
        return dict(required=[], superseded=[], citations=[], status=fact["status"], speaker=fact["speaker"]), True
    return dict(required=[now["value"]], superseded=old, status=fact["status"], speaker=fact["speaker"],
                citations=[dict(turn=v["turn"], role=fact["speaker"]) for v in fact["values"] if v["turn"] <= checkpoint],
                in_paste=bool(fact.get("in_paste"))), False


def occurrences(plan):
    pid, points, out = plan["id"], checkpoints(plan), []
    # A derived plan (S8 from S1) samples exactly its source's asks, so the pair differs
    # in writing style only.
    src = plan.get("derived_from", pid)
    for fact in plan["facts"]:
        fam = f"{pid}:{fact['id']}"
        asks = ask_points(fact, points, src)
        early = [c for c in points if c < fact["values"][0]["turn"]]
        if early and _h(f"{src}:{fact['id']}", "early") % 1000 < EARLY_ABSTAIN_SHARE * 1000:
            asks = [early[-1]] + asks
        for i, cp in enumerate(asks):
            ref, abstain = reference(fact, cp)
            out.append(dict(family=fam, type=kind_of(fact), checkpoint_turn=cp, repeat_index=i, question=None,
                            ask=dict(about=fact["about"], status=fact["status"], project=fact.get("project"),
                                     component=fact.get("component"), topic=fact.get("topic"),
                                     probe=fact.get("probe")),
                            reference=ref, expected_abstention=abstain))
    for event in plan.get("events") or []:
        fam = f"{pid}:{event['id']}"
        cp = first_after(event["told_turn"], points)
        if cp is not None:
            out.append(dict(family=fam, type="temporal", checkpoint_turn=cp, repeat_index=0, question=None,
                            ask=dict(about=event["what"], component=event.get("component"), probe=event.get("probe")),
                            reference=dict(required=[f"{event['what']} on {event['date']}"], superseded=[],
                                           citations=[dict(turn=event["told_turn"], role="user")],
                                           date=event["date"]), expected_abstention=False))
    cap = CAPS.get(pid)
    rank = lambda o: _h(o["family"].replace(pid, src, 1), o["checkpoint_turn"])
    if cap and len(out) > cap:
        if any(o["ask"].get("component") for o in out):
            # The ablation conversation: an equal share per component group, so every
            # leave-one-out condition has questions it should hurt.
            groups = {}
            for o in out:
                groups.setdefault(o["ask"].get("component"), []).append(o)
            share = cap // len(groups)
            keep = [o for g in groups.values() for o in sorted(g, key=rank)[:share]]
        else:
            keep = sorted(out, key=rank)[:cap]
        out = [o for o in out if o in keep]
    return sorted(out, key=lambda o: (o["checkpoint_turn"], o["family"]))


def main() -> int:
    from experiments.synth.plans import validate
    for pid in BRIEFS:
        path = PLANS / f"{pid}.json"
        if not path.exists():
            continue
        plan = json.loads(path.read_text())
        if validate(plan):
            print(f"{pid}: plan does not validate; no keys")
            continue
        occ = occurrences(plan)
        (PLANS / f"{pid}.occurrences.json").write_text(json.dumps(occ, indent=1, ensure_ascii=False) + "\n")
        kinds = {}
        for o in occ:
            kinds[o["type"]] = kinds.get(o["type"], 0) + 1
        print(f"{pid}: {len(occ)} occurrences {kinds}, abstentions {sum(o['expected_abstention'] for o in occ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
