"""FINAL synthetic LSREP set: the plan format, the per-conversation briefs, and the
validator (FINAL_experiments.md §2.1, decided 2026-10-26).

A plan is the rough flow of one conversation, written before any text is generated:
persona, sessions (date, size, writing style, tag mix, beats) and every planted fact with
the turns where it is introduced, changed, corrected or merely re-mentioned. The answer
key is computed from the plan, so a plan that is internally impossible would corrupt the
key silently — every plan passes `validate` before generation.

    uv run --no-sync python experiments/synth/plans.py            # validate all plans
    uv run --no-sync python experiments/synth/plans.py --twins    # also derive S8 from a valid S1
"""

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLANS = HERE / "plans"
ROOT = HERE.parents[1]

STYLES = {
    "terse-lowercase-typos": "short lowercase messages, missing punctuation, common typos and chat slang (u, ur, idk, pls)",
    "rambling-runon": "long run-on messages, several thoughts per message, filler (like, ok so), self-corrections mid-sentence",
    "formal": "complete sentences, correct capitalisation and punctuation, structured requests",
    "casual": "relaxed everyday writing, mostly lowercase, some slang, medium length",
    "polished": "carefully written, well organised, assistant-like clarity (the BEAM convention)",
}
STATUSES = {"decided", "suggested", "hypothetical", "rejected", "preference", "instruction"}
SPEAKER_OF = {"decided": "user", "hypothetical": "user", "preference": "user", "instruction": "user",
              "suggested": "assistant", "rejected": "assistant"}
CHANGES = {"introduce", "update", "correction", "reversal"}
COMPONENTS = {"bm25", "rerank", "codex", "summaries", "notes", "procedural", "gate", "clusters",
              "temporal", "maintenance"}


def labels():
    schema = json.loads((ROOT / "data/labeled/label_schema.json").read_text())
    return {h["name"]: {l["name"] for l in h["labels"]} for h in schema["heads"]}


BRIEFS = {
    "S1": dict(pairs=400, target="changing decisions: values updated, corrected and reversed; answers are asked as of each checkpoint",
               styles=["terse-lowercase-typos"], topics=["Admin_&_Productivity", "Business_&_Finance"],
               intents=["Strategic_Planning", "Decision_Making"], contexts=["Needs_Memory", "Temporal_Recall"],
               quotas=dict(facts=40, changed_share=0.6, corrections=10)),
    "S2": dict(pairs=300, target="who said what: assistant suggestions vs user decisions, hypotheticals, rejected options",
               styles=["rambling-runon"], topics=["Business_&_Finance", "Creative_&_Media"],
               intents=["Ideation", "Decision_Making"], contexts=["Needs_Memory"],
               quotas=dict(facts=40, suggested_or_rejected_share=0.35, hypothetical=10)),
    "S3": dict(pairs=200, target="pasted material: long documents, logs and code; facts inside pastes are not things the user said",
               styles=["formal"], topics=["Software_&_Tech", "STEM_&_Academics"],
               intents=["Troubleshooting", "Analysis_&_Summarization", "Code_Change", "Utility_Formatting"],
               contexts=["Needs_Memory", "High_Complexity"], quotas=dict(facts=30, in_paste_share=0.4)),
    "S4": dict(pairs=1200, target="long retention: facts planted early, never repeated, asked 100/300/600/1,000 turns later",
               styles=["terse-lowercase-typos", "rambling-runon", "formal", "casual"],
               topics=["Lifestyle_&_Health", "Social_&_Relationships", "Creative_&_Media"],
               intents=["Open_Exploration", "Factual_Retrieval", "Generation", "Emotional_Processing"],
               contexts=["Needs_Memory"], quotas=dict(facts=80, early_untouched=60, early_until=250)),
    "S5": dict(pairs=500, target="time: real calendar gaps, relative dates, durations, event order across sessions",
               styles=["casual"], topics=["Lifestyle_&_Health", "Admin_&_Productivity"],
               intents=["Strategic_Planning", "Factual_Retrieval"], contexts=["Temporal_Recall"],
               quotas=dict(facts=20, events=30, span_days=120)),
    "S6": dict(pairs=300, target="standing preferences and instructions that must persist; some changed later",
               styles=["terse-lowercase-typos"], topics=["Lifestyle_&_Health", "General_Reference_&_Trivia"],
               intents=["Generation", "Utility_Formatting"], contexts=["Needs_Memory"],
               quotas=dict(facts=25, preference_or_instruction=15, changed_preferences=5)),
    "S7": dict(pairs=600, target="topic hopping: four interleaved projects with same-shaped facts (scoping, distractors)",
               styles=["casual"], topics=["Software_&_Tech", "Creative_&_Media", "Business_&_Finance", "Lifestyle_&_Health"],
               intents=["Analysis_&_Summarization", "Generation", "Decision_Making", "Troubleshooting"],
               contexts=["Needs_Memory"], quotas=dict(facts=60, per_project=12, twins=10)),
    "S8": dict(pairs=400, target="style twin of S1: the same plan written formally (derived from S1 by code)",
               styles=["polished"], topics=["Admin_&_Productivity", "Business_&_Finance"],
               intents=["Strategic_Planning", "Decision_Making"], contexts=["Needs_Memory", "Temporal_Recall"],
               quotas=dict(facts=40, changed_share=0.6, corrections=10), derived_from="S1"),
    "S9": dict(pairs=1000, target="ablation conversation: every component group has facts that need it",
               styles=["terse-lowercase-typos", "rambling-runon", "formal", "casual"],
               topics=["Software_&_Tech", "Lifestyle_&_Health", "Business_&_Finance", "Creative_&_Media"],
               intents=["Factual_Retrieval", "Strategic_Planning", "Generation", "Troubleshooting"],
               contexts=["Needs_Memory", "Temporal_Recall"], quotas=dict(facts=200, per_component=20)),
}


def turn_dates(plan):
    """Turn number -> session date."""
    out, turn = {}, 0
    for s in plan["sessions"]:
        for _ in range(s["pairs"]):
            turn += 1
            out[turn] = date.fromisoformat(s["date"])
    return out


def validate(plan, brief=None, tags=None):
    """A list of problems; empty means the plan can be generated."""
    brief = brief or BRIEFS[plan["id"]]
    tags = tags or labels()
    p, pairs = [], brief["pairs"]
    sessions = plan.get("sessions") or []
    if [s.get("index") for s in sessions] != list(range(1, len(sessions) + 1)):
        p.append("sessions must be indexed 1..n in order")
    if sum(s.get("pairs", 0) for s in sessions) != pairs:
        p.append(f"session pairs sum to {sum(s.get('pairs', 0) for s in sessions)}, brief says {pairs}")
    try:
        days = [date.fromisoformat(s["date"]) for s in sessions]
        if any(b < a for a, b in zip(days, days[1:])):
            p.append("session dates go backwards")
    except (KeyError, ValueError) as exc:
        p.append(f"bad session date: {exc}")
        days = []
    for s in sessions:
        if s.get("style") not in STYLES:
            p.append(f"session {s.get('index')}: unknown style {s.get('style')!r}")
        if not set(s.get("topics", [])) <= tags["topic"] or not set(s.get("intents", [])) <= tags["intent"]:
            p.append(f"session {s.get('index')}: topic/intent not in the label schema")
        if not s.get("beats"):
            p.append(f"session {s.get('index')}: no beats")
    facts = plan.get("facts") or []
    ids = [f.get("id") for f in facts] + [e.get("id") for e in plan.get("events") or []]
    if len(ids) != len(set(ids)):
        p.append("duplicate fact/event ids")
    for f in facts:
        fid, vals = f.get("id"), f.get("values") or []
        if f.get("status") not in STATUSES:
            p.append(f"{fid}: unknown status {f.get('status')!r}")
        elif f.get("speaker") != SPEAKER_OF[f["status"]]:
            p.append(f"{fid}: a {f['status']} fact is said by the {SPEAKER_OF[f['status']]}")
        if not vals or vals[0].get("change") != "introduce" or any(v.get("change") == "introduce" for v in vals[1:]):
            p.append(f"{fid}: must start with one introduce, then only update/correction/reversal")
        if any(v.get("change") not in CHANGES for v in vals):
            p.append(f"{fid}: unknown change kind")
        turns = [v.get("turn") for v in vals]
        if any(not isinstance(t, int) or not 1 <= t <= pairs for t in turns) or turns != sorted(set(turns)):
            p.append(f"{fid}: value turns must be increasing integers in 1..{pairs}")
        for m in f.get("mentions") or []:
            if not isinstance(m, int) or not 1 <= m <= pairs or (turns and m <= turns[0]) or m in turns:
                p.append(f"{fid}: mention {m} must come after the introduce and not on a change turn")
        if f.get("component") is not None and f["component"] not in COMPONENTS:
            p.append(f"{fid}: unknown component {f['component']!r}")
        if not str(f.get("about", "")).strip() or any(not str(v.get("value", "")).strip() for v in vals):
            p.append(f"{fid}: empty subject or value")
    dated = turn_dates(plan) if not [x for x in p if "session" in x] else {}
    for e in plan.get("events") or []:
        t = e.get("told_turn")
        if not isinstance(t, int) or not 1 <= t <= pairs:
            p.append(f"{e.get('id')}: told_turn out of range")
            continue
        try:
            when = date.fromisoformat(e["date"])
        except (KeyError, ValueError):
            p.append(f"{e.get('id')}: bad event date")
            continue
        if dated and abs((dated[t] - when).days) > 120:
            p.append(f"{e.get('id')}: event is {abs((dated[t] - when).days)} days from the session that tells it")
        if not str(e.get("relative_phrase", "")).strip():
            p.append(f"{e.get('id')}: no relative phrase")
    quiet = plan.get("no_memory_turns") or []
    if not 0.07 * pairs <= len(set(quiet)) <= 0.14 * pairs or any(not 1 <= t <= pairs for t in quiet):
        p.append(f"no_memory_turns: {len(set(quiet))}, want ~10% of {pairs} within range")
    p += quota_problems(plan, brief, days)
    return p


def quota_problems(plan, brief, days):
    q, facts, p = brief["quotas"], plan.get("facts") or [], []
    n = len(facts)
    changed = [f for f in facts if len(f.get("values") or []) >= 2]
    if n < q.get("facts", 0):
        p.append(f"{n} facts, brief wants >= {q['facts']}")
    if "changed_share" in q and n and len(changed) / n < q["changed_share"]:
        p.append(f"{len(changed)}/{n} facts change, want >= {q['changed_share']:.0%}")
    if "corrections" in q:
        k = sum(1 for f in facts for v in f.get("values") or [] if v.get("change") in ("correction", "reversal"))
        if k < q["corrections"]:
            p.append(f"{k} corrections/reversals, want >= {q['corrections']}")
    if "suggested_or_rejected_share" in q and n:
        k = sum(f.get("status") in ("suggested", "rejected") for f in facts)
        if k / n < q["suggested_or_rejected_share"]:
            p.append(f"{k}/{n} suggested/rejected, want >= {q['suggested_or_rejected_share']:.0%}")
    if "hypothetical" in q and sum(f.get("status") == "hypothetical" for f in facts) < q["hypothetical"]:
        p.append(f"fewer than {q['hypothetical']} hypothetical facts")
    if "in_paste_share" in q and n and sum(bool(f.get("in_paste")) for f in facts) / n < q["in_paste_share"]:
        p.append(f"fewer than {q['in_paste_share']:.0%} of facts inside pastes")
    if "early_untouched" in q:
        k = sum(1 for f in facts if len(f.get("values") or []) == 1 and not f.get("mentions")
                and (f["values"][0].get("turn") or 10**9) <= q["early_until"])
        if k < q["early_untouched"]:
            p.append(f"{k} facts planted by turn {q['early_until']} and never touched again, want >= {q['early_untouched']}")
    if "events" in q and len(plan.get("events") or []) < q["events"]:
        p.append(f"{len(plan.get('events') or [])} events, want >= {q['events']}")
    if "span_days" in q and days and (days[-1] - days[0]).days < q["span_days"]:
        p.append(f"sessions span {(days[-1] - days[0]).days} days, want >= {q['span_days']}")
    if "preference_or_instruction" in q:
        pref = [f for f in facts if f.get("status") in ("preference", "instruction")]
        if len(pref) < q["preference_or_instruction"]:
            p.append(f"{len(pref)} preferences/instructions, want >= {q['preference_or_instruction']}")
        if sum(len(f.get("values") or []) >= 2 for f in pref) < q.get("changed_preferences", 0):
            p.append(f"fewer than {q['changed_preferences']} preferences changed later")
    if "per_project" in q:
        by = {}
        for f in facts:
            by[f.get("project")] = by.get(f.get("project"), 0) + 1
        if len([k for k, v in by.items() if k and v >= q["per_project"]]) < 4:
            p.append(f"need 4 projects with >= {q['per_project']} facts each (field 'project'); got {by}")
        if sum(bool(f.get("twin_of")) for f in facts) < q["twins"]:
            p.append(f"fewer than {q['twins']} same-shaped twins across projects (field 'twin_of')")
    if "per_component" in q:
        by = {c: 0 for c in COMPONENTS}
        for f in facts:
            if f.get("component") in by:
                by[f["component"]] += 1
        for e in plan.get("events") or []:
            if e.get("component") in by:
                by[e["component"]] += 1
        short = {c: k for c, k in by.items() if k < q["per_component"]}
        if short:
            p.append(f"components below {q['per_component']}: {short}")
    return p


def style_twin(plan, twin_id="S8", style="polished"):
    """S8 from S1: every fact, turn, date and beat unchanged; only the writing style (and
    the persona's voice line) differ, so an S1-vs-S8 score difference is style alone."""
    import copy
    twin = copy.deepcopy(plan)
    twin["id"] = twin_id
    twin["derived_from"] = plan["id"]
    twin["persona"]["voice"] = STYLES[style]
    for session in twin["sessions"]:
        session["style"] = style
    return twin


def write_twins():
    """Write every derived plan whose source plan validates (S8 from S1)."""
    for twin_id, brief in BRIEFS.items():
        source = brief.get("derived_from")
        path = PLANS / f"{source}.json"
        if source and path.exists():
            plan = json.loads(path.read_text())
            if not validate(plan):
                (PLANS / f"{twin_id}.json").write_text(json.dumps(style_twin(plan, twin_id), indent=1) + "\n")


def main() -> int:
    tags, bad = labels(), 0
    if "--twins" in sys.argv:
        write_twins()
    for path in sorted(PLANS.glob("S*.json")):
        plan = json.loads(path.read_text())
        problems = validate(plan, tags=tags)
        bad += bool(problems)
        print(f"{path.name}: {'OK' if not problems else f'{len(problems)} problems'}")
        for x in problems[:25]:
            print("   -", x)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
