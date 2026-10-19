"""Z2 D6a: qualify every judge prompt on the development history before B/C.

Each prompt runs on planted controls built from the 67 reviewed development
questions (content contracts; dev history ecc64aab — never in Z2 or FINAL results):

1. a writer family (DeepSeek, never a judge) writes from the question, its reviewed
   requirements and the gold turns: a CORRECT answer, a WRONG one (on topic, same
   style, one specific changed — it must not pass), a PARTIAL one, a faithful one-turn
   SUMMARY and an UNFAITHFUL one (one specific changed), and a CORRECTION message that
   would change the answer;
2. the other writer family (MiMo) checks each plant does what it claims; a plant that
   fails the check is dropped and counted (never repaired);
3. every judge prompt is run on the surviving plants exactly as its stage runs it:
   S11 grade (correct / wrong / partial / abstain), S11 ranking (two orders), S11 claim
   checks, S8 judge (faithful vs unfaithful summary), D4 walk verifier (a correction vs
   unrelated turns), S11 controls (right vs wrong reply).

Recorded in qualification.json with each prompt's sha256, the measured accuracies,
pass/fail against the thresholds below, and a sample sheet for the maintainer.

    uv run --no-sync python experiments/z2/qualify.py --out logs/z2-qualify-<date>-r1
"""

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import bg_audit, grade, walk_audit
from experiments.z2 import ledger as L
from experiments.z2 import ledger_merge as M
from experiments.z2.ledger_recall import contracts
from experiments.z2.questions import ask_object

POLICY = "ice-z2-qualify-1"
DEV = "dev-ecc64aab"
WRITER, CHECKER, JUDGE = "deepseek", "mimo", "muse"
# Pass bars, fixed before running: a planted wrong answer must essentially never pass;
# right material must be accepted nearly always.
BARS = dict(wrong_rejected=0.95, correct_accepted=0.90, ranking_right=0.90, faithful_split=0.90,
            walk_split=0.90, claims_split=0.90, controls_right=0.90, partwise_split=0.90)

PLANT_SYSTEM = """You build test material for checking an answer grader. You get a QUESTION a user
asked about their conversation with an AI assistant, the REQUIREMENTS a correct answer must contain
(reviewed by a human), and the conversation TURNS they come from.
Write, in the assistant's natural voice:
- correct: an answer containing every requirement;
- wrong: the same kind of answer, on topic and confident, but with ONE specific changed to a
  plausible false value (a number, name, item, decision or direction); wrong_change says what you
  changed, as "<true value> -> <false value>";
- partial: a correct answer that leaves out about half of the requirements;
- summary: a faithful 2-4 sentence summary of TURN_TO_SUMMARIZE;
- unfaithful_summary: the same summary with ONE specific changed to a false value;
- correction: a short message the user might send later that changes the answer to QUESTION
  (e.g. "actually I changed X to Y"); new_answer: the answer after that correction.
Return only JSON with keys correct, wrong, wrong_change, partial, summary, unfaithful_summary,
correction, new_answer."""

CHECK_SYSTEM = """Check test material for an answer grader against REQUIREMENTS and TURNS.
Return only JSON:
{"correct_has_all": true|false, "wrong_contradicts": true|false, "partial_misses_some": true|false,
 "summary_faithful": true|false, "unfaithful_contradicts": true|false, "correction_changes_answer": true|false}"""


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def plants(items, originals, out, *, models):
    from concurrent.futures import ThreadPoolExecutor
    calls = M.Calls(out / "plants-raw.jsonl")
    session = lambda role: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{role}"))
    turns = lambda c: [dict(turn=t, user=originals[t - 1]["prompt"][:4000], assistant=originals[t - 1]["response"][:4000])
                       for t in sorted(c["gold_turns"])]

    def one(c):
        body = dict(QUESTION=c["question"], REQUIREMENTS=c["requirements"], TURNS=turns(c),
                    TURN_TO_SUMMARIZE=turns(c)[0])
        made = ask_object(calls, WRITER, models[WRITER][1], models[WRITER][0], PLANT_SYSTEM,
                          json.dumps(body, ensure_ascii=False), session(WRITER), max_output_tokens=4000) or {}
        if not all(isinstance(made.get(k), str) and made[k].strip() for k in
                   ("correct", "wrong", "partial", "summary", "unfaithful_summary", "correction")):
            return dict(contract=c, plant=None, reason="writer reply incomplete")
        check = ask_object(calls, CHECKER, models[CHECKER][1], models[CHECKER][0], CHECK_SYSTEM,
                           json.dumps(dict(REQUIREMENTS=c["requirements"], TURNS=turns(c), **{
                               k: made[k] for k in ("correct", "wrong", "partial", "summary",
                                                    "unfaithful_summary", "correction")}), ensure_ascii=False),
                           session(CHECKER)) or {}
        return dict(contract=c, plant=made, check=check, turns=turns(c))

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, items))


def judge_runs(rows, out, *, models):
    """Every judge prompt on the plants that passed the other family's check."""
    from concurrent.futures import ThreadPoolExecutor
    calls = M.Calls(out / "judge-raw.jsonl")
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{JUDGE}"))
    ask = lambda system, prompt: ask_object(calls, JUDGE, models[JUDGE][1], models[JUDGE][0], system, prompt,
                                            session, max_output_tokens=8000)

    def one(row):
        c, p, chk = row["contract"], row["plant"], row["check"]
        res = dict(probe_id=c["probe_id"])
        cited = [dict(turn=t["turn"], role=role, text=t[role]) for t in row["turns"] for role in ("user", "assistant")]
        grade_body = lambda answer: json.dumps(dict(QUESTION=c["question"], REQUIRED=c["requirements"], SOURCES=cited,
                                                    SUPERSEDED=[], EXPECT_ABSTENTION=False, ANSWER=answer),
                                               ensure_ascii=False)
        if chk.get("correct_has_all"):
            res["grade_correct"] = ask(grade.GRADE_SYSTEM, grade_body(p["correct"]))
        if chk.get("wrong_contradicts"):
            res["grade_wrong"] = ask(grade.GRADE_SYSTEM, grade_body(p["wrong"]))
        if chk.get("partial_misses_some"):
            res["grade_partial"] = ask(grade.GRADE_SYSTEM, grade_body(p["partial"]))
        res["grade_abstain"] = ask(grade.GRADE_SYSTEM, grade_body("I don't have any record of that in our conversation."))
        if chk.get("correct_has_all") and chk.get("wrong_contradicts"):
            orders = []
            for first, second in ((p["correct"], p["wrong"]), (p["wrong"], p["correct"])):
                v = ask(grade.RANK_SYSTEM, json.dumps(dict(QUESTION=c["question"], REQUIRED=c["requirements"],
                                                           SOURCES=cited, ANSWERS=[dict(label="A", text=first), dict(label="B", text=second)]),
                                                      ensure_ascii=False)) or {}
                r = v.get("ranks") or {}
                correct_label = "A" if first == p["correct"] else "B"
                other = "B" if correct_label == "A" else "A"
                orders.append(isinstance(r.get(correct_label), (int, float)) and isinstance(r.get(other), (int, float))
                              and r[correct_label] < r[other])
            res["rank_right"] = orders
            passages = [dict(index=n, turn=t["turn"], role="assistant", text=t["assistant"]) for n, t in enumerate(row["turns"])] + \
                       [dict(index=len(row["turns"]) + n, turn=t["turn"], role="user", text=t["user"]) for n, t in enumerate(row["turns"])]
            true_claim = c["requirements"][0]
            false_claim = (p.get("wrong_change") or "").split("->")[-1].strip()
            if false_claim:
                res["claim_true_found"] = (ask(grade.CLAIM_SOURCE_SYSTEM, json.dumps(dict(CLAIM=true_claim, PASSAGES=passages),
                                                                                  ensure_ascii=False)) or {}).get("found")
                res["claim_false_found"] = (ask(grade.CLAIM_SOURCE_SYSTEM, json.dumps(dict(
                    CLAIM=f"{c['question']} — {false_claim}", PASSAGES=passages), ensure_ascii=False)) or {}).get("found")
        turn0 = row["turns"][0]
        src = [dict(turn=turn0["turn"], text=f"User: {turn0['user']}\n\nAssistant: {turn0['assistant']}")]
        for key, text in (("summary_faithful", p["summary"]), ("summary_unfaithful", p["unfaithful_summary"])):
            if chk.get("summary_faithful" if key == "summary_faithful" else "unfaithful_contradicts"):
                item = dict(id=f"q:{key}", kind="turn_summary", text=text, sources=src)
                res[key] = (ask(bg_audit.JUDGE_SYSTEM, bg_audit.judge_prompt(item, [])[1]) or {}).get("faithful", {}).get("verdict")
        # S8 part-wise judging (sources over the judge's capacity): the same summaries with
        # the source split in two parts, combined as the stage combines them.
        whole = src[0]["text"]
        halves = [dict(turn=turn0["turn"], text=whole[:len(whole) // 2]), dict(turn=turn0["turn"], text=whole[len(whole) // 2:])]
        for key, text in (("partwise_faithful", p["summary"]), ("partwise_unfaithful", p["unfaithful_summary"])):
            if chk.get("summary_faithful" if key == "partwise_faithful" else "unfaithful_contradicts"):
                item = dict(id=f"q:{key}", kind="session_summary", text=text, sources=halves)
                numbered = bg_audit.statements(item)
                verdicts = [ask(bg_audit.PART_SYSTEM, json.dumps(dict(
                    STATEMENTS={str(i): t for i, t in enumerate(numbered, 1)}, PASSAGES=[part]), ensure_ascii=False))
                    for part in halves]
                res[key] = bg_audit.combine_parts(len(numbered), verdicts)[0]["faithful"]
        if chk.get("correction_changes_answer"):
            later = max(c["gold_turns"]) + 1
            base = dict(QUESTION=c["question"], CURRENT_ANSWER=c["requirements"], ANSWER_TURN=max(c["gold_turns"]))
            with_change = dict(base, NEW_TURNS=[dict(turn=later, role="user", text=p["correction"])])
            unrelated = dict(base, NEW_TURNS=[dict(turn=later, role="user", text="thanks, that's all for now")])
            res["walk_change"] = (ask(walk_audit.WALK_SYSTEM, json.dumps(with_change, ensure_ascii=False)) or {}).get("changes")
            res["walk_none"] = (ask(walk_audit.WALK_SYSTEM, json.dumps(unrelated, ensure_ascii=False)) or {}).get("changes")
        return res

    def safe(row):
        try:
            return one(row)
        except Exception as exc:                  # recorded and counted, never fatal
            return dict(probe_id=row["contract"]["probe_id"], error=f"{type(exc).__name__}: {exc}"[:300])

    good = [r for r in rows if r.get("plant")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(safe, good))


CONTROL_PLANTS = [("what's 17 times 23", "17 x 23 = 391.", "17 x 23 = 401."),
                  ("capital of australia?", "Canberra is the capital of Australia.", "Sydney is the capital of Australia."),
                  ("how many days in a leap year", "A leap year has 366 days.", "A leap year has 365 days."),
                  ("translate 'thank you' to spanish", "'Thank you' in Spanish is 'gracias'.", "'Thank you' in Spanish is 'por favor'."),
                  ("boiling point of water at sea level in celsius", "100 °C.", "90 °C.")]


def controls_run(out, *, models):
    calls = M.Calls(out / "controls-raw.jsonl")
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|controls"))
    right = []
    for msg, good, bad in CONTROL_PLANTS:
        for a, b, good_is in ((good, bad, "A"), (bad, good, "B")):
            v = (ask_object(calls, JUDGE, models[JUDGE][1], models[JUDGE][0], grade.CONTROL_SYSTEM,
                            json.dumps(dict(MESSAGE=msg, A=a, B=b), ensure_ascii=False), session,
                            max_output_tokens=4000) or {}).get("better")
            right.append(v == good_is)
    return right


def rate(xs):
    xs = [x for x in xs if x is not None]
    return (round(sum(xs) / len(xs), 3), len(xs)) if xs else (None, 0)


def summarize(rows, judged, control_right):
    g = lambda key, test: [test(j[key]) if j.get(key) else None for j in judged if key in j]
    metrics = dict(
        correct_accepted=rate(g("grade_correct", lambda v: v.get("grade") == "correct" and (v.get("score") or 0) >= 4)),
        wrong_rejected=rate(g("grade_wrong", lambda v: v.get("grade") != "correct" and (v.get("score") or 5) <= 3)),
        partial_graded_partial=rate(g("grade_partial", lambda v: v.get("grade") == "partial")),
        abstain_not_correct=rate(g("grade_abstain", lambda v: v.get("grade") != "correct" and v.get("abstained") is True)),
        ranking_right=rate([o for j in judged for o in j.get("rank_right", [])]),
        faithful_split=rate([j["summary_faithful"] == "yes" for j in judged if "summary_faithful" in j]
                            + [j["summary_unfaithful"] == "no" for j in judged if "summary_unfaithful" in j]),
        claims_split=rate([j.get("claim_true_found") is True for j in judged if "claim_true_found" in j]
                          + [j.get("claim_false_found") is False for j in judged if "claim_false_found" in j]),
        walk_split=rate([j.get("walk_change") is True for j in judged if "walk_change" in j]
                        + [j.get("walk_none") is False for j in judged if "walk_none" in j]),
        controls_right=rate(control_right),
        partwise_split=rate([j["partwise_faithful"] == "yes" for j in judged if "partwise_faithful" in j]
                            + [j["partwise_unfaithful"] == "no" for j in judged if "partwise_unfaithful" in j]))
    checks = [r.get("check") or {} for r in rows if r.get("plant")]
    plant_quality = {k: rate([c.get(k) is True for c in checks]) for k in
                     ("correct_has_all", "wrong_contradicts", "partial_misses_some", "summary_faithful",
                      "unfaithful_contradicts", "correction_changes_answer")}
    passed = {k: (metrics[k][0] is not None and metrics[k][0] >= bar) for k, bar in BARS.items()}
    return metrics, plant_quality, passed


def sheet(rows, judged, path, n=8):
    by_id = {j["probe_id"]: j for j in judged}
    lines = ["# D6a qualification sample — for the maintainer", "",
             "Each row: a planted answer and what the grader said. Mark any verdict you disagree with.", ""]
    for r in [r for r in rows if r.get("plant")][:n]:
        c, p, j = r["contract"], r["plant"], by_id.get(r["contract"]["probe_id"], {})
        lines += [f"## {c['probe_id']}", f"**Question:** {c['question']}", f"**Requirements:** {c['requirements']}", ""]
        for key, label in (("correct", "CORRECT"), ("wrong", "WRONG (" + str(p.get("wrong_change")) + ")"), ("partial", "PARTIAL")):
            v = j.get(f"grade_{key}") or {}
            lines += [f"- {label}: {p[key][:400]}", f"  → grader: {v.get('grade')} / {v.get('score')} — {v.get('reason')}"]
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("qualification material is private: keep it under logs/")
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    args.out.mkdir(parents=True, exist_ok=True)
    models = {r: (generator(r).generate, metadata(r)) for r in (WRITER, CHECKER, JUDGE)}
    items = contracts()[:args.limit] if args.limit else contracts()
    rows = plants(items, L.load_history(DEV), args.out, models=models)
    judged = judge_runs(rows, args.out, models=models)
    control_right = controls_run(args.out, models=models)
    metrics, plant_quality, passed = summarize(rows, judged, control_right)
    prompts = {name: sha(text) for name, text in (
        ("grade.GRADE_SYSTEM", grade.GRADE_SYSTEM), ("grade.RANK_SYSTEM", grade.RANK_SYSTEM),
        ("grade.CLAIM_SOURCE_SYSTEM", grade.CLAIM_SOURCE_SYSTEM), ("grade.CONTROL_SYSTEM", grade.CONTROL_SYSTEM),
        ("bg_audit.JUDGE_SYSTEM", bg_audit.JUDGE_SYSTEM), ("bg_audit.PART_SYSTEM", bg_audit.PART_SYSTEM),
        ("walk_audit.WALK_SYSTEM", walk_audit.WALK_SYSTEM))}
    record = dict(policy=POLICY, judge=metadata(JUDGE), writer=WRITER, checker=CHECKER, contracts=len(items),
                  judge_errors=[j for j in judged if "error" in j],
                  plants_kept=sum(1 for r in rows if r.get("plant")), prompt_sha256=prompts, bars=BARS,
                  metrics=metrics, plant_quality=plant_quality, passed=passed, all_passed=all(passed.values()))
    (args.out / "qualification.json").write_text(json.dumps(dict(record, judged=judged), indent=1, ensure_ascii=False,
                                                            default=str) + "\n")
    sheet(rows, judged, args.out / "review-sample.md")
    print(json.dumps(dict(metrics=metrics, passed=passed), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
