#!/usr/bin/env python3
"""Check the current v3 answer judge in both orders without declaring it qualified.

Authored controls diagnose rubric failures. Human-reviewed real answer pairs
measure agreement on the actual workload; they are a separate packet kind.
Every cloud call goes through the campaign's judge_one, never a second rubric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.z1 import judge_answers
from scripts.z1.answer_as_of import atomic_json, private_path
from scripts.z1.run_meta import file_digest, run_meta


def authored_controls() -> dict:
    cases = []

    def add(identity, category, source, question, expected, a, b, grades, winner):
        cases.append({"case_id": identity, "category": category,
                      "question": question, "question_time": "2025-03-03T12:00:00+00:00",
                      "source": source, "expected_answer": expected,
                      "answer_a": a, "answer_b": b,
                      "reference": {"a_grade": grades[0], "b_grade": grades[1],
                                    "winner": winner,
                                    "reason": "Authored source states the answer or its absence explicitly."}})

    port = "Source turn 1 recorded at 2025-01-01T12:00:00+00:00\nUser: I set the Cedar service port to 7412."
    add("exact", "exact", port, "Which port did I set for Cedar?", "7412",
        "7412", "8124", ("correct", "incorrect"), "a")
    add("paraphrase", "paraphrase", port, "Which port did I set for Cedar?", "7412",
        "Cedar listens on port seven thousand four hundred and twelve.",
        "Use a standard HTTP port.", ("correct", "incorrect"), "a")
    add("both_fail", "both_failed", port, "Which port did I set for Cedar?", "7412",
        "8124", "9000", ("incorrect", "incorrect"), "TIE")
    add("equivalent", "equivalent", port, "Which port did I set for Cedar?", "7412",
        "7412", "You set Cedar to port 7412.", ("correct", "correct"), "TIE")
    add("partial", "multi_source", port + "\nSource turn 2 recorded at 2025-01-02T12:00:00+00:00\nUser: Cedar's owner is Mira.",
        "What are Cedar's port and owner?", "Port 7412; owner Mira.",
        "Port 7412.", "Port 7412, owned by Mira.", ("partial", "correct"), "b")
    add("wrong_entity", "entity", port + "\nUser: Alder uses port 8124.",
        "Which port did I set for Cedar?", "7412", "8124", "7412",
        ("incorrect", "correct"), "b")
    update = ("Source turn 1 recorded at 2025-01-01T12:00:00+00:00\nUser: My current theme is green.\n"
              "Source turn 2 recorded at 2025-02-01T12:00:00+00:00\nUser: I changed my theme to blue; green is the old one.")
    add("updated", "temporal_update", update, "What is my current theme?", "blue",
        "Green.", "Blue.", ("incorrect", "correct"), "b")
    add("historical", "temporal_history", update, "What theme did I use in January?", "green",
        "Green.", "Blue.", ("correct", "incorrect"), "a")
    suggestion = ("Source turn 1 recorded at 2025-01-01T12:00:00+00:00\n"
                  "Assistant: You could use Cedar.\nUser: I have not chosen a service yet.")
    add("suggestion", "attribution", suggestion, "Which service did I choose?",
        "No service chosen; Cedar was an assistant suggestion.",
        "You chose Cedar.", "You have not chosen; I only suggested Cedar.",
        ("incorrect", "correct"), "b")
    add("unsupported", "insufficient_evidence", port,
        "What is my pet's name?", "Insufficient evidence to identify the name.",
        "Pip.", "Momo.", ("uncertain", "uncertain"), "TIE")
    add("abstention", "explicit_absence", "Source turn 1 recorded at 2025-01-01T12:00:00+00:00\nUser: I have not chosen a launch date.",
        "What launch date did I choose?", "No launch date chosen.",
        "You have not chosen a date.", "You chose March 1.",
        ("correct", "incorrect"), "a")
    long_source = ("Source turn 1 recorded at 2025-01-01T12:00:00+00:00\nUser: "
                   + "Archive filler without a code. " * 420
                   + "The final archive code is pebble-63.")
    add("long_tail", "long_source", long_source, "What archive code did I give?",
        "pebble-63", "pebble-63", "pebble-36", ("correct", "incorrect"), "a")
    return {"version": "v3", "kind": "authored_answer_judge_controls",
            "reviewer": "authored diagnostic references; not human calibration",
            "cases": cases}


def validate_packet(packet: dict) -> list[dict]:
    if (packet.get("version") != "v3" or packet.get("kind") not in
            {"authored_answer_judge_controls", "human_reviewed_answer_pairs"}
            or not str(packet.get("reviewer", "")).strip()):
        raise ValueError("packet needs explicit v3 kind and independent reference reviewer")
    cases, seen = packet.get("cases"), set()
    if not isinstance(cases, list) or not cases:
        raise ValueError("calibration packet has no cases")
    for case in cases:
        for key in ("case_id", "category", "question", "question_time", "source",
                    "expected_answer", "answer_a", "answer_b"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"calibration case lacks {key}")
        if case["case_id"] in seen:
            raise ValueError("duplicate calibration case")
        seen.add(case["case_id"])
        ref = case.get("reference") or {}
        if (ref.get("a_grade") not in judge_answers.GRADES
                or ref.get("b_grade") not in judge_answers.GRADES
                or ref.get("winner") not in {"a", "b", "TIE"}
                or not str(ref.get("reason", "")).strip()):
            raise ValueError("calibration references need both grades, winner and reason")
        a, b = ref["a_grade"], ref["b_grade"]
        ranks = {"incorrect": 0, "partial": 1, "correct": 2}
        if a in ranks and b in ranks and a != b:
            if ref["winner"] != ("a" if ranks[a] > ranks[b] else "b"):
                raise ValueError("reference preference contradicts its absolute grades")
        if a == b == "incorrect" and ref["winner"] != "TIE":
            raise ValueError("two failed reference answers must tie")
    return cases


def mapped_result(case: dict, order: str, verdict: dict) -> dict:
    if not isinstance(verdict, dict) or verdict.get("verdict") != "ERROR":
        verdict = judge_answers.validate_verdict(verdict, absolute=True)
    first_is_a = order == "ab"
    winner = verdict["verdict"]
    if winner in {"A", "B"}:
        winner = "a" if ((winner == "A") == first_is_a) else "b"
    return {"case_id": case["case_id"], "category": case["category"],
            "order": order, "winner": winner, "reason": verdict["reason"],
            "a_grade": verdict.get("A_grade" if first_is_a else "B_grade"),
            "b_grade": verdict.get("B_grade" if first_is_a else "A_grade"),
            "note": verdict.get("note", "")}


def summarize(cases: list[dict], results: list[dict]) -> dict:
    references = {c["case_id"]: c["reference"] for c in cases}
    expected_keys = {(c["case_id"], order) for c in cases for order in ("ab", "ba")}
    seen, by_case = set(), {}
    preference_hits, grade_hits, confusion, errors = 0, 0, Counter(), 0
    for row in results:
        key = (row["case_id"], row["order"])
        if key not in expected_keys or key in seen:
            raise ValueError("unknown or duplicated calibration call identity")
        seen.add(key)
        ref = references[row["case_id"]]
        errors += row["winner"] == "ERROR"
        preference_hits += row["winner"] == ref["winner"]
        for arm in ("a", "b"):
            actual = row.get(f"{arm}_grade") or "ERROR"
            expected = ref[f"{arm}_grade"]
            grade_hits += actual == expected
            confusion[(expected, actual)] += 1
        by_case.setdefault(row["case_id"], []).append(row)
    complete_pairs = [rows for rows in by_case.values() if len(rows) == 2]
    comparable = [rows for rows in complete_pairs
                  if all(row["winner"] != "ERROR" for row in rows)]
    consistent = sum(all(rows[0][key] == rows[1][key]
                         for key in ("winner", "a_grade", "b_grade"))
                     for rows in comparable)
    return {"planned_cases": len(cases), "planned_calls": len(expected_keys),
            "observed_calls": len(results), "errors": errors,
            "complete": seen == expected_keys and errors == 0,
            "preference_matches": preference_hits,
            "preference_denominator": len(results),
            "grade_matches": grade_hits, "grade_denominator": 2 * len(results),
            "order_consistent_cases": consistent,
            "order_pair_denominator": len(complete_pairs),
            "order_comparable_cases": len(comparable),
            "confusion": [{"reference": expected, "judge": actual, "count": count}
                          for (expected, actual), count in sorted(confusion.items())]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-controls", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.write_controls:
        path = private_path(args.write_controls)
        if path.exists():
            raise FileExistsError("controls packet already exists")
        atomic_json(path, authored_controls())
        print(f"v3 authored controls: 12 cases; {path}")
        return 0
    if not args.packet or not args.out:
        parser.error("--packet and --out are required unless writing controls")
    packet_path, output = private_path(args.packet), private_path(args.out)
    raw = packet_path.read_bytes()
    packet = json.loads(raw)
    cases = validate_packet(packet)  # All references checked before any cloud call.
    identity = {"packet_sha256": hashlib.sha256(raw).hexdigest(),
                "packet_kind": packet["kind"],
                **judge_answers.judge_provider_identity(),
                "rubric_sha256": hashlib.sha256(judge_answers.SYSTEM_V3.encode()).hexdigest(),
                "judge_implementation_sha256": hashlib.sha256(
                    Path(judge_answers.__file__).read_bytes()).hexdigest()}
    if args.plan:
        print(json.dumps({**identity, "cases": len(cases), "calls": 2 * len(cases),
                          "cloud_called": False, "score_of_record": False}, indent=2))
        return 0
    results = []
    if output.exists():
        if not args.resume:
            raise FileExistsError("calibration output exists; use matching --resume")
        saved = json.loads(output.read_text())
        if saved.get("identity") != identity:
            raise ValueError("calibration resume has different packet/model/rubric/implementation")
        results = saved["results"]
        summarize(cases, results)
    meta = run_meta(script=__file__, args=vars(args),
                    inputs=[file_digest(packet_path)], extra=identity)
    observed = {(r["case_id"], r["order"]) for r in results}
    for case in cases:
        for order in ("ab", "ba"):
            if (case["case_id"], order) in observed:
                continue
            first, second = ((case["answer_a"], case["answer_b"]) if order == "ab"
                             else (case["answer_b"], case["answer_a"]))
            verdict = judge_answers.judge_one(
                case["question"], case["source"], first, second,
                expected_answer=case["expected_answer"], question_time=case["question_time"],
                session_id="ice-z1-calibration-" + identity["packet_sha256"][:20]
                + "-" + hashlib.sha256(case["case_id"].encode()).hexdigest()[:12])
            results.append(mapped_result(case, order, verdict))
            summary = summarize(cases, results)
            atomic_json(output, {"version": "v3", "kind": "answer_judge_calibration",
                                 "meta": meta, "identity": identity,
                                 "score_of_record": False, "qualified": False,
                                 "summary": summary, "results": results})
            print(f"v3 {len(results)}/{2 * len(cases)} {case['case_id']} {order}: "
                  f"{results[-1]['winner']}", flush=True)
            if (len(results) >= 3
                    and all(row["winner"] == "ERROR" for row in results[-3:])):
                print("v3 calibration stopped after three consecutive judge errors; "
                      "partial results preserved", flush=True)
                return 1
    summary = summarize(cases, results)
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
