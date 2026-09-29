#!/usr/bin/env python3
"""Aggregate a private v3 replay trace without printing conversation content.

This reports mechanism coverage and failure stages. It cannot score whether a
new answering model used memory, because the seed stores recorded replies.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(path: Path) -> dict:
    events = Counter()
    failures = Counter()
    maintenance = Counter()
    gate = Counter()
    produced = Counter()
    budgeted = Counter()
    selected = Counter()
    generated_gold_by_leg = Counter()
    gold_funnel = Counter()
    gold_funnel_by_type = defaultdict(Counter)
    source_counts = Counter()
    prompt_tokens = Counter()
    controls = Counter()
    probe_timing = Counter()
    lineage = Counter()
    written = Counter()
    complete = None
    run = None
    for line_number, line in enumerate(path.open(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}") from exc
        event = row.get("event")
        events[event] += 1
        if event == "run":
            if run is not None or line_number != 1:
                raise ValueError("trace must start with exactly one run event")
            run = row
        elif event == "complete":
            complete = row
        elif event == "failed":
            failures[f"{row.get('stage')}:{row.get('error_type')}"] += 1
        elif event == "maintenance_failed":
            failures[f"maintenance:{row.get('job')}:{row.get('error_type')}"] += 1
        elif event == "maintenance":
            maintenance[row["job"]] += 1
        elif event == "written":
            written["turns"] += 1
            written["lossless"] += bool(row.get("lossless"))
            written["inject_raw"] += bool(row.get("inject_raw"))
            written["summary_support_unknown"] += (
                row.get("summary_support") is None)
            source_counts.update(row.get("source_counts") or {})
        elif event in ("turn", "as_of_probe"):
            stage = row["preflight"]
            kind = "historical_turn" if event == "turn" else "as_of_probe"
            gate[f"{kind}:B2_on"] += bool(stage["gate"]["b2_retrieve"])
            gate[f"{kind}:final_on"] += bool(stage["gate"]["final_retrieve"])
            gate[f"{kind}:source_checked"] += bool(
                stage["gate"]["source_refinement_applied"])
            gate[f"{kind}:source_action:{stage['gate']['source_action']}"] += 1
            prompt_tokens[f"{kind}:total"] += stage["prompt_tokens"]
            prompt_tokens[f"{kind}:selected"] += stage["selected_tokens"]
            prompt_tokens[f"{kind}:budget"] += stage["gate"]["total_budget"]
            for leg, rows in stage["generated_by_leg"].items():
                produced[f"{kind}:{leg}"] += len(rows)
            for frag in stage["budgeted"]:
                budgeted[f"{kind}:{frag['leg']}"] += 1
            for frag in stage["selected"]:
                selected[f"{kind}:{frag['leg']}"] += 1
            if (any(stage["generated_by_leg"].values())
                    and not stage["selected"]):
                gate[f"{kind}:candidates_but_no_selected_fragments"] += 1
            source = stage.get("lineage") or {}
            lineage["selected_with_source_identity"] += source.get(
                "selected_with_source_identity", 0)
            lineage["selected_total"] += source.get("selected_total", 0)
            lineage["unresolved_occurrences"] += len(source.get(
                "unresolved_source_ids", []))
            lineage["future_occurrences"] += len(source.get(
                "future_source_ids", []))
            if event == "as_of_probe":
                probe_timing[f"cutoff:{row.get('cutoff_kind', 'unversioned')}"] += 1
                probe_timing[f"label:{row.get('label_status', 'unversioned')}"] += 1
                for arm, arm_stage in row.get("controls", {}).items():
                    controls[f"{arm}:probes"] += 1
                    controls[f"{arm}:prompt_tokens"] += arm_stage["prompt_tokens"]
                    controls[f"{arm}:selected_tokens"] += arm_stage["selected_tokens"]
                    controls[f"{arm}:selected_fragments"] += len(arm_stage["selected"])
                    arm_coverage = arm_stage.get("gold_fragment_coverage") or {}
                    for metric in ("gold_turns", "generated", "budgeted", "selected"):
                        controls[f"{arm}:gold_{metric}"] += arm_coverage.get(metric, 0)
                if not row.get("gold_turns"):
                    gold_funnel["unlabeled_section_prompts"] += 1
                    continue
                coverage = stage.get("gold_fragment_coverage")
                if coverage is None:
                    gold_funnel["not_instrumented"] += 1
                    continue
                typ = row["type"]
                for metric in ("gold_turns", "generated", "budgeted", "selected"):
                    gold_funnel[metric] += coverage[metric]
                    gold_funnel_by_type[typ][metric] += coverage[metric]
                for leg, count in coverage["generated_by_leg"].items():
                    generated_gold_by_leg[leg] += count
                if coverage["selected"] == coverage["gold_turns"]:
                    gold_funnel["all_gold_in_selected_fragments_probes"] += 1
                    gold_funnel_by_type[typ]["all_gold_in_selected_fragments_probes"] += 1
                elif coverage["generated"] == 0:
                    gold_funnel["no_gold_candidate_probes"] += 1
                    gold_funnel_by_type[typ]["no_gold_candidate_probes"] += 1
                elif coverage["budgeted"] == 0:
                    gold_funnel["gold_lost_before_budgeted_probes"] += 1
                    gold_funnel_by_type[typ]["gold_lost_before_budgeted_probes"] += 1
                elif coverage["selected"] == 0:
                    gold_funnel["gold_lost_before_final_prompt_probes"] += 1
                    gold_funnel_by_type[typ]["gold_lost_before_final_prompt_probes"] += 1
                else:
                    gold_funnel["some_gold_selected_probes"] += 1
                    gold_funnel_by_type[typ]["some_gold_selected_probes"] += 1
    if run is None:
        raise ValueError("trace has no run event")
    return {
        "version": "v3",
        "complete_corpus_replay": bool(complete and complete.get("complete_selected_corpus")),
        "events": dict(events),
        "turns_by_conversation": complete.get("turns_by_conversation") if complete else None,
        "as_of_probes": complete.get("as_of_probes") if complete else None,
        "failures": dict(failures),
        "maintenance_calls": dict(maintenance),
        "gate": dict(gate),
        "generated_fragments_by_leg": dict(produced),
        "budgeted_fragments_by_leg": dict(budgeted),
        "selected_fragments_by_leg": dict(selected),
        "gold_fragment_funnel": dict(gold_funnel),
        "gold_fragment_funnel_by_type": {key: dict(value) for key, value
                                         in sorted(gold_funnel_by_type.items())},
        "gold_candidates_by_leg": dict(generated_gold_by_leg),
        "source_lineage": dict(lineage),
        "post_flight_writes": dict(written),
        "post_flight_source_counts": dict(source_counts),
        "prompt_token_sums": dict(prompt_tokens),
        "matched_controls": dict(controls),
        "probe_timing_and_label_status": dict(probe_timing),
        "table_counts": complete.get("table_counts") if complete else None,
        "limits": [
            "Recorded assistant replies were not generated by ICE; answer use is unmeasured.",
            "Gold funnel credits provenance of selected fragments, not recent-history text.",
            "Delayed probe expected answers need review against intervening turns before scoring.",
            "Periodic jobs use source-time due checks but their internals see wall clock.",
            "Decay, cold archive and project-document paths were not replayed chronologically.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, help="optional aggregate JSON under logs/")
    args = parser.parse_args()
    result = summarize(args.trace)
    body = json.dumps(result, indent=2, sort_keys=True)
    print(body)
    if args.out:
        logs = (Path(__file__).resolve().parents[2] / "logs").resolve()
        out = args.out.resolve()
        if not args.trace.resolve().is_relative_to(logs) or not out.is_relative_to(logs):
            raise ValueError("trace and report must remain under logs/")
        out.write_text(body + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
