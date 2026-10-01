#!/usr/bin/env python3
"""Aggregate a private v3 replay trace without printing conversation content.

This reports mechanism coverage and failure stages. It cannot score whether a
new answering model used memory, because the seed stores recorded replies.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
    gold_storage = Counter()
    gold_funnel_by_type = defaultdict(Counter)
    source_counts = Counter()
    prompt_tokens = Counter()
    controls = Counter()
    probe_timing = Counter()
    lineage = Counter()
    written = Counter()
    clock = Counter()
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
        elif event == "clock":
            clock["historical_turn_scopes"] += 1
            clock["explicit_sql_now_statements"] += row["sql_now_statements"]
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
            blocks = stage.get("prompt_block_counts") or {}
            gate[f"{kind}:prompts_with_bookmarks"] += bool(blocks.get("bookmarks"))
            gate[f"{kind}:prompts_with_slots"] += bool(blocks.get("slots"))
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
                    for metric in ("gold_turns", "generated", "ranked", "rank_at_5",
                                   "rank_at_10", "source_turn_rank_at_5",
                                   "source_turn_rank_at_10", "budgeted", "selected",
                                   "source_note", "selected_or_source_note"):
                        controls[f"{arm}:gold_{metric}"] += arm_coverage.get(metric, 0)
                if not row.get("gold_turns"):
                    gold_funnel["unlabeled_section_prompts"] += 1
                    continue
                coverage = stage.get("gold_fragment_coverage")
                storage = row.get("source_storage_at_cutoff")
                if storage is None:
                    gold_storage["probes_not_instrumented"] += 1
                else:
                    for gold in row["gold_sources"]:
                        locations = [storage[source_id] for source_id in gold["source_ids"]
                                     if source_id in storage]
                        location = locations[0] if len(locations) == 1 else None
                        if location is None:
                            gold_storage["unresolved_gold_turns"] += 1
                        else:
                            gold_storage[f"{location['tier']}_gold_turns"] += 1
                            gold_storage["archived_gold_turns"] += bool(location["is_archived"])
                if coverage is None:
                    gold_funnel["not_instrumented"] += 1
                    continue
                if "source_turn_rank_at_5" not in coverage:
                    gold_funnel["source_turn_rank_not_instrumented_probes"] += 1
                typ = row["type"]
                for metric in ("gold_turns", "generated", "ranked", "rank_at_5",
                               "rank_at_10", "source_turn_rank_at_5",
                               "source_turn_rank_at_10", "budgeted", "selected", "source_note",
                               "selected_or_source_note",
                               "ranked_candidate_fragments",
                               "ranked_distinct_source_turns"):
                    gold_funnel[metric] += coverage.get(metric, 0)
                    gold_funnel_by_type[typ][metric] += coverage.get(metric, 0)
                gold_funnel["scored_source_linked_probes"] += 1
                gold_funnel_by_type[typ]["scored_source_linked_probes"] += 1
                for k in (5, 10):
                    hits = coverage[f"rank_at_{k}"]
                    if hits:
                        gold_funnel[f"any_gold_rank_at_{k}_probes"] += 1
                        gold_funnel_by_type[typ][f"any_gold_rank_at_{k}_probes"] += 1
                    if hits == coverage["gold_turns"]:
                        gold_funnel[f"all_gold_rank_at_{k}_probes"] += 1
                        gold_funnel_by_type[typ][f"all_gold_rank_at_{k}_probes"] += 1
                    if f"source_turn_rank_at_{k}" in coverage:
                        source_hits = coverage[f"source_turn_rank_at_{k}"]
                        if source_hits:
                            gold_funnel[f"any_gold_source_turn_rank_at_{k}_probes"] += 1
                            gold_funnel_by_type[typ][f"any_gold_source_turn_rank_at_{k}_probes"] += 1
                        if source_hits == coverage["gold_turns"]:
                            gold_funnel[f"all_gold_source_turn_rank_at_{k}_probes"] += 1
                            gold_funnel_by_type[typ][f"all_gold_source_turn_rank_at_{k}_probes"] += 1
                for leg, count in coverage["generated_by_leg"].items():
                    generated_gold_by_leg[leg] += count
                if coverage["selected"] == coverage["gold_turns"]:
                    gold_funnel["all_gold_in_selected_fragments_probes"] += 1
                    gold_funnel_by_type[typ]["all_gold_in_selected_fragments_probes"] += 1
                if coverage["selected_or_source_note"] == coverage["gold_turns"]:
                    gold_funnel["all_gold_in_final_evidence_probes"] += 1
                    gold_funnel_by_type[typ]["all_gold_in_final_evidence_probes"] += 1
                elif coverage["generated"] == 0 and coverage["source_note"] == 0:
                    gold_funnel["no_gold_candidate_probes"] += 1
                    gold_funnel_by_type[typ]["no_gold_candidate_probes"] += 1
                elif coverage["budgeted"] == 0 and coverage["source_note"] == 0:
                    gold_funnel["gold_lost_before_budgeted_probes"] += 1
                    gold_funnel_by_type[typ]["gold_lost_before_budgeted_probes"] += 1
                elif coverage["selected_or_source_note"] == 0:
                    gold_funnel["gold_lost_before_final_prompt_probes"] += 1
                    gold_funnel_by_type[typ]["gold_lost_before_final_prompt_probes"] += 1
                else:
                    gold_funnel["some_gold_in_final_evidence_probes"] += 1
                    gold_funnel_by_type[typ]["some_gold_in_final_evidence_probes"] += 1
    if run is None:
        raise ValueError("trace has no run event")
    complete_verified = False
    completion_error = None
    if complete and complete.get("complete_selected_corpus"):
        from scripts.z1.replay_validation import validate_complete_replay
        from scripts.z1.seed_v3 import EXPECTED
        try:
            with path.open() as source:
                validate_complete_replay((json.loads(line) for line in source), EXPECTED)
            complete_verified = True
        except ValueError as exc:
            completion_error = str(exc)
    return {
        "version": "v3",
        "complete_corpus_replay": complete_verified,
        "completion_validation_error": completion_error,
        "events": dict(events),
        "turns_by_conversation": complete.get("turns_by_conversation") if complete else None,
        "as_of_probes": complete.get("as_of_probes") if complete else None,
        "failures": dict(failures),
        "maintenance_calls": dict(maintenance),
        "historical_clock": dict(clock),
        "seed_clock_policy": run["meta"].get("extra", {}).get("clock_policy"),
        "timestamp_provenance_by_conversation": run["meta"].get("extra", {}).get(
            "timestamp_provenance_by_conversation", "not instrumented"),
        "gate": dict(gate),
        "generated_fragments_by_leg": dict(produced),
        "budgeted_fragments_by_leg": dict(budgeted),
        "selected_fragments_by_leg": dict(selected),
        "gold_fragment_funnel": dict(gold_funnel),
        "gold_source_storage": dict(gold_storage),
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
            "A complete original carried by a visible source-mode conversation note is credited separately from retrieved fragments.",
            "Bookmarked turns and persistent slots have no gold-source attribution in this trace; prompts using them are counted separately.",
            "Rank@5/10 counts gold turns whose source ID occurs in the first 5/10 pre-budget fragments; duplicate fragments and multi-source fragments remain distinct ranking slots.",
            "Source-turn rank@5/10 collapses repeated source turns before budget; sources cited in one multi-source fragment tie, and unprovenanced fragments have no source-turn rank.",
            "Delayed probe expected answers need review against intervening turns before scoring.",
            "vector_only is an ICE warm-vector-leg ablation with the shared gate/reranker, not an independent all-originals vector-memory baseline.",
            "no_codex removes direct graph/claim/timeline evidence; writer state and query expansion remain the full system's.",
            "Periodic jobs use a synchronous source-time due-check schedule, not the asynchronous runtime; clock scope is declared in seed_clock_policy.",
            "Constructed five-minute text-export clocks test simulated cadence/order, not authentic calendar dates or elapsed-time retention; check timestamp provenance.",
            "Memory job coverage is declared in the trace; idle session bursts and project/document paths are outside this replay.",
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
    return 1 if result["completion_validation_error"] or result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
