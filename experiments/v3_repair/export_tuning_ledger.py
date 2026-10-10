"""Export every sealed v3 settings candidate and its explicit tuning disposition."""

import argparse
import copy
import fcntl
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.prepare_inputs import digest, fingerprint, write_or_verify
from experiments.v3_repair.run_tuning_contexts import load_coordinate

PENDING = {
    "trust_activation": ("codex_direct_trust_floor", "codex_deep_strength_floor",
        "codex_entity_match_threshold", "codex_relation_fit_weight", "procedural_min_conf",
        "procedural_min_cited_turns", "turn_summary_coverage_threshold"),
    "gates_style": ("ltm_decision_threshold", "ltm_prior_bias", "ltm_length_weight",
        "ltm_pressure_midpoint_tokens", "ltm_pressure_scale_tokens", "confidence_fallback_threshold"),
    "lifecycle": ("decay_daily_unaccessed", "decay_daily_accessed", "decay_daily_creative",
        "decay_archive_threshold", "decay_cold_threshold", "decay_creative_floor", "decay_strengthen_amount",
        "codex_decay_daily", "codex_retention_floor", "codex_retention_increment",
        "codex_retention_cap", "codex_retention_rank_weight", "procedural_stale_days",
        "procedural_min_reinforcement", "cluster_similarity_threshold", "cluster_merge_similarity_threshold"),
}
FIXED_PROFILE = ("maintenance_intervals", "runtime_cycles_cap", "auto_finetune",
    "agent_max_scanned", "agent_max_llm_decisions", "agent_max_tier2_proposals",
    "agent_max_applications", "agent_pileup_min_pending", "agent_pileup_min_active_overlap",
    "agent_dup_cosine_threshold", "agent_dup_pairs_per_run", "agent_max_attempts",
    "agent_stale_slot_days", "agent_stale_task_days", "reflection_enrich_limit",
    "reflection_enrich_refresh_days")


def build(coordinates, inventory, profile):
    rows = copy.deepcopy(inventory["settings"])
    by_name = {r["name"]: r for r in rows}
    if len(by_name) != len(rows): raise ValueError("duplicate tuning inventory field")
    ranges, stages, caps = {}, {}, []
    pending = {name: stage for stage, names in PENDING.items() for name in names}
    if not set(pending).issubset(by_name): raise ValueError("named pending knob absent from sealed inventory")
    for coordinate in coordinates["rows"]:
        if "input_prompt_cap" in coordinate: caps.append(coordinate["input_prompt_cap"])
        for name, value in coordinate["overrides"].items():
            if name not in by_name: raise ValueError("planned knob absent from sealed inventory")
            if value not in ranges.setdefault(name, []): ranges[name].append(value)
            stages.setdefault(name, set()).add(coordinate["stage"])
    for row in rows:
        name = row["name"]
        row.update(trial_values=ranges.get(name), stage=sorted(stages[name]) if name in stages else pending.get(name),
            disposition="planned_read_grid" if name in ranges else
                        "pending_qualified_stage" if name in pending else
                        "retired_no_tuning" if name == "procedural_similarity_threshold" else "held_current_policy",
            requires_writer_replay=((name in PENDING["lifecycle"] and name != "codex_retention_rank_weight") or
                name in {"procedural_min_cited_turns", "turn_summary_coverage_threshold"}) if name in pending else
                False if name in ranges else None,
            dependency=coordinates["blocked_stages"].get(pending.get(name)), qualified=False)
    return dict(version="v3", method="named_tuning_dispositions_v2", score_of_record=False,
        coordinate_count=len(coordinates["rows"]), planned_settings_fields=len(ranges),
        disposition_counts=dict(Counter(r["disposition"] for r in rows)), settings=rows,
        research_input_cap=dict(name="input_prompt_cap", trial_values=sorted(set(caps)),
            consumer="production_parity.capped_window and final context ledger", stage="budget", qualified=False),
        fixed_profile_controls=profile, dynamic_reads=inventory["dynamic_reads"],
        rules=["All sealed fields have a disposition; no static reader becomes a measured live effect.",
               "Pending stages have no invented trial ranges; writer-affecting changes require a fresh state.",
               "Hard source/completion/privacy guards and model identity stay fixed, never optimized against the final test.",
               "Baseline plus at most two source-qualified survivors per read stage receive diagnostic cloud answers.",
               "Maintenance controls excluded by the original prefix inventory remain explicit fixed controls."])


def export(plan):
    seal, _ = load_coordinate(plan, "baseline")
    from src.api.config import settings
    if digest(settings.model_dump(mode="json")) != seal["identity"]["settings_sha256"]:
        raise ValueError("current fixed profile differs from sealed tuning settings")
    coordinates = json.loads((plan / "coordinates.json").read_text())
    inventory = json.loads((plan / "consumer-ledger.json").read_text())
    profile = {name: dict(resolved_value=getattr(settings, name), disposition="fixed_profile_not_a_score_knob")
               for name in FIXED_PROFILE}
    value = build(coordinates, inventory, profile)
    value["identity"] = dict(tuning_plan=seal["identity_sha256"],
                             instrument=fingerprint(Path(__file__)), config=fingerprint(ROOT / "src/api/config.py"))
    value["identity_sha256"] = digest(value["identity"])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan, out = args.tuning_plan.resolve(), args.out.resolve()
    if (any(not p.is_relative_to((ROOT / "logs").resolve()) for p in (plan, out))
            or out.is_relative_to(plan) or plan.is_relative_to(out)):
        raise ValueError("ledger source/output must be separate private paths")
    value = export(plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_or_verify(out, value)
    print(json.dumps(dict(version="v3", coordinate_count=value["coordinate_count"],
                         planned_settings_fields=value["planned_settings_fields"],
                         disposition_counts=value["disposition_counts"], score_of_record=False)))


if __name__ == "__main__": main()
