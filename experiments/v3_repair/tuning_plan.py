"""Seal bounded ICE v3 read-side tuning candidates, without running them."""

import argparse
import fcntl
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.prepare_inputs import digest, encoded, fingerprint, prepare, write_or_verify

POLICY = "ice-v3-bounded-tuning-coordinates-1"


def coordinates(settings):
    base = settings.model_dump(mode="json")
    rows = [dict(id="baseline", stage="baseline", overrides={},
                 live_delta=None, cost=None, verdict="unmeasured", requires_writer_replay=False)]
    def add(stage, overrides, *, prompt_cap=None):
        if prompt_cap is None and all(base[k] == v for k, v in overrides.items()): return
        # Validate the full config in memory, never serialize credentials.
        type(settings).model_validate(base | overrides)
        row = dict(stage=stage, overrides=overrides, live_delta=None, cost=None,
                   verdict="unmeasured", requires_writer_replay=False)
        if prompt_cap is not None: row["input_prompt_cap"] = prompt_cap
        row["id"] = stage + "-" + digest(overrides)[:12]
        if not any(r["id"] == row["id"] for r in rows): rows.append(row)
    for cap in (4000, 8000, 16000, 32000):
        add("budget", dict(context_budget_min=cap, context_budget_max=cap, context_budget_fallback=cap), prompt_cap=cap)
    for scale in (0.75, 1.25):
        bounded = lambda fraction: round(max(settings.context_recent_fraction_min,
            min(settings.context_recent_fraction_max, fraction * scale)), 8)
        add("recent_allocation", dict(
            context_recent_fraction_ladder=[[n, bounded(f)] for n, f in settings.context_recent_fraction_ladder],
            context_recent_fraction_default=bounded(settings.context_recent_fraction_default)))
    for field in ("retrieval_bm25_candidate_limit", "retrieval_vector_candidate_limit",
                  "retrieval_chunk_candidate_limit", "codex_claim_candidate_limit"):
        for scale in (0.5, 2.0): add("candidates", {field: max(1, round(getattr(settings, field) * scale))})
    for size in (32, 64, 128): add("rerank_shortlist", dict(retrieval_rerank_candidates=size))
    for k in (30, 60, 90): add("fusion", dict(retrieval_rrf_k=k))
    for leg, weight in settings.retrieval_leg_base_weights.items():
        for scale in (0.5, 1.5):
            add("leg_weight", dict(retrieval_leg_base_weights=settings.retrieval_leg_base_weights |
                                   {leg: weight * scale}))
    for enabled in (False, True): add("rerank_mechanism", dict(retrieval_rerank_enabled=enabled))
    return rows


@contextmanager
def applied(settings, coordinate):
    """One validated planned read policy; never silently keep a partial override."""
    known = {r["id"]: r for r in coordinates(settings)}
    if coordinate.get("id") not in known or coordinate != known[coordinate["id"]]:
        raise ValueError("coordinate differs from the bound resolved policy")
    if coordinate["overrides"]:
        with patch.multiple(settings, **coordinate["overrides"]): yield
    else:
        yield


def plan(inputs, out):
    inputs, out = inputs.resolve(), out.resolve()
    if (not out.is_relative_to((ROOT / "logs").resolve()) or inputs == out
            or out.is_relative_to(inputs) or inputs.is_relative_to(out)):
        raise ValueError("tuning/input bundles must be separate private logs/ directories")
    manifest = prepare(inputs)
    from src.api.config import settings
    from scripts.z1.replay_checkpoint import code_files
    _, files = code_files()
    files = sorted(set(files + [Path(__file__), ROOT / "experiments/v3_repair/run_development.py",
                                ROOT / "scripts/z1/metric_contract.py"]))
    identity = dict(policy=POLICY, preparation=manifest["identity_sha256"],
                    code=[fingerprint(p) for p in files], settings_sha256=digest(settings.model_dump(mode="json")))
    inventory = json.loads((inputs / "settings-candidates.json").read_text())
    candidates = coordinates(settings)
    products = {"coordinates.json": dict(version="v3", policy=POLICY, rows=candidates,
        blocked_stages={"trust_activation": "independent_source_truth_required",
                        "relevance_admission": "required_content_noise_and_rejection_opponents_required",
                        "gates_style": "opposing_style_source_controls_required",
                        "lifecycle": "declared_aging_targets_schedule_required"},
        cloud_call_rule="baseline_and_at_most_two_survivors_per_stage_after_source_guards",
        interaction_rule="up_to_twelve_checks_for_three_load_bearing_knobs_after_selection"),
        "consumer-ledger.json": inventory}
    seal = dict(version="v3", identity=identity, identity_sha256=digest(identity),
        coordinate_count=len(candidates), score_ready=False, execution_ready=False,
        missing=["complete_repaired_development_state", "qualified_content_writer_source_panels",
                 "live_consumer_deltas", "resumable_coordinate_executor", "independent_judge_qualification"],
        artifacts={name: digest(value) for name, value in products.items()})
    products["manifest.json"] = seal
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".plan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, value in products.items():
            if (out / name).exists() and (out / name).read_bytes() != encoded(value) + b"\n":
                raise ValueError("tuning coordinate identity changed; preserve this bundle and use a new directory")
        for name, value in products.items(): write_or_verify(out / name, value)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sealed = plan(args.inputs, args.out)
    print(json.dumps({k: sealed[k] for k in ("version", "identity_sha256", "coordinate_count",
                                            "score_ready", "execution_ready", "missing")}, sort_keys=True))


if __name__ == "__main__": main()
