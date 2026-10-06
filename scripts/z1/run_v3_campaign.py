#!/usr/bin/env python3
"""Manual v3 campaign entry point. Default is status; --run is explicit.

Every stage has its own artifacts and resume checks. The private bundle owns
one persistent isolated database; it never targets the user's normal store.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from experiments.lme.cloud_provider import PROFILES, load_selected_env
from scripts.z1.answer_as_of import private_path, sha256_file
from scripts.z1.campaign_progress import ArtifactProgress, TerminalProgress
from scripts.z1.campaign_recovery import (
    RETRY_DELAYS,
    attempt_receipt,
    failure_evidence,
    failure_path,
    recoverable,
    stamp,
)
from scripts.z1.label_review import validate_packet, validate_review
from scripts.z1.replay_checkpoint import atomic_json
from src.api.config import settings
from src.memory.models import Base

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("seed", "snapshot", "answers", "judge", "report")


def repeat_arguments(root: Path) -> list[str]:
    path = root / "development-repeat-review.json"
    return ["--development-repeat-review", str(path)] if path.exists() else []


def progress_totals(root: Path) -> tuple[int, int]:
    from scripts.z1.seed_v3 import EXPECTED, load_plan
    repeat = root / "development-repeat-review.json"
    _, probes, _, _ = load_plan(capture_unlabeled_native=True,
                               development_repeat_review=repeat if repeat.exists() else None)
    return sum(EXPECTED.values()), sum(map(len, probes.values()))


def judge_selection(config: dict) -> dict:
    """Report the phase-four choice without API calls or loading credentials."""
    name = config.get("judge_profile")
    if name:
        profile = PROFILES[name]
        return {"profile": name, "model": profile.model, "endpoint": profile.endpoint,
                "qualification": "pending"}
    from scripts.z1.judge_answers import _env
    return {"profile": "legacy_PROBE_MODEL", "model": _env("PROBE_MODEL"),
            "endpoint": "chat_completions", "qualification": "pending"}


def schema_signature(engine, tables: list[str]) -> str:
    """Column/default, constraint and index definitions, never user rows."""
    queries = [
        """SELECT c.relname, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod),
                  a.attnotnull, pg_get_expr(d.adbin,d.adrelid), a.attidentity
           FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
           JOIN pg_attribute a ON a.attrelid=c.oid
           LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum
           WHERE n.nspname='public' AND c.relname=ANY(:tables)
             AND a.attnum>0 AND NOT a.attisdropped ORDER BY c.relname,a.attnum""",
        """SELECT tablename,indexname,indexdef FROM pg_indexes
           WHERE schemaname='public' AND tablename=ANY(:tables) ORDER BY tablename,indexname""",
        """SELECT c.relname,k.conname,k.contype,pg_get_constraintdef(k.oid)
           FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid
           JOIN pg_namespace n ON n.oid=c.relnamespace
           WHERE n.nspname='public' AND c.relname=ANY(:tables) ORDER BY c.relname,k.conname""",
    ]
    with engine.connect() as conn:
        value = [[list(row) for row in conn.execute(text(query), {"tables": tables})]
                 for query in queries]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def label_counts(path: Path, field: str) -> dict:
    rows = json.loads(path.read_text()).get("records", []) if path.exists() else []
    valid = incomplete = 0
    for row in rows:
        if row.get(field) != "valid":
            continue
        try:
            validate_review(row, row["cutoff_turn"])
            if not row.get("reason", "").strip():
                raise ValueError("missing reason")
        except (KeyError, ValueError):
            incomplete += 1
        else:
            valid += 1
    reviewed = sum(
        row.get(field) is not None
        and row.get("review_scope") in {"complete_history_through_cutoff", "complete_source_and_recent_counterexample_only"}
        and isinstance(row.get("reviewer"), str) and bool(row["reviewer"].strip())
        for row in rows)
    return {"present": path.exists(), "rows": len(rows),
            "valid": valid, "marked_valid_but_incomplete": incomplete,
            "unreviewed": sum(r.get(field) is None for r in rows),
            "invalid": sum(r.get(field) == "invalid" for r in rows),
            "uncertain": sum(r.get(field) == "uncertain" for r in rows),
            "source_answer_reviews_recorded": reviewed,
            "without_recorded_source_answer_review": len(rows) - reviewed}


def validate_labels(root: Path) -> int:
    from scripts.z1.development_repeats import KINDS
    from scripts.z1.run_meta import file_digest
    from scripts.z1.seed_v3 import CORPUS, UNIFIED, TYPED, DERIVED, GENERATED, load_plan
    repeat_path = root / "development-repeat-review.json"
    _, scheduled, _, _ = load_plan(capture_unlabeled_native=True,
                                  development_repeat_review=repeat_path if repeat_path.exists() else None)
    probes = [p for group in scheduled.values() for p in group]
    inputs = [file_digest(p) for p in (CORPUS, UNIFIED, TYPED, DERIVED, GENERATED)]
    valid = 0
    for native, filename in ((False, "labels-source-linked.json"), (True, "labels-native.json")):
        catalog = {p["probe_id"]: p for p in probes
                   if p["cutoff_kind"] not in KINDS
                   and (p["cutoff_kind"] == "native_unlabeled_checkpoint") == native}
        valid += validate_packet(json.loads((root / filename).read_text()), catalog, inputs,
                                 settings.recent_window_max_turns, native=native)
    return valid + sum(p["cutoff_kind"] in KINDS for p in probes)


def initialize(root: Path) -> dict:
    config = root / "campaign.json"
    if config.exists():
        raise ValueError("campaign already exists; use status or --run to resume it")
    from scripts.z1.build_longterm_label_review import build_packet as linked_packet
    from scripts.z1.build_checkpoint_source_review import build_packet as native_packet
    identity = uuid.uuid4().hex
    value = {"format": "ice-v3-manual-campaign-1", "campaign_id": identity,
             "database": "ice_v3_campaign_" + identity[:16],
             "snapshot_arm": "manual-v3-" + identity,
             "answer_profile": "opencode-luna6", "checkpoint_every": 1,
             "judge_profile": "opencode-muse13",
             "worker_failure_policy": "continue",
             "cloud_failure_policy": "continue",
             "background_model": "gemma4:e4b",
             "codex_extraction_model": "hf.co/numind/NuExtract3-GGUF:Q8_0",
             "probe_panel": "existing"}
    atomic_json(root / "labels-source-linked.json", linked_packet(0, 0))
    atomic_json(root / "labels-native.json", native_packet())
    atomic_json(config, value)
    return value


def read_config(root: Path) -> dict:
    config = json.loads((root / "campaign.json").read_text())
    identity = config.get("campaign_id", "")
    if (config.get("format") != "ice-v3-manual-campaign-1"
            or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity)
            or config.get("database") != "ice_v3_campaign_" + identity[:16]
            or config.get("snapshot_arm") != "manual-v3-" + identity
            or type(config.get("checkpoint_every")) is not int or config["checkpoint_every"] < 1
            or config.get("worker_failure_policy", "strict") not in {"strict", "continue"}
            or config.get("cloud_failure_policy", "strict") not in {"strict", "continue"}
            or config.get("answer_profile") not in PROFILES
            or (config.get("judge_profile") is not None and config["judge_profile"] not in PROFILES)
            or not isinstance(config.get("background_model"), str) or not config["background_model"].strip()
            or not isinstance(config.get("codex_extraction_model"), str) or not config["codex_extraction_model"].strip()
            or config.get("probe_panel") != "existing"):
        raise ValueError("invalid campaign identity or replay policy")
    return config


def status(root: Path, config: dict) -> dict:
    linked = label_counts(root / "labels-source-linked.json", "verdict")
    native = label_counts(root / "labels-native.json", "answer_verdict")
    state_path = root / "stage-status.json"
    progress = json.loads(state_path.read_text()) if state_path.exists() else {}
    trace = root / "seed.jsonl"
    repeat_path = root / "development-repeat-review.json"
    repeats = {"present": repeat_path.exists(), "families": 0, "reviewed_occurrences": 0}
    if repeat_path.exists():
        from scripts.z1.seed_v3 import load_plan
        _, schedule, _, _ = load_plan(capture_unlabeled_native=True, development_repeat_review=repeat_path)
        repeats["reviewed_occurrences"] = sum(bool(p.get("development_repeat")) for group in schedule.values() for p in group)
        repeats["families"] = repeats["reviewed_occurrences"] // 2
    labels_available = linked["valid"] + native["valid"] + repeats["reviewed_occurrences"] > 0
    missing_reviews = (linked["without_recorded_source_answer_review"]
                       + native["without_recorded_source_answer_review"])
    source_review_limit = (
        "Source/answer reviews are recorded for every row in the present packets; "
        "invalid/uncertain cases remain excluded and independent qualification is separate."
        if linked["present"] and native["present"] and linked["rows"] + native["rows"] > 0
        and missing_reviews == 0
        else "Expected-answer and source labels still need complete source/intervening/recent review."
    )
    answer_plan = {"verified": False, "reason": "complete replay and reviewed labels required"}
    if trace.exists() and labels_available:
        planned = subprocess.run([sys.executable, str(ROOT / "scripts/z1/answer_as_of.py"),
            "--trace", str(trace), "--out", str(root / "answers-full.json"),
            "--arm", "full", "--profile", config["answer_profile"], "--plan",
            "--validated-probes", str(root / "labels-source-linked.json"),
            "--validated-native-sources", str(root / "labels-native.json"), *repeat_arguments(root)],
            cwd=ROOT, capture_output=True, text=True)
        if planned.returncode:
            answer_plan = {"verified": False, "reason": "trace or label validation failed; run answer_as_of.py --plan for details"}
        else:
            plan = json.loads(planned.stdout)
            answer_plan = {"verified": plan["probes"] > 0, "arms": plan["matched_arms"],
                           "review_coverage": plan["review_coverage"]}
    return {"version": "v3", "run_directory": str(root), "stages": progress,
            "judge_selection": judge_selection(config),
            "ground_truth": {"source_linked_review": linked, "native_review": native,
                             "development_repeat_review": repeats},
            "reviewed_label_candidates_available": labels_available,
            "cloud_answers_ready": answer_plan["verified"], "answer_plan": answer_plan,
            "score_of_record": False,
            "limits": [source_review_limit,
                       "Source rank/presence does not establish semantic prompt support or actual answer use.",
                       "The four arms measure direct Codex, warm-vector and recent-history contrasts, not every leg or a standalone vector baseline.",
                       "Broader answer-judge, graph and summary judgment qualification remains pending.",
                       "Two histories have simulated dates; replay does not test async scheduling, model stickiness or project/document workflows."]}


def install_schema(root: Path, source, target) -> None:
    """Freeze current memory-table DDL only; retain production indexes/defaults."""
    from scripts.z1.snapshot import CONTAINER, USER
    template, receipt = root / "schema.sql", root / "schema.json"
    tables = sorted(Base.metadata.tables)
    if receipt.exists():
        metadata = json.loads(receipt.read_text())
        if metadata.get("tables") != tables or sha256_file(template) != metadata.get("sha256"):
            raise ValueError("campaign schema template changed; refusing attachment")
    else:
        original = create_engine(source)
        try:
            if not set(tables).issubset(inspect(original).get_table_names(schema="public")):
                raise ValueError("current production schema lacks a memory table; align it before initializing this campaign")
            signature = schema_signature(original, tables)
        finally:
            original.dispose()
        result = subprocess.run(["docker", "exec", CONTAINER, "pg_dump", "-U", USER,
            "-d", source.database, "--schema-only", "--no-owner", "--no-privileges",
            *[f"--table=public.{table}" for table in tables]], capture_output=True, check=True)
        temp = template.with_suffix(".tmp")
        with temp.open("wb") as sink:
            sink.write(result.stdout)
            sink.flush()
            os.fsync(sink.fileno())
        temp.replace(template)
        atomic_json(receipt, {"sha256": sha256_file(template), "tables": tables,
                             "schema_signature": signature,
                             "basis": "current production memory-table DDL only; no user data; migration chain unqualified"})
        metadata = json.loads(receipt.read_text())
    engine = create_engine(target)
    try:
        present = set(inspect(engine).get_table_names(schema="public"))
        if present:
            if present != set(tables):
                raise ValueError("campaign schema is incomplete or has unexpected tables; refusing replay")
            if schema_signature(engine, tables) != metadata["schema_signature"]:
                raise ValueError("campaign memory schema changed; refusing replay")
            return
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        with template.open("rb") as dump:
            subprocess.run(["docker", "exec", "-i", CONTAINER, "psql", "-U", USER,
                "-d", target.database, "-v", "ON_ERROR_STOP=1", "--single-transaction"],
                stdin=dump, capture_output=True, check=True)
        if set(inspect(engine).get_table_names(schema="public")) != set(tables):
            raise ValueError("installed campaign schema lacks a memory table")
        if schema_signature(engine, tables) != metadata["schema_signature"]:
            raise ValueError("cloned memory schema differs from the frozen production template")
    finally:
        engine.dispose()


def database_environment(config: dict, root: Path) -> dict:
    """Create or attach only the random database bearing this bundle's marker."""
    name = config["database"]
    marker = "ice-v3-campaign:" + config["campaign_id"]
    source = make_url(settings.database_url)
    admin = create_engine(source.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            row = conn.execute(text("SELECT shobj_description(oid, 'pg_database') FROM pg_database WHERE datname=:n"),
                               {"n": name}).first()
            if row is None:
                conn.execute(text(f'CREATE DATABASE "{name}"'))
                conn.execute(text(f"COMMENT ON DATABASE \"{name}\" IS '{marker}'"))
            elif row[0] != marker:
                raise ValueError("database lacks this campaign's ownership marker; refusing attachment")
    finally:
        admin.dispose()
    target = source.set(database=name)
    install_schema(root, source, target)
    return dict(os.environ, DATABASE_URL=target.render_as_string(hide_password=False),
                ICE_TEST_DATABASE=name, BACKGROUND_MODEL_MODE="shared",
                BACKGROUND_MODEL_NAME=config["background_model"],
                CODEX_EXTRACTION_MODEL=config["codex_extraction_model"],
                CODEX_EXTRACTION_NER_TIER="background")


def campaign_report(root: Path, config: dict) -> dict:
    """Reference complete private artifacts without promoting diagnostic scores."""
    result = status(root, config)
    result["artifacts"] = {}
    for path in [root / "seed.jsonl", root / "replay-report.json",
                 *sorted(root.glob("answers-*.json")), *sorted(root.glob("judge-full-vs-*.json"))]:
        if not path.exists():
            continue
        receipt = {"path": str(path), "sha256": sha256_file(path)}
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            receipt.update({key: data.get(key) for key in
                            ("complete", "complete_corpus_replay", "judge_status", "score_of_record", "question_families",
                             "memory_processing", "worker_degradation",
                             "instrument_continuations",
                             "memory_processing_strata",
                             "processing_complete", "cloud_errors", "cloud_failure_policy",
                             "calibration_status", "paired_prompt_cost", "absolute_by_type",
                             "reviewed_outcome_strata", "retention_comparison", "order_checks") if key in data})
            receipt["records"] = len(data.get("records", data.get("results", [])))
            if path.name.startswith("judge-"):
                receipt["source_grade_repetitions_are_not_independent"] = True
        result["artifacts"][path.name] = receipt
    result["ground_truth_status"] = {
        "answer_labels": "reviewed admitted questions only; inspect answer_plan.review_coverage",
        "source_rank": "source identity credit; not semantic answer-bearing-word survival",
        "prompt_semantic_support": "unreviewed; exact answering messages preserved for review",
        "answer_memory_use": "not inferred from correctness; paired controls estimate effects",
        "graph_summary_update_procedural_quality": "requires output-specific judgments against original sources",
        "judge_reliability": "qualification_pending; authored controls and three all-correct pairs are insufficient",
        "token_cost": "estimated common-tokenizer prompt counts; provider usage retained separately; no matched-quality cost curve yet",
    }
    answers = root / "answers-full.json"
    if answers.exists():
        data = json.loads(answers.read_text())
        basis = data.get("trace_sha256")
        if (root / "seed.jsonl").exists() and sha256_file(root / "seed.jsonl") != basis:
            raise ValueError("answer report belongs to a different seed trace")
        result["admitted_source_funnel"] = {"probes": len(data["records"]),
            "complete_answers": sum(bool(r.get("answer")) and not r.get("error") for r in data["records"]),
            "metrics": {key: sum(r.get("gold_fragment_coverage", {}).get(key, 0) for r in data["records"])
                for key in ("gold_turns", "generated", "rank_at_5", "rank_at_10", "source_turn_rank_at_5",
                            "source_turn_rank_at_10", "budgeted", "selected", "selected_or_source_note")},
            "basis": "reviewed answer-file source labels, not all screened candidates in replay report"}
    return result


def execute(root: Path, config: dict, stage: str) -> int:
    state_path = root / "stage-status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    trace = root / "seed.jsonl"
    chosen = STAGES if stage == "all" else (stage,)
    admitted = 0
    if stage in {"all", "answers", "judge"} and not status(root, config)["reviewed_label_candidates_available"]:
        print("BLOCKED: no reviewed valid source/answer labels. Review the private packets first; --stage seed can reconstruct the store separately.")
        return 2
    if stage in {"all", "answers", "judge"}:
        admitted = validate_labels(root)
    elif stage == "seed" and repeat_arguments(root):
        admitted = validate_labels(root)
    if any(s in {"answers", "judge"} for s in chosen):
        load_selected_env()
        if "answers" in chosen:
            PROFILES[config["answer_profile"]].resolved_base_url()
            PROFILES[config["answer_profile"]].resolved_api_key()
        if "judge" in chosen:
            from scripts.z1.judge_answers import _env
            required = ("PROBE_API_KEY", "PROBE_API_BASE_URL") + (() if config.get("judge_profile") else ("PROBE_MODEL",))
            missing = [key for key in required if not _env(key)]
            if missing:
                raise ValueError("judge configuration missing: " + ", ".join(missing))
            if config.get("judge_profile"):
                PROFILES[config["judge_profile"]].resolved_base_url()
                PROFILES[config["judge_profile"]].resolved_api_key()
    turns, probes = progress_totals(root) if "seed" in chosen else (0, 0)
    print("v3 preparing campaign; checking configuration/schema", flush=True)
    env = database_environment(config, root) if any(s in {"seed", "snapshot"} for s in chosen) else dict(os.environ)

    def call(script: str, arguments: list[str], *, arm=""):
        log_path = root / (Path(script).stem + ".log")
        print(f"v3 {script}: starting/resuming; details in {log_path}", flush=True)
        label = f"{STAGES.index(current) + 1}/{len(STAGES)} {current}" + (f" {arm}" if arm else "")
        reader = ArtifactProgress(root, current, arm=arm, turns=turns, probes=probes, admitted=admitted)
        for attempt in range(len(RETRY_DELAYS) + 1):
            before = stamp(failure_path(root, current, arm))
            attempt_receipt(root, stage=current, arm=arm, attempt=attempt + 1, event="started")
            try:
                with log_path.open("a") as output:
                    with TerminalProgress(reader, label):
                        try:
                            child_env = dict(env)
                            if current == "judge" and config.get("judge_profile"):
                                child_env["ICE_JUDGE_PROFILE"] = config["judge_profile"]
                            else:
                                child_env.pop("ICE_JUDGE_PROFILE", None)
                            subprocess.run([sys.executable, str(ROOT / "scripts/z1" / script), *arguments],
                                           cwd=ROOT, env=child_env, stdout=output, stderr=subprocess.STDOUT, check=True)
                        finally:
                            output.flush()
                            os.fsync(output.fileno())
            except subprocess.CalledProcessError:
                evidence = failure_evidence(root, current, arm, before)
                retry = attempt < len(RETRY_DELAYS) and recoverable(current, evidence)
                attempt_receipt(root, stage=current, arm=arm, attempt=attempt + 1,
                                event="retrying" if retry else "paused", failure=evidence)
                if not retry:
                    atomic_json(root / "campaign-pause.json", {
                        "stage": current, "arm": arm, "attempts": attempt + 1,
                        "failure": evidence, "log": str(log_path), "progress_retained": True})
                    raise
                delay = RETRY_DELAYS[attempt]
                print(f"v3 {current} {arm}: recoverable failure; retry {attempt + 2}/{len(RETRY_DELAYS) + 1} in {delay}s, saved progress retained",
                      flush=True)
                time.sleep(delay)
                path = failure_path(root, current, arm)
                if path is not None and path.exists() and "--resume" not in arguments:
                    arguments = [*arguments, "--resume"]
            except KeyboardInterrupt:
                attempt_receipt(root, stage=current, arm=arm, attempt=attempt + 1, event="interrupted")
                raise
            else:
                attempt_receipt(root, stage=current, arm=arm, attempt=attempt + 1, event="completed")
                (root / "campaign-pause.json").unlink(missing_ok=True)
                return

    for current in chosen:
        state[current] = "running"
        atomic_json(state_path, state)
        try:
            if current == "seed":
                call("seed_v3.py", ["--out", str(trace), "--checkpoint-every", str(config["checkpoint_every"]),
                    "--worker-failure-policy", config.get("worker_failure_policy", "strict"),
                                     "--probe-panel", config["probe_panel"],
                                     *repeat_arguments(root),
                                     *(["--resume"] if trace.exists() else [])])
            elif current == "snapshot":
                call("snapshot.py", ["save", "--arm", config["snapshot_arm"], "--trace", str(trace)])
            elif current == "answers":
                for arm in ("full", "no_codex", "vector_only", "recent_only"):
                    out = root / f"answers-{arm}.json"
                    call("answer_as_of.py", ["--trace", str(trace), "--out", str(out), "--arm", arm,
                         "--failure-policy", config.get("cloud_failure_policy", "strict"),
                         "--profile", config["answer_profile"],
                         "--validated-probes", str(root / "labels-source-linked.json"),
                         "--validated-native-sources", str(root / "labels-native.json"),
                         *repeat_arguments(root),
                         *(["--resume"] if out.exists() else [])], arm=arm)
            elif current == "judge":
                for arm in ("no_codex", "vector_only", "recent_only"):
                    out = root / f"judge-full-vs-{arm}.json"
                    call("judge_answers.py", ["--a", str(root / "answers-full.json"),
                         "--failure-policy", config.get("cloud_failure_policy", "strict"),
                         "--b", str(root / f"answers-{arm}.json"), "--out", str(out),
                         *(["--resume"] if out.exists() or out.with_suffix(".partial.json").exists() else [])], arm=arm)
            elif current == "report":
                call("report_v3_replay.py", [str(trace), "--out", str(root / "replay-report.json")])
            state[current] = "complete_diagnostic"
            atomic_json(state_path, state)
            if current == "report":
                atomic_json(root / "campaign-report.json", campaign_report(root, config))
        except (Exception, KeyboardInterrupt) as exc:
            state[current] = "interrupted_or_failed_resume_required"
            atomic_json(state_path, state)
            if not isinstance(exc, subprocess.CalledProcessError):
                atomic_json(root / "campaign-pause.json", {
                    "stage": current, "error_type": type(exc).__name__,
                    "progress_retained": True, "operator_review_required": True})
            raise
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="private campaign bundle under logs/")
    parser.add_argument("--init", action="store_true", help="prepare config + complete review packets, no seed or API calls")
    parser.add_argument("--run", action="store_true", help="explicitly execute/resume the selected stages")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    if args.init and args.run:
        parser.error("initialize and review first; --init cannot launch a run")
    root = private_path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("this campaign already has an active operator") from None
        config = initialize(root) if args.init else read_config(root)
        if args.run:
            try:
                return execute(root, config, args.stage)
            except subprocess.CalledProcessError as exc:
                script = Path(exc.cmd[1]).stem
                print(f"v3 campaign paused (exit {exc.returncode}). Details: {root / (script + '.log')}\n"
                      "Saved artifacts retained. See campaign-pause.json for the reason; resolve persistent errors before resuming the same command.",
                      file=sys.stderr, flush=True)
                return 1
            except KeyboardInterrupt:
                print("\nv3 interrupted. Saved artifacts retained; rerun the same command to resume.",
                      file=sys.stderr, flush=True)
                return 130
            except Exception as exc:
                print(f"v3 stopped: {exc}\nSaved artifacts retained; check configuration and stage logs before resuming.",
                      file=sys.stderr, flush=True)
                return 1
        print(json.dumps(status(root, config), indent=2))
        return 0


def main() -> int:
    """Configuration/lock failures also leave a readable operator boundary."""
    try:
        return _main()
    except KeyboardInterrupt:
        print("\nv3 interrupted; existing artifacts retained.", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"v3 cannot start/resume: {type(exc).__name__}: {exc}\n"
              "Existing artifacts retained; resolve this error before retrying.",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
