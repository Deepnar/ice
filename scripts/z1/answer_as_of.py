#!/usr/bin/env python3
"""Answer frozen v3 historical-cutoff prompts with one cloud model per arm.

The seed trace is a private, append-only record of the store *at question time*.
This script never queries the final store. Run one matched arm at a time; the
existing paired judge can compare the resulting complete-source answer files.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.lme.cloud_provider import PROFILES, TextGenerator, load_selected_env
from scripts.z1.label_review import question_family_counts, reviewed_expected_answer, validate_review
from scripts.z1.replay_checkpoint import atomic_json
from scripts.z1.replay_validation import validate_complete_replay
from scripts.z1.seed_v3 import (CORPUS, EXPECTED, PROMPT_ARMS,
                                SOURCE_TIMESTAMP_PROVENANCE, gold_fragment_coverage)


LOGS = (Path(__file__).resolve().parents[2] / "logs").resolve()
_RUN_LOCK = None


def private_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(LOGS):
        raise ValueError("v3 answers contain private source text and must stay under logs/")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_digest(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def load_probes(path: Path, *, allow_partial: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    if not rows or rows[0].get("event") != "run":
        raise ValueError("not a v3 seed trace")
    gap = rows[0].get("meta", {}).get("settings_resolved", {}).get(
        "recent_window_max_turns")
    if not allow_partial and (type(gap) is not int or gap < 1):
        raise ValueError("full trace lacks its resolved recent-history window")
    if not allow_partial:
        validate_complete_replay(rows, EXPECTED)
        if rows[0].get("meta", {}).get("extra", {}).get(
                "timestamp_provenance_by_conversation") != SOURCE_TIMESTAMP_PROVENANCE:
            raise ValueError("full trace lacks the verified corpus timestamp origins")
    probes = [r for r in rows if r.get("event") == "as_of_probe"]
    keys = set()
    for probe in probes:
        key = (probe["probe_id"], probe["conversation"], probe["split_turn"])
        if key in keys:
            raise ValueError("duplicate historical probe identity")
        keys.add(key)
        if probe["state_turns"] != probe["split_turn"]:
            raise ValueError("probe state and historical cutoff disagree")
        unlabelled = probe.get("cutoff_kind") == "native_unlabeled_checkpoint"
        if not probe.get("expected_answer"):
            raise ValueError("as-of probe lacks its expected answer")
        if unlabelled and (probe["gold_turns"] or probe["gold_sources"]):
            raise ValueError("unlabeled checkpoint has unexpected gold")
        if not allow_partial and not unlabelled and (
                probe.get("cutoff_kind") not in
                {"delayed", "native_checkpoint", "generated_at_checkpoint"}
                or not probe["gold_turns"]
                or probe["split_turn"] - max(probe["gold_turns"]) < gap):
            raise ValueError("answer campaign includes a recent-history-confounded probe")
        control_names = set(probe.get("controls", {}))
        wanted_controls = set(PROMPT_ARMS) - {"full"}
        if control_names != wanted_controls and not (
                allow_partial and control_names == {"vector_only", "recent_only"}):
            raise ValueError("as-of probe lacks matched prompt controls")
        if len(probe["gold_sources"]) != len(probe["gold_turns"]):
            raise ValueError("as-of probe lacks complete gold source")
        if any(source["turn"] > probe["split_turn"] for source in probe["gold_sources"]):
            raise ValueError("gold source lies after question time")
        for arm in ("preflight", *probe["controls"]):
            stage = (probe["preflight"] if arm == "preflight"
                     else probe["controls"][arm])
            if not stage.get("prompt_messages"):
                raise ValueError("as-of arm has no final prompt")
            if stage["lineage"]["future_source_ids"]:
                raise ValueError("future source in an as-of prompt")
    return probes


def reviewed_native_probes(trace: Path, audit_path: Path,
                           all_probes: list[dict]) -> list[dict]:
    """Attach reviewed source turns to frozen unlabeled section prompts."""
    audit = json.loads(audit_path.read_text())
    rows = [json.loads(line) for line in trace.open()]
    if (audit.get("kind") != "native_checkpoint_source_review"
            or audit.get("inputs") != rows[0]["meta"]["inputs"][:4]
            or audit.get("recent_window_turns") != rows[0]["meta"]
            .get("settings_resolved", {}).get("recent_window_max_turns")):
        raise ValueError("native source review belongs to different corpus/catalog")
    gap = rows[0]["meta"]["settings_resolved"]["recent_window_max_turns"]
    by_id = {p["probe_id"]: p for p in all_probes
             if p["cutoff_kind"] == "native_unlabeled_checkpoint"}
    written = {(r["conversation"], r["turn"]):
               {r["episodic_id"], r["batch_id"]}
               for r in rows if r.get("event") == "written"}
    source_turn_by_id = {source_id: f"{slug}:{turn}"
                         for (slug, turn), ids in written.items()
                         for source_id in ids}
    source = defaultdict(list)
    for line in CORPUS.open():
        row = json.loads(line)
        slug = row["conversation_id"][:8]
        if slug in {p["conversation"] for p in by_id.values()}:
            source[slug].append(row)
    chosen = []
    seen = set()
    for record in audit.get("records", []):
        identity = record["probe_id"]
        if identity in seen or identity not in by_id:
            raise ValueError("native review has duplicate or unknown probe")
        seen.add(identity)
        probe = by_id[identity]
        if (record["conversation"] != probe["conversation"]
                or record["cutoff_turn"] != probe["split_turn"]
                or record["question"] != probe["question"]
                or record.get("expected_answer") != probe.get("expected_answer")):
            raise ValueError("native review identity differs from replay")
        verdict = record.get("answer_verdict")
        if verdict not in ("valid", "invalid", "uncertain", None):
            raise ValueError("native review has unknown verdict")
        if verdict != "valid":
            continue
        gold = record.get("reviewed_gold_turns") or []
        cutoff = probe["split_turn"]
        if (not record.get("reason", "").strip() or not gold
                or any(type(turn) is not int for turn in gold)
                or len(gold) != len(set(gold))
                or min(gold) < 1 or max(gold) > cutoff - gap):
            raise ValueError("valid native review needs old source turns and a reason")
        if any((probe["conversation"], turn) not in written for turn in gold):
            raise ValueError("reviewed source turn is missing from trace")
        validate_review(record, cutoff)
        gold_ids = {turn: written[(probe["conversation"], turn)] for turn in gold}
        admitted = dict(probe)
        admitted["catalog_expected_answer"] = probe["expected_answer"]
        admitted["expected_answer"] = reviewed_expected_answer(record)
        admitted["gold_turns"] = sorted(gold)
        admitted["gold_sources"] = [{"turn": turn,
                                      "recorded_at": source[probe["conversation"]][turn - 1]["timestamp"],
                                      "ts_provenance": SOURCE_TIMESTAMP_PROVENANCE[probe["conversation"]],
                                      "prompt": source[probe["conversation"]][turn - 1]["prompt"],
                                      "response": source[probe["conversation"]][turn - 1]["response"],
                                      "source_ids": sorted(gold_ids[turn])}
                                     for turn in sorted(gold)]
        for stage in [admitted["preflight"], *admitted["controls"].values()]:
            stage["gold_fragment_coverage"] = gold_fragment_coverage(
                stage, gold_ids, source_turn_by_id)
        admitted["label_status"] = "reviewed_source_and_answer"
        admitted["review"] = {key: record[key] for key in
                              ("knowledge_scope", "task_types", "reviewed_through_turn",
                               "recent_only_answerable", "reason")}
        admitted["review"].update({key: record[key] for key in
                                  ("reviewer", "review_scope", "question_family_id") if key in record})
        chosen.append(admitted)
    return chosen


def stratified(probes: list[dict], n: int, seed: int) -> list[dict]:
    if not n or n >= len(probes):
        return sorted(probes, key=lambda p: p["probe_id"])
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for probe in probes:
        grouped[probe["type"]].append(probe)
    for group in grouped.values():
        group.sort(key=lambda p: p["probe_id"])
        rng.shuffle(group)
    chosen = []
    kinds = sorted(grouped)
    while len(chosen) < n:
        for kind in kinds:
            if grouped[kind] and len(chosen) < n:
                chosen.append(grouped[kind].pop())
    return chosen


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=PROMPT_ARMS, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="opencode-luna6")
    parser.add_argument("--n", type=int, default=0, help="0 means all available probes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", action="store_true", help="validate and count; no cloud call")
    parser.add_argument("--allow-partial", action="store_true",
                        help="development check only; never a complete quality result")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validated-probes", type=Path,
                        help="reviewed private long-term packet from build_longterm_label_review.py")
    parser.add_argument("--validated-native-sources", type=Path,
                        help="reviewed private packet from build_checkpoint_source_review.py")
    args = parser.parse_args()
    if args.n < 0:
        raise ValueError("--n cannot be negative")
    source, output = private_path(args.trace), private_path(args.out)
    trace_hash = sha256_file(source)
    seed_run = json.loads(source.open().readline())
    clock_policy = seed_run.get("meta", {}).get("extra", {}).get("clock_policy")
    all_probes = load_probes(source, allow_partial=args.allow_partial)
    available = [p for p in all_probes if p["gold_turns"]]
    audit_hashes = {}
    if args.validated_probes:
        audit_path = private_path(args.validated_probes)
        audit = json.loads(audit_path.read_text())
        run = json.loads(source.open().readline())
        if (audit.get("kind") != "longterm_probe_label_review"
                or audit.get("inputs") != run["meta"]["inputs"][:5]):
            raise ValueError("label audit belongs to different source/catalog inputs")
        by_id = {p["probe_id"]: p for p in available}
        valid = set()
        reviewed = set()
        for record in audit.get("records", []):
            probe_id = record["probe_id"]
            if probe_id in reviewed or probe_id not in by_id:
                raise ValueError("label audit has duplicate or unknown probe")
            reviewed.add(probe_id)
            if (record["cutoff_turn"] != by_id[probe_id]["split_turn"]
                    or record["conversation"] != by_id[probe_id]["conversation"]
                    or record["question"] != by_id[probe_id]["question"]
                    or record.get("expected_answer") !=
                    by_id[probe_id].get("expected_answer")
                    or record["gold_turns"] != by_id[probe_id]["gold_turns"]):
                raise ValueError("label audit cutoff differs from replay")
            if record.get("verdict") not in ("valid", "invalid", "uncertain", None):
                raise ValueError("label audit has unknown verdict")
            if record.get("verdict") == "valid":
                if not record.get("reason", "").strip():
                    raise ValueError("valid label needs a concrete review reason")
                validate_review(record, by_id[probe_id]["split_turn"])
                admitted = dict(by_id[probe_id])
                admitted["catalog_expected_answer"] = admitted["expected_answer"]
                admitted["expected_answer"] = reviewed_expected_answer(record)
                admitted["review"] = {key: record[key] for key in
                    ("knowledge_scope", "task_types", "reviewed_through_turn",
                     "recent_only_answerable", "reason")}
                admitted["review"].update({key: record[key] for key in
                                          ("reviewer", "review_scope", "question_family_id") if key in record})
                by_id[probe_id] = admitted
                valid.add(probe_id)
        available = [by_id[p["probe_id"]] for p in available if p["probe_id"] in valid]
        audit_hashes["existing"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    elif args.validated_native_sources or (not args.allow_partial and not args.plan):
        available = []
    if args.validated_native_sources:
        native_path = private_path(args.validated_native_sources)
        available += reviewed_native_probes(source, native_path, all_probes)
        audit_hashes["native"] = hashlib.sha256(native_path.read_bytes()).hexdigest()
    if not args.allow_partial and not args.plan and not audit_hashes:
        raise ValueError("long-term answers need reviewed labels before cloud calls")
    probes = stratified(available, args.n, args.seed)
    identifiers = [p["probe_id"] for p in probes]
    if args.arm != "full" and any(args.arm not in p["controls"] for p in probes):
        raise ValueError("selected trace did not freeze the requested prompt arm")
    if args.plan:
        print(json.dumps({"version": "v3", "arm": args.arm, "probes": len(probes),
                          "label_audit_applied": bool(audit_hashes),
                          "review_coverage": {
                              "question_families": question_family_counts([p.get("review") or {} for p in probes]),
                              "by_conversation": {slug: sum(p["conversation"] == slug for p in probes)
                                  for slug in sorted({p["conversation"] for p in probes})},
                              "checkpoint_times": len({(p["conversation"], p["split_turn"]) for p in probes}),
                              "knowledge_scopes": {scope: sum(p.get("review", {}).get("knowledge_scope") == scope
                                  for p in probes) for scope in sorted({p.get("review", {}).get("knowledge_scope", "unreviewed")
                                  for p in probes})},
                              "semantic_task_types": {task: sum(task in p.get("review", {}).get("task_types", [])
                                  for p in probes) for task in sorted({t for p in probes
                                      for t in p.get("review", {}).get("task_types", [])})}},
                          "types": {kind: sum(p["type"] == kind for p in probes)
                                    for kind in sorted({p["type"] for p in probes})},
                          "prompt_tokens": sum((p["preflight"] if args.arm == "full"
                                                else p["controls"][args.arm])["prompt_tokens"]
                                               for p in probes),
                          "matched_arms": {arm: {"probes": len(probes),
                              "prompt_tokens_est": sum((p["preflight"] if arm == "full"
                                  else p["controls"][arm])["prompt_tokens"] for p in probes)}
                              for arm in PROMPT_ARMS
                              if all(arm == "full" or arm in p["controls"] for p in probes)},
                          "recorded_answers_not_used": True}, indent=2))
        return 0
    if not probes:
        raise ValueError("no validated probes selected for cloud answering")

    output.parent.mkdir(parents=True, exist_ok=True)
    global _RUN_LOCK
    _RUN_LOCK = output.with_suffix(".lock").open("a+")
    try:
        fcntl.flock(_RUN_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("this answer stage already has an active operator") from None

    load_selected_env()
    profile = PROFILES[args.profile]
    decoding = {"max_output_tokens": 1500, "temperature": 0}
    identity = {"version": "v3", "tag": args.arm,
                "trace_sha256": trace_hash, "probe_ids": identifiers,
                "seed_clock_policy": clock_policy,
                "label_audit_sha256": audit_hashes,
                "answer_profile": profile.name, "answer_model": profile.model,
                "decoding": {**decoding, "temperature_sent": profile.supports_temperature,
                             "temperature_policy": "requested_zero" if profile.supports_temperature
                             else "provider_default"},
                "provider": profile.metadata(),
                "implementation_sha256": {
                    "answer_runner": sha256_file(Path(__file__)),
                    "cloud_adapter": sha256_file(Path(__file__).resolve().parents[2]
                                                 / "experiments/lme/cloud_provider.py")},
                "development_partial": args.allow_partial,
                "sample_seed": args.seed}
    if output.exists():
        if not args.resume:
            raise ValueError("answer output exists; use --resume or a new path")
        result = json.loads(output.read_text())
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError("resume output is from a different trace/sample/model")
    else:
        result = {**identity, "complete": False, "records": []}
        atomic_json(output, result)
    by_id = {p["probe_id"]: p for p in probes}
    observed = set()
    for row in result["records"]:
        identity_key = row["probe_id"]
        if identity_key not in by_id or identity_key in observed:
            raise ValueError("resume output has duplicate or undeclared answers")
        observed.add(identity_key)
        stage = (by_id[identity_key]["preflight"] if args.arm == "full"
                 else by_id[identity_key]["controls"][args.arm])
        if (row.get("answer_input_sha256") != input_digest(stage["prompt_messages"])
                or input_digest(row.get("answer_input_messages", [])) != row["answer_input_sha256"]):
            raise ValueError("saved answer input differs from the frozen prompt")
    done = {row["probe_id"]: row for row in result["records"]
            if row.get("answer") and not row.get("error")}
    if len(done) == len(probes):
        result["complete"] = True
        atomic_json(output, result)
        return 0
    answerer = TextGenerator(profile)
    for index, probe in enumerate(probes, 1):
        if probe["probe_id"] in done:
            continue
        prior = next((row for row in result["records"]
                      if row["probe_id"] == probe["probe_id"] and row.get("error")), None)
        if prior:
            result.setdefault("failed_attempts", []).append(prior)
        stage = (probe["preflight"] if args.arm == "full"
                 else probe["controls"][args.arm])
        source_text = "\n\n".join(
            f"Source turn {gold['turn']} at {gold['recorded_at']} "
            f"(timestamp provenance: {gold.get('ts_provenance', 'not recorded')})\n"
            f"User: {gold['prompt']}\nAssistant: {gold['response']}"
            for gold in probe["gold_sources"])
        record = {"probe_id": probe["probe_id"], "conversation": probe["conversation"],
                  "split_turn": probe["split_turn"],
                  "question_time": probe.get("question_time"),
                  "question_time_provenance": probe.get("question_time_provenance"),
                  "probe_type": probe["type"],
                  "question": probe["question"], "gold_turns": probe["gold_turns"],
                  "expected_answer": probe.get("expected_answer"),
                  "catalog_expected_answer": probe.get("catalog_expected_answer", probe.get("expected_answer")),
                  "label_review": probe.get("review"),
                  "gold_source_complete": True, "gold_turn_text": source_text,
                  "gold_source_storage": {
                      str(gold["turn"]): next((probe.get("source_storage_at_cutoff", {}).get(source_id)
                                               for source_id in gold["source_ids"]
                                               if source_id in probe.get("source_storage_at_cutoff", {})), None)
                      for gold in probe["gold_sources"]},
                  "answer_profile": profile.name, "answer_model": profile.model,
                  "prompt_tokens": stage["prompt_tokens"],
                  "selected_tokens": stage["selected_tokens"],
                  "gold_fragment_coverage": stage["gold_fragment_coverage"],
                  "answer_input_messages": stage["prompt_messages"],
                  "answer_input_sha256": input_digest(stage["prompt_messages"]),
                  "evidence_receipt": {key: stage.get(key) for key in
                                       ("gate", "selected", "source_note_ids", "lineage",
                                        "evicted", "prompt_block_counts", "context_ledger")},
                  "prompt_evidence_support": "unreviewed",
                  "answer_memory_use": "not_inferred_from_correctness_or_source_presence",
                  "answer": "", "error": None}
        try:
            answer = answerer.generate(
                stage["prompt_messages"], **decoding,
                session_id=f"ice-v3-z1-{args.arm}-{probe['probe_id']}")
            record.update(answer=answer.text, usage=answer.usage,
                          response_id=answer.response_id)
            if not answer.text:
                raise RuntimeError("cloud answer was empty")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        result["records"] = [r for r in result["records"]
                             if r["probe_id"] != probe["probe_id"]] + [record]
        atomic_json(output, result)
        print(f"{index}/{len(probes)} {probe['probe_id']} "
              f"{'failed' if record['error'] else 'answered'}", flush=True)
        if record["error"]:
            raise RuntimeError(record["error"])
    result["complete"] = (len(result["records"]) == len(probes)
                          and all(r.get("answer") and not r.get("error")
                                  for r in result["records"]))
    atomic_json(output, result)
    return 0 if result["complete"] else 1


def main() -> int:
    global _RUN_LOCK
    try:
        return _main()
    finally:
        if _RUN_LOCK is not None:
            _RUN_LOCK.close()
            _RUN_LOCK = None


if __name__ == "__main__":
    raise SystemExit(main())
