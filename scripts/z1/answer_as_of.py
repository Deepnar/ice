#!/usr/bin/env python3
"""Answer frozen v3 historical-cutoff prompts with one cloud model per arm.

The seed trace is a private, append-only record of the store *at question time*.
This script never queries the final store. Run one matched arm at a time; the
existing paired judge can compare the resulting complete-source answer files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.lme.cloud_provider import PROFILES, TextGenerator, load_selected_env


LOGS = (Path(__file__).resolve().parents[2] / "logs").resolve()


def private_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(LOGS):
        raise ValueError("v3 answers contain private source text and must stay under logs/")
    return resolved


def load_probes(path: Path, *, allow_partial: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    if not rows or rows[0].get("event") != "run":
        raise ValueError("not a v3 seed trace")
    completed = next((r for r in rows if r.get("event") == "complete"
                      and r.get("complete_selected_corpus")), None)
    if not allow_partial and completed is None:
        raise ValueError("full answer campaign needs a complete selected-corpus seed")
    probes = [r for r in rows if r.get("event") == "as_of_probe"]
    if not allow_partial and (not probes or len(probes) != completed["expected_as_of_probes"]):
        raise ValueError("full answer campaign needs every declared as-of probe")
    keys = set()
    for probe in probes:
        key = (probe["probe_id"], probe["conversation"], probe["split_turn"])
        if key in keys:
            raise ValueError("duplicate historical probe identity")
        keys.add(key)
        if probe["state_turns"] != probe["split_turn"]:
            raise ValueError("probe state and historical cutoff disagree")
        if not allow_partial and (probe.get("cutoff_kind") not in
                                  {"delayed", "native_checkpoint",
                                   "generated_at_checkpoint"}
                                  or probe["split_turn"] - max(probe["gold_turns"]) < 40):
            raise ValueError("answer campaign includes a recent-history-confounded probe")
        if set(probe.get("controls", {})) != {"vector_only", "recent_only"}:
            raise ValueError("as-of probe lacks matched prompt controls")
        if len(probe["gold_sources"]) != len(probe["gold_turns"]):
            raise ValueError("as-of probe lacks complete gold source")
        if any(source["turn"] > probe["split_turn"] for source in probe["gold_sources"]):
            raise ValueError("gold source lies after question time")
        for arm in ("preflight", "vector_only", "recent_only"):
            stage = (probe["preflight"] if arm == "preflight"
                     else probe["controls"][arm])
            if not stage.get("prompt_messages"):
                raise ValueError("as-of arm has no final prompt")
            if stage["lineage"]["future_source_ids"]:
                raise ValueError("future source in an as-of prompt")
    return probes


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


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=["full", "vector_only", "recent_only"], required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="opencode-luna6")
    parser.add_argument("--n", type=int, default=0, help="0 means all available probes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", action="store_true", help="validate and count; no cloud call")
    parser.add_argument("--allow-partial", action="store_true",
                        help="development check only; never a complete quality result")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validated-probes", type=Path,
                        help="reviewed private long-term packet from build_longterm_label_review.py")
    args = parser.parse_args()
    if args.n < 0:
        raise ValueError("--n cannot be negative")
    source, output = private_path(args.trace), private_path(args.out)
    trace_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    available = load_probes(source, allow_partial=args.allow_partial)
    audit_hash = None
    if args.validated_probes:
        audit_path = private_path(args.validated_probes)
        audit = json.loads(audit_path.read_text())
        run = json.loads(source.open().readline())
        if (audit.get("kind") != "longterm_probe_label_review"
                or audit.get("inputs") != run["meta"]["inputs"]):
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
                    or record["conversation"] != by_id[probe_id]["conversation"]):
                raise ValueError("label audit cutoff differs from replay")
            if record.get("verdict") not in ("valid", "invalid", "uncertain", None):
                raise ValueError("label audit has unknown verdict")
            if record.get("verdict") == "valid":
                if not record.get("reason", "").strip():
                    raise ValueError("valid label needs a concrete review reason")
                valid.add(probe_id)
        available = [p for p in available if p["probe_id"] in valid]
        audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    elif not args.allow_partial and not args.plan:
        raise ValueError("long-term expected answers need --validated-probes before cloud calls")
    probes = stratified(available, args.n, args.seed)
    identifiers = [p["probe_id"] for p in probes]
    if args.plan:
        print(json.dumps({"version": "v3", "arm": args.arm, "probes": len(probes),
                          "label_audit_applied": bool(args.validated_probes),
                          "types": {kind: sum(p["type"] == kind for p in probes)
                                    for kind in sorted({p["type"] for p in probes})},
                          "prompt_tokens": sum((p["preflight"] if args.arm == "full"
                                                else p["controls"][args.arm])["prompt_tokens"]
                                               for p in probes),
                          "recorded_answers_not_used": True}, indent=2))
        return 0
    if not probes:
        raise ValueError("no validated probes selected for cloud answering")

    profile = PROFILES[args.profile]
    identity = {"version": "v3", "tag": args.arm,
                "trace_sha256": trace_hash, "probe_ids": identifiers,
                "label_audit_sha256": audit_hash,
                "answer_profile": profile.name, "answer_model": profile.model,
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
    done = {row["probe_id"]: row for row in result["records"] if row.get("answer")}
    load_selected_env()
    answerer = TextGenerator(profile)
    for index, probe in enumerate(probes, 1):
        if probe["probe_id"] in done:
            continue
        stage = (probe["preflight"] if args.arm == "full"
                 else probe["controls"][args.arm])
        source_text = "\n\n".join(
            f"User: {gold['prompt']}\nAssistant: {gold['response']}"
            for gold in probe["gold_sources"])
        record = {"probe_id": probe["probe_id"], "conversation": probe["conversation"],
                  "split_turn": probe["split_turn"], "probe_type": probe["type"],
                  "question": probe["question"], "gold_turns": probe["gold_turns"],
                  "gold_source_complete": True, "gold_turn_text": source_text,
                  "answer_profile": profile.name, "answer_model": profile.model,
                  "prompt_tokens": stage["prompt_tokens"],
                  "selected_tokens": stage["selected_tokens"],
                  "gold_fragment_coverage": stage["gold_fragment_coverage"],
                  "answer": "", "error": None}
        try:
            answer = answerer.generate(
                stage["prompt_messages"], max_output_tokens=1500,
                temperature=0,
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


if __name__ == "__main__":
    raise SystemExit(main())
