#!/usr/bin/env python3
"""Calibrate cloud answerers on public LongMemEval oracle evidence.

This is a bounded model-selection instrument, not a paper result.  It gives every
candidate the same complete evidence-only prompt, then applies LongMemEval's
official decision prompt through a separately selected judge profile.  Artifacts
are atomic and resumable at (question, model) granularity.

The default answerer set deliberately excludes Muse because Muse is the judge
candidate with the strongest existing human-labelled calibration.  A model must
not judge its own answers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

from cloud_provider import PROFILES, TextGenerator, get_profile, load_selected_env
from lme_run import DATA_DIR, instance_layout
from score import get_anscheck_prompt, judge_selftest


DEFAULT_OUT = HARNESS_DIR / "runs" / "model-calibration-v1"
DEFAULT_ANSWERERS = (
    "opencode-luna",
    "opencode-omen-alpha",
    "opencode-mimo25",
    "opencode-longcat20",
    "opencode-qwen38-flash",
    "opencode-deepseek-v4-flash",
)


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def answer_matches_profile(record: dict | None, profile) -> bool:
    if not isinstance(record, dict):
        return False
    provider = record.get("provider") or {}
    if provider.get("profile") != profile.name:
        return False
    if provider.get("model") != profile.model:
        return False
    if record.get("status") == "mute":
        return False
    return bool((record.get("answer") or "").strip())


def judgement_matches_profile(record: dict | None, profile) -> bool:
    if not isinstance(record, dict):
        return False
    judge = record.get("judge") or {}
    return (
        judge.get("profile", record.get("provider_profile")) == profile.name
        and judge.get("model", record.get("judge_model")) == profile.model
        and judge.get("endpoint", record.get("provider_endpoint")) == profile.endpoint
    )


def select_stratified(corpus: list[dict], per_type: int) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for instance in sorted(corpus, key=lambda x: x["question_id"]):
        group = ("abstention" if instance["question_id"].endswith("_abs")
                 else instance["question_type"])
        by_type[group].append(instance)
    selected = []
    for question_type in sorted(by_type):
        selected.extend(by_type[question_type][:per_type])
    return selected


def direct_oracle_messages(instance: dict) -> list[dict[str, str]]:
    """Complete evidence, preserving LongMemEval's session chronology."""
    layout = instance_layout(instance, "oracle")
    sections = []
    for index, session in enumerate(layout["sessions"], 1):
        turns = []
        for user, assistant in session["pairs"]:
            turns.append(f"User: {user}\nAssistant: {assistant}")
        sections.append(
            f"[Session {index}; date={session.get('raw_date') or 'unknown'}]\n"
            + "\n\n".join(turns)
        )
    context = "\n\n".join(sections)
    prompt = (
        "Answer the question using only the supplied conversation history. "
        "Give the answer directly and do not discuss these instructions.\n\n"
        f"Conversation history:\n{context}\n\n"
        f"Question: {instance['question']}"
    )
    return [{"role": "user", "content": prompt}]


def answer_path(root: Path, profile: str, qid: str) -> Path:
    return root / "answers" / profile / f"{qid}.json"


def judgement_path(root: Path, profile: str, qid: str) -> Path:
    return root / "judgements" / profile / f"{qid}.json"


def generate_one(root: Path, profile_name: str, instance: dict,
                 max_tokens: int) -> dict:
    path = answer_path(root, profile_name, instance["question_id"])
    profile = get_profile(profile_name)
    existing = read_json(path)
    if existing and answer_matches_profile(existing, profile):
        return existing
    generator = TextGenerator(profile)
    started = time.time()
    provider_session_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"ice-lme-cal-answer:{profile.name}:{instance['question_id']}",
    ))
    try:
        result = generator.generate(
            direct_oracle_messages(instance),
            temperature=0,
            max_output_tokens=max_tokens,
            session_id=provider_session_id,
        )
        payload = {
            "status": "complete" if result.text else "mute",
            "question_id": instance["question_id"],
            "question_type": instance["question_type"],
            "is_abstention": instance["question_id"].endswith("_abs"),
            "question": instance["question"],
            "reference_answer": instance.get("answer"),
            "answer": result.text,
            "provider": profile.metadata(),
            "provider_response_id": result.response_id,
            "provider_usage": result.usage,
            "provider_session_id": provider_session_id,
            "seconds": round(time.time() - started, 2),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:  # noqa: BLE001 - failure is calibration output
        payload = {
            "status": "error",
            "question_id": instance["question_id"],
            "question_type": instance["question_type"],
            "provider": profile.metadata(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "seconds": round(time.time() - started, 2),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    atomic_write(path, payload)
    return payload


def judge_one(root: Path, judge: TextGenerator, answer: dict,
              max_tokens: int) -> dict:
    profile_name = answer["provider"]["profile"]
    qid = answer["question_id"]
    path = judgement_path(root, profile_name, qid)
    existing = read_json(path)
    if existing and existing.get("status") == "complete":
        if judgement_matches_profile(existing, judge.profile):
            if existing.get("spoke") or existing.get("judge_raw"):
                return existing
        else:
            raise RuntimeError(
                f"judgement artifact {path} belongs to a different judge profile"
            )
        return existing
    prompt = get_anscheck_prompt(
        answer["question_type"],
        answer["question"],
        answer["reference_answer"],
        answer["answer"],
        abstention=answer.get("is_abstention", False),
    )
    started = time.time()
    provider_session_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"ice-lme-cal-judge:{judge.profile.name}:{profile_name}:{qid}",
    ))
    try:
        result = judge.generate(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_output_tokens=max_tokens,
            session_id=provider_session_id,
        )
        raw = result.text.strip()
        payload = {
            "status": "complete",
            "question_id": qid,
            "question_type": answer["question_type"],
            "answer_profile": profile_name,
            "judge": judge.profile.metadata(),
            "judge_model": judge.profile.model,
            "provider_profile": judge.profile.name,
            "provider_endpoint": judge.profile.endpoint,
            "judge_raw": raw,
            "spoke": bool(raw),
            "label": "yes" in raw.lower(),
            "provider_response_id": result.response_id,
            "provider_usage": result.usage,
            "provider_session_id": provider_session_id,
            "seconds": round(time.time() - started, 2),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:  # noqa: BLE001 - failure is calibration output
        payload = {
            "status": "error",
            "question_id": qid,
            "answer_profile": profile_name,
            "judge": judge.profile.metadata(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "seconds": round(time.time() - started, 2),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    atomic_write(path, payload)
    return payload


def usage_output_tokens(record: dict) -> int | None:
    usage = record.get("provider_usage") or {}
    value = usage.get("output_tokens", usage.get("completion_tokens"))
    return value if isinstance(value, int) else None


def print_report(root: Path, profiles: tuple[str, ...]) -> None:
    print("\nprofile                       answers judged correct spoken  ans_s judge_s out_tok")
    print("-" * 88)
    for profile in profiles:
        answers = [read_json(p) for p in sorted((root / "answers" / profile).glob("*.json"))]
        answers = [x for x in answers if x]
        judgements = [read_json(p) for p in sorted((root / "judgements" / profile).glob("*.json"))]
        judgements = [x for x in judgements if x and x.get("status") == "complete"]
        spoken = [x for x in judgements if x.get("spoke")]
        correct = sum(bool(x.get("label")) for x in spoken)
        answer_seconds = [x["seconds"] for x in answers if isinstance(x.get("seconds"), (int, float))]
        judge_seconds = [x["seconds"] for x in judgements if isinstance(x.get("seconds"), (int, float))]
        tokens = [usage_output_tokens(x) for x in answers]
        tokens = [x for x in tokens if x is not None]
        print(
            f"{profile:29} {len(answers):7d} {len(judgements):6d} "
            f"{correct:7d} {len(spoken):6d} "
            f"{statistics.mean(answer_seconds) if answer_seconds else 0:6.1f} "
            f"{statistics.mean(judge_seconds) if judge_seconds else 0:7.1f} "
            f"{statistics.mean(tokens) if tokens else 0:7.0f}"
        )
    print(f"\nartifacts: {root}")
    print("⚠ Calibration only. Read the answers and hand-check judge disagreements "
          "before selecting a model.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answer-profile", action="append", dest="answer_profiles",
                    choices=sorted(PROFILES),
                    help="repeat for multiple candidates; defaults to five non-Muse profiles")
    ap.add_argument("--judge-profile", default="opencode-muse13",
                    choices=sorted(PROFILES))
    ap.add_argument("--per-type", type=int, default=1)
    ap.add_argument("--answer-max-tokens", type=int, default=2048)
    ap.add_argument("--judge-max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stage", choices=("answer", "judge", "all"), default="all")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    profiles = tuple(args.answer_profiles or DEFAULT_ANSWERERS)
    judge_profile = get_profile(args.judge_profile)
    if (args.judge_profile in profiles
            or any(get_profile(name).model == judge_profile.model for name in profiles)):
        print("⛔ the judge model cannot also be an answerer", file=sys.stderr)
        return 2
    load_selected_env()
    corpus_path = DATA_DIR / "longmemeval_oracle"
    if not corpus_path.is_file():
        print(f"⛔ missing {corpus_path}", file=sys.stderr)
        return 1
    corpus = json.loads(corpus_path.read_text())
    selected = select_stratified(corpus, args.per_type)
    groups = {
        "abstention" if x["question_id"].endswith("_abs") else x["question_type"]
        for x in selected
    }
    print(f"selected {len(selected)} oracle questions across {len(groups)} groups")

    # Refuse to resume a directory under a changed model/profile. A directory is
    # an experiment identity, not a cache that can silently mix generations.
    mismatches = []
    for profile_name in profiles:
        profile = get_profile(profile_name)
        for path in (args.out / "answers" / profile_name).glob("*.json"):
            record = read_json(path)
            if record and record.get("status") in ("complete", "mute", "error"):
                provider = record.get("provider") or {}
                if provider and (
                    provider.get("profile") != profile.name
                    or provider.get("model") != profile.model
                    or provider.get("endpoint") != profile.endpoint
                ):
                    mismatches.append(path)
    for path in args.out.glob("judgements/*/*.json"):
        record = read_json(path)
        if record and record.get("status") == "complete" and not judgement_matches_profile(
                record, judge_profile):
            mismatches.append(path)
    if mismatches:
        print(
            "⛔ calibration artifacts use a different model/profile; refusing to "
            f"mix them. First files: {', '.join(str(p) for p in mismatches[:5])}",
            file=sys.stderr,
        )
        return 5

    if args.stage in ("answer", "all"):
        jobs = [(profile, instance) for profile in profiles for instance in selected]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_one, args.out, profile, instance,
                            args.answer_max_tokens): (profile, instance["question_id"])
                for profile, instance in jobs
            }
            for index, future in enumerate(as_completed(futures), 1):
                profile, qid = futures[future]
                result = future.result()
                print(f"answer {index:3d}/{len(jobs)} {profile:29} {qid} "
                      f"{result['status']} {result.get('seconds', 0):.1f}s", flush=True)

    if args.stage in ("judge", "all"):
        judge = TextGenerator(judge_profile)
        if not judge_selftest(judge, args.judge_max_tokens):
            return 3
        answer_records = []
        for profile in profiles:
            for instance in selected:
                record = read_json(answer_path(args.out, profile, instance["question_id"]))
                if record and record.get("status") == "complete":
                    answer_records.append(record)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(judge_one, args.out, judge, record,
                            args.judge_max_tokens): (
                    record["provider"]["profile"], record["question_id"]
                )
                for record in answer_records
            }
            for index, future in enumerate(as_completed(futures), 1):
                profile, qid = futures[future]
                result = future.result()
                print(f"judge  {index:3d}/{len(futures)} {profile:29} {qid} "
                      f"{result['status']} {result.get('seconds', 0):.1f}s", flush=True)

    print_report(args.out, profiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
