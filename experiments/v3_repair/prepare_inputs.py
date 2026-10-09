"""Freeze private v3 development/final inputs, without replay or model calls."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.settings_inventory import build_inventory
from scripts.z1.label_review import reviewed_expected_answer, validate_review

DEVELOPMENT = {"ecc64aab": 251, "355a5709": 101}
FINAL = {
    "B": dict(prefix="bb558b5f", turns=1119, families=91, occurrences=638,
              checkpoints=20, archives=["EC-961862eb-FULL.json"]),
    "C": dict(prefix="a77c15cf", turns=325, families=37, occurrences=154,
              checkpoints=10, archives=["EC-a77c15cf-TURN117.json",
                                       "EC-a77c15cf-TURN172.json", "EC-a77c15cf-TURN238.json"]),
}
POLICY = dict(version="v3", method="offline_repaired_inputs_v1",
              final_histories=["B", "C"], tuning_excludes_final_histories=True,
              final_labels="legacy_unqualified", long_run_launched=False)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def fingerprint(path, root=ROOT):
    return dict(path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def load_corpus(path, expected):
    groups = {prefix: [] for prefix in expected}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        prefix = row.get("conversation_id", "")[:8]
        if prefix not in groups:
            continue
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("conversation_id", "prompt", "response", "timestamp")):
            raise ValueError(f"incomplete original pair at corpus line {line_number}")
        stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError(f"unknown clock timezone at corpus line {line_number}")
        groups[prefix].append(dict(row, turn_number=len(groups[prefix]) + 1,
                                   corpus_line=line_number, original_pair_sha256=digest(row)))
    for prefix, count in expected.items():
        rows = groups[prefix]
        if len(rows) != count or len({r["conversation_id"] for r in rows}) != 1:
            raise ValueError(f"fixture identity/count differs for {prefix}")
        stamps = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in rows]
        if any(a >= b for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"nonordered source clock for {prefix}")
        for row in rows:
            row["timestamp_provenance"] = ("original" if prefix == "355a5709"
                                            else "synthetic_raw_import")
    return groups


def development_keys(review, groups):
    selected, excluded = [], Counter()
    for section, verdict_key, gold_key in (("native", "answer_verdict", "reviewed_gold_turns"),
                                           ("source_linked", "verdict", "gold_turns")):
        for row in review[section]["records"]:
            prefix = row["conversation"][:8]
            if prefix not in DEVELOPMENT:
                continue
            status = row.get(verdict_key)
            if status != "valid":
                excluded[status or "unset"] += 1
                continue
            cutoff = row["cutoff_turn"]
            validate_review(row, cutoff)
            gold = row.get(gold_key)
            if (type(cutoff) is not int or not 1 <= cutoff <= len(groups[prefix])
                    or not isinstance(gold, list) or not gold
                    or any(type(t) is not int or not 1 <= t <= cutoff for t in gold)
                    or len(gold) != len(set(gold))):
                raise ValueError("invalid reviewed development source/cutoff")
            group = row.get("question_family_id") or "conversation:" + prefix
            selected.append(dict(review=row, reviewed_answer=reviewed_expected_answer(row),
                                 gold_turns=gold, partition_group=group))
    identities = [(r["review"]["conversation"], r["review"]["probe_id"],
                   r["review"]["cutoff_turn"]) for r in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate reviewed development occurrence")
    family_groups = sorted({r["partition_group"] for r in selected}, key=digest)
    validation = set(family_groups[:max(1, len(family_groups) // 5)]) if len(family_groups) > 1 else set()
    for row in selected:
        row["partition"] = "validation" if row["partition_group"] in validation else "tuning"
    return dict(records=selected, excluded=dict(excluded),
                partition_policy="declared_family_or_conservative_conversation_group",
                validation_available=bool(validation), untouched_external_test=False)


def verify_review_inputs(review, root):
    """An old review cannot certify changed original/catalog bytes."""
    bound = {}
    for section in ("native", "source_linked"):
        inputs = review[section].get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("review has no original/catalog binding")
        for item in inputs:
            path = (root / item["path"]).resolve()
            expected = item.get("sha256")
            if (not path.is_relative_to(root.resolve()) or not isinstance(expected, str)
                    or len(expected) < 16 or len(expected) > 64
                    or any(c not in "0123456789abcdef" for c in expected)):
                raise ValueError("invalid review input identity")
            actual = fingerprint(path, root)
            if actual["bytes"] != item.get("bytes") or not actual["sha256"].startswith(expected):
                raise ValueError("review source/catalog bytes changed: " + actual["path"])
            bound[actual["path"]] = path
        if "data/simulation/simulation_full.jsonl" not in {i["path"] for i in inputs}:
            raise ValueError("review does not bind the selected original corpus")
    return list(bound.values())


def archive_coverage(rows, paths):
    covered, mismatches, outside = set(), [], 0
    for path in paths:
        data = json.loads(path.read_text())
        for old in data["historical_context_block"] + data["future_reference_block"]:
            number = old["turn_number"]
            if type(number) is not int or number < 1:
                raise ValueError("invalid historical source index")
            if number > len(rows):
                outside += 1
                continue
            current = rows[number - 1]
            if old["user_input"] == current["prompt"] and old["ai_response"] == current["response"]:
                covered.add(number)
            else:
                mismatches.append(dict(archive=path.name, turn=number))
    return dict(exact_matching_pairs=len(covered), total_pairs=len(rows),
                unmatched_pairs=len(rows) - len(covered), mismatches=mismatches,
                excluded_after_endpoint=outside,
                historical_full_byte_identity_confirmed=len(covered) == len(rows) and not mismatches)


def legacy_references(results, groups):
    records, summaries = [], {}
    for label, definition in FINAL.items():
        prefix = definition["prefix"]
        rows = [r for r in results if r["conversation_id"].startswith(prefix)]
        counts = (len(rows), len({r["probe_id"] for r in rows}),
                  len({r["turn_index"] for r in rows}))
        if counts != (definition["occurrences"], definition["families"], definition["checkpoints"]):
            raise ValueError(f"paper {label} observation/family/cutoff counts differ")
        seen = set()
        for row in rows:
            cutoff = row["turn_index"]
            identity = (row["probe_id"], cutoff)
            if (identity in seen or type(cutoff) is not int or not 1 <= cutoff <= len(groups[prefix])
                    or not isinstance(row.get("question"), str) or not row["question"].strip()
                    or not isinstance(row.get("ground_truth"), str) or not row["ground_truth"].strip()):
                raise ValueError(f"invalid/duplicate paper {label} legacy reference")
            seen.add(identity)
            records.append(dict(history=label, probe_id=row["probe_id"], cutoff_turn=cutoff,
                                question=row["question"], legacy_reference_answer=row["ground_truth"],
                                status="legacy_unqualified", reviewed_gold_turns=None,
                                reviewed_through_turn=None))
        summaries[label] = dict(original_pairs=len(groups[prefix]), legacy_occurrences=counts[0],
                                legacy_families=counts[1], observed_cutoffs=sorted({r["turn_index"] for r in rows}))
    return dict(records=records, score_ready=False), summaries


def oracle_keys(path):
    records = json.loads(path.read_text())
    if not isinstance(records, list) or len(records) != 500:
        raise ValueError("oracle must bind the complete 500-question release")
    seen, keys = set(), []
    for index, row in enumerate(records):
        qid = row.get("question_id")
        sessions, ids = row.get("haystack_sessions"), row.get("haystack_session_ids")
        dates, gold = row.get("haystack_dates"), row.get("answer_session_ids")
        if (not isinstance(qid, str) or not qid or qid in seen
                or not isinstance(sessions, list) or not isinstance(ids, list)
                or not isinstance(dates, list) or not isinstance(gold, list)
                or len(sessions) != len(ids) or len(dates) != len(ids)
                or len(set(ids)) != len(ids) or not set(gold).issubset(ids)
                or not isinstance(row.get("question"), str) or not row["question"].strip()
                or not (type(row.get("answer")) is int
                        or (isinstance(row.get("answer"), str) and row["answer"].strip()))
                or not isinstance(row.get("question_date"), str) or not row["question_date"].strip()
                or not all(isinstance(s, str) and s for s in ids + dates + gold)):
            raise ValueError(f"invalid oracle question/source identity at index {index}")
        if any(not isinstance(session, list) or not session
               or any(not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}
                      or not isinstance(message.get("content"), str) or not message["content"].strip()
                      for message in session) for session in sessions):
            raise ValueError(f"invalid oracle original-role content at index {index}")
        seen.add(qid)
        keys.append(dict(question_id=qid, input_index=index, official_answer=row["answer"],
                         answer_session_ids=gold, available_session_ids=ids,
                         label_origin="benchmark_reference", execution_qualified=False))
    return dict(version="v3", condition="oracle", questions=keys,
                full_s_included=False, runtime_or_judge_qualified=False)


def write_or_verify(path, value):
    """An existing immutable artifact is never overwritten, even before the seal."""
    content = encoded(value) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable preparation mismatch: {path.name}; use a new directory")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare(out, *, root=ROOT):
    out = out.resolve()
    if not out.is_relative_to((root / "logs").resolve()):
        raise ValueError("private preparation must stay under logs/")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        corpus = root / "data/simulation/simulation_full.jsonl"
        review_path = root / "logs/z1-v3-manual-2026-10-04-r4/source-reviewed-labels-2026-10-03-r3.json"
        result_path = root / "experiments/mature/intermediates/master_results.json"
        oracle_path = root / "experiments/lme/data/longmemeval_oracle"
        expected = DEVELOPMENT | {d["prefix"]: d["turns"] for d in FINAL.values()}
        groups = load_corpus(corpus, expected)
        review = json.loads(review_path.read_text())
        review_inputs = verify_review_inputs(review, root)
        keys = development_keys(review, groups)
        references, summaries = legacy_references(
            json.loads(result_path.read_text())["evaluation_run_results"], groups)
        oracle = oracle_keys(oracle_path)
        archives = {label: [root / "experiments/curation_files" / name for name in d["archives"]]
                    for label, d in FINAL.items()}
        for label, paths in archives.items():
            summaries[label]["historical_source_coverage"] = archive_coverage(groups[FINAL[label]["prefix"]], paths)
        from src.api.config import Settings, settings
        inventory = build_inventory(root, Settings, settings)
        inputs = sorted(set([corpus, review_path, result_path, oracle_path] + review_inputs
                            + [p for paths in archives.values() for p in paths]))
        code = sorted((root / "src").rglob("*.py")) + [
            root / "experiments/v3_repair/prepare_inputs.py",
            root / "experiments/v3_repair/settings_inventory.py",
            root / "scripts/z1/label_review.py"]
        identity = dict(policy=POLICY, inputs=[fingerprint(p, root) for p in inputs],
                        code=[fingerprint(p, root) for p in code], inventory_sha256=digest(inventory))
        artifacts = {
            "development-originals.json": {p: groups[p] for p in DEVELOPMENT},
            "development-keys.json": keys, "final-legacy-references.json": references,
            "settings-candidates.json": inventory, "lme-oracle-keys.json": oracle,
            **{f"final-originals-{label}.json": groups[d["prefix"]] for label, d in FINAL.items()},
        }
        manifest = dict(identity=identity, identity_sha256=digest(identity), histories=summaries,
                        development_original_pairs=sum(DEVELOPMENT.values()),
                        development_reviewed_occurrences=len(keys["records"]),
                        final_original_pairs=sum(d["turns"] for d in FINAL.values()),
                        oracle_questions=len(oracle["questions"]),
                        score_ready=False, replay_ready=False,
                        artifacts={name: hashlib.sha256(encoded(value) + b"\n").hexdigest()
                                   for name, value in artifacts.items()})
        seal = out / "manifest.json"
        if seal.exists() and seal.read_bytes() != encoded(manifest) + b"\n":
            raise ValueError("preparation inputs/code/settings changed; use a new directory")
        # Verify every existing product before writing any missing one.
        for name, value in artifacts.items():
            if (out / name).exists() and (out / name).read_bytes() != encoded(value) + b"\n":
                raise ValueError(f"immutable preparation mismatch: {name}")
        for name, value in artifacts.items():
            write_or_verify(out / name, value)
        write_or_verify(seal, manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.out)
    except (ValueError, OSError) as exc:
        print(f"v3 preparation stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(dict(version="v3", output=str(args.out.resolve()),
                          development_pairs=result["development_original_pairs"],
                          development_keys=result["development_reviewed_occurrences"],
                          final_pairs=result["final_original_pairs"],
                          oracle_questions=result["oracle_questions"],
                          histories=result["histories"], score_ready=False, replay_started=False), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
