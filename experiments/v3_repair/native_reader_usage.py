"""Offline source-exposure bridge for completed native chronological v3 stores.

No store restore, model call or re-preparation. Source lineage is not an exact
derived artifact ID, an eligibility denominator or evidence of answer use.
"""

import argparse
import fcntl
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair import catalog_store
from experiments.v3_repair import chronological_sources as sources
from experiments.v3_repair.artifact_inventory import validate as validate_inventory
from experiments.v3_repair.prepare_inputs import digest, write_or_verify
from experiments.v3_repair.report_reader_usage import ARMS, count_usage
from scripts.z1.replay_checkpoint import prefix_digest

POLICY = "ice-v3-native-source-exposure-frequency-1"
PARENT = "ice-v3-native-chronological-parent-1"


def adapt(rows, *, root=None):
    """Validate the native event stream before translating its measured stages."""
    if not rows or rows[0].get("event") != "chronological_meta":
        raise ValueError("native chronological metadata missing")
    meta = rows[0]
    identity, plan = meta["identity"], sources.validate(meta["source_plan"])
    if (meta.get("version") != "v3" or meta.get("score_of_record") is not False
            or identity.get("format") != PARENT or identity.get("condition") != "full"
            or identity.get("source_plan") != plan["identity_sha256"]
            or set(identity.get("diagnostic_arms", [])) != set(ARMS)
            or len(identity["diagnostic_arms"]) != len(ARMS)):
        raise ValueError("native full-memory/four-arm identity differs")
    expected = plan["events"]
    history = plan["identity"]["history"]
    source_events = [event for event in expected if event["kind"] == "source"]
    if (any(event["kind"] not in {"source", "probe"} for event in expected)
            or [e["turn"] for e in source_events] != list(range(1, len(source_events) + 1))
            or [e["id"] for e in source_events] != [f"source:{i}" for i in range(1, len(source_events) + 1)]
            or [e["source_sha256"] for e in source_events] != plan["identity"]["sources"]):
        raise ValueError("native source-plan roster differs")
    preflights, diagnostics, originals, completed = [], [], {}, 0
    seen_preflight = seen_write = seen_probe = False
    ended = False
    for row in rows[1:]:
        kind = row["event"]
        if ended or kind == "chronological_meta":
            raise ValueError("duplicate metadata or appended native trace")
        event = expected[completed] if completed < len(expected) else None
        if kind in {"source_preflight", "source_written", "chronological_probe", "chronological_event_complete"}:
            if event is None or row.get("event_id") != event["id"]:
                raise ValueError("native event ordering/identity differs")
        if kind == "source_preflight":
            if event["kind"] != "source" or seen_preflight or seen_write or row["turn"] != len(originals) + 1:
                raise ValueError("duplicate/misordered original preflight")
            preflights.append(dict(event="as_of_probe", probe_id=event["id"], conversation=history,
                split_turn=len(originals), state_turns=len(originals), preflight=row["preflight"], controls={}))
            seen_preflight = True
        elif kind == "source_written":
            sequence = row["source_sequence"]
            if (event["kind"] != "source" or not seen_preflight or seen_write
                    or row["turn"] != event["turn"] or row["clock"] != event["clock"]
                    or row["original_time"] != event["original_time"]
                    or row["ts_provenance"] != event["provenance"]
                    or sequence["position"] != row["turn"]
                    or sequence["sequence_sha256"] != plan["identity"]["source_sequence_sha256"]
                    or sequence["raw_sha256"] != row["source_raw_sha256"]):
                raise ValueError("native written source binding differs")
            adapted = dict(event="written", conversation=history, turn=row["turn"],
                           episodic_id=row["episodic_id"], batch_id=row["batch_id"])
            originals[str(row["turn"])] = [row["episodic_id"], row["batch_id"]]
            preflights.append(adapted)
            diagnostics.append(adapted)
            seen_write = True
        elif kind == "chronological_probe":
            if (event["kind"] != "probe" or seen_probe or row["probe_id"] != event["probe_id"]
                    or row["split_turn"] != len(originals) or row["split_turn"] != event["turn"]
                    or row["conversation"] != history or row["question_time"] != event["clock"]
                    or row["clock_basis"] != event["clock_basis"]
                    or row.get("probe_writes") is not False or row.get("answer_quality_scored") is not False
                    or digest(row["scorer"]) != event["probe_sha256"]
                    or row["question"] != row["scorer"]["question"]
                    or set(row["stages"]) != set(ARMS)):
                raise ValueError("native diagnostic cutoff/roster/scorer binding differs")
            gold = row["scorer"].get("gold_turns")
            if (not isinstance(gold, list) or not gold or len(gold) != len(set(gold))
                    or any(type(t) is not int or not 1 <= t <= event["turn"] for t in gold)):
                raise ValueError("native diagnostic sources are future/invalid")
            sources.reviewed_occurrence(row["scorer"], history, event["turn"])
            diagnostics.append(dict(event="as_of_probe", probe_id=row["probe_id"],
                conversation=history, split_turn=row["split_turn"], state_turns=len(originals),
                preflight=row["stages"]["full"],
                controls={a: row["stages"][a] for a in ARMS if a != "full"}))
            seen_probe = True
        elif kind == "chronological_event_complete":
            if (row["completed"] != completed + 1 or (event["kind"] == "source" and not seen_write)
                    or (event["kind"] == "probe" and not seen_probe)):
                raise ValueError("native incomplete/duplicate event receipt")
            completed += 1
            seen_preflight = seen_write = seen_probe = False
        elif kind == "chronological_complete":
            if (completed != len(expected) or row["events"] != completed or row["version"] != "v3"
                    or row["source_turns"] != len(originals)
                    or row["diagnostic_occurrences"] != len(plan["identity"]["occurrences"])
                    or row["clock_basis"] != plan["identity"]["clock_basis"]
                    or row["condition"] != "full" or row.get("answer_quality_scored") is not False):
                raise ValueError("native chronological completion differs")
            ended = True
    if not ended:
        raise ValueError("native chronological trace is incomplete")
    # Counting checks aliases and duplicate source/occurrence denominators too.
    catalog_store.verify_trace(root, rows)
    resolver = catalog_store.Reader(root)
    actual = count_usage(preflights, arms=("full",), native_evidence_only=True,
                         catalog_resolver=resolver)
    diagnostic = count_usage(diagnostics, native_evidence_only=True, catalog_resolver=resolver)
    captured = 0
    refs, logical, revisions = set(), set(), set()
    expected_catalogs = 0
    for row in rows:
        if row["event"] in {"source_written", "source_preflight"}:
            owners = [row if row["event"] == "source_written" else row["preflight"]]
        elif row["event"] == "chronological_probe":
            owners = list(row["stages"].values())
        else:
            continue
        expected_catalogs += len(owners)
        for owner in owners:
            value = resolver(owner)
            if value is None:
                continue
            catalog = validate_inventory(value)
            captured += 1
            refs.update(catalog)
            logical.update(tuple(item["logical_identity"]) for item in catalog.values())
            revisions.update((ref, item["stored_content_sha256"]) for ref, item in catalog.items())
    if captured and captured != expected_catalogs:
        raise ValueError("native observed catalogs have partial capture coverage")
    catalog_summary = (dict(captured_catalogs=captured, observed_physical_ids=len(refs),
        observed_logical_ids=len(logical), observed_payload_revisions=len(revisions),
        lifetime_write_events="unobserved", all_query_eligibility="unobserved",
        boundary="Observed before reads/after originals; between-capture writes/removals are unobserved.")
        if captured else "unobserved")
    return dict(version="v3", policy=POLICY, score_of_record=False,
        actual_preflight_exposure=actual, readonly_diagnostic_preparation=diagnostic,
        observed_catalog_lifetime=catalog_summary,
        source_ids=originals, event_prefix=sources.prefix_receipt(plan, completed),
        limits=["Captured exact artifacts/final blocks are exposure only; absent receipts, stored content versions and eligibility remain unobserved.",
                "These are separate source-exposure channels, not independent trials or answer support.",
                "Source inventory is lifetime observed writes, not a warm/live/eligible inventory."])


def file_receipt(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=prefix_digest(path, path.stat().st_size))


def report(root):
    root = Path(root).resolve()
    if not root.is_relative_to((ROOT / "logs").resolve()):
        raise ValueError("native reader reports require private logs")
    trace = root / "chronological.jsonl"
    body = trace.read_bytes()
    rows = [json.loads(line) for line in body.splitlines()]
    value = adapt(rows, root=root)
    pointer_path = root / "recovery" / "current.json"
    pointer = json.loads(pointer_path.read_text())
    generation = pointer["generation"]
    if not generation.startswith("generation-") or Path(generation).name != generation:
        raise ValueError("invalid native recovery generation")
    directory = root / "recovery" / generation
    paths = [pointer_path, directory / "recovery.json", directory / "store.manifest.json",
             directory / "store.sql", directory / "store.counts"]
    if any(not path.resolve().is_relative_to(root) for path in [trace, *paths]):
        raise ValueError("native reader input escaped producer")
    manifest = json.loads(paths[1].read_text())
    store = json.loads(paths[2].read_text())
    identity = rows[0]["identity"]
    references = catalog_store.verify_trace(root, rows)
    for key, reference in sorted(references.items()):
        path = root / catalog_store.DIRECTORY / (key + ".json.gz")
        receipt = file_receipt(path)
        if receipt["sha256"] != reference["stored_sha256"] or receipt["bytes"] != reference["stored_bytes"]:
            raise ValueError("counted catalog changed before report binding")
        paths.append(path)
    count_lines = paths[4].read_text().splitlines()[1:]
    saved_counts = {name: int(value) for name, value in (line.split("=", 1) for line in count_lines)}
    if (manifest["identity"] != identity or manifest["generation"] != generation
            or manifest["trace_path"] != str(trace.resolve())
            or manifest["trace_bytes"] != len(body)
            or manifest["trace_sha256"] != prefix_digest(trace, len(body))
            or manifest["state"].get("complete") is not True
            or manifest["state"]["event_prefix"] != value["event_prefix"]
            or manifest["state"]["source_ids"] != value["source_ids"]
            or store["database"] != identity["database"] or store["tables"] != identity["tables"]
            or set(store["counts"]) != set(identity["tables"])
            or set(store["sha256_by_table"]) != set(identity["tables"])
            or saved_counts != store["counts"] or len(count_lines) != len(saved_counts)
            or store["dump_sha256"] != prefix_digest(paths[3], paths[3].stat().st_size)):
        raise ValueError("native complete checkpoint/trace/store binding differs")
    value["identity"] = dict(producer=identity, inputs=[file_receipt(p) for p in [trace, *paths]],
        instruments=[file_receipt(Path(__file__)), file_receipt(Path(sources.__file__)),
                     file_receipt(Path(count_usage.__code__.co_filename)),
                     file_receipt(Path(count_usage.__globals__["observe_final_blocks"].__code__.co_filename)),
                     file_receipt(Path(catalog_store.__file__)),
                     file_receipt(Path(count_usage.__globals__["validate_inventory"].__code__.co_filename))])
    value["identity_sha256"] = digest(value["identity"])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chronological-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.chronological_run.resolve(), args.out.resolve()
    if (not out.is_relative_to((ROOT / "logs").resolve())
            or out.is_relative_to(root) or root.is_relative_to(out)):
        raise ValueError("native reader output must be a separate private path")
    value = report(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_or_verify(out, value)
    print(json.dumps(dict(version="v3", preflights=len(value["actual_preflight_exposure"]["occurrences"]),
        diagnostics=len(value["readonly_diagnostic_preparation"]["occurrences"]), score_of_record=False)))


if __name__ == "__main__":
    main()
