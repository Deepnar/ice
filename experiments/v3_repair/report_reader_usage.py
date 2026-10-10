"""Offline v3 source exposure/frequency accounting; missing artifact IDs stay unknown."""

import argparse
import fcntl
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.compare_tuning_contexts import load_contexts
from experiments.v3_repair.prepare_inputs import digest, fingerprint, write_or_verify
from experiments.v3_repair.run_tuning_contexts import source_probes

POLICY = "ice-v3-source-exposure-frequency-1"
ARMS = ("full", "no_codex", "vector_only", "recent_only")


def source_ids(fragment):
    return {s for s in [fragment.get("source_row"), *(fragment.get("origin_batches") or [])] if s}


def count_usage(rows, probes=None):
    aliases, inventory, arm_counts = {}, {}, {arm: Counter() for arm in ARMS}
    frequencies = {arm: defaultdict(Counter) for arm in ARMS}
    edge_counts = {arm: Counter() for arm in ARMS}
    versions = {arm: Counter() for arm in ARMS}
    occurrences, seen_probes = [], set()
    for row in rows:
        if row["event"] == "written":
            key = f"{row['conversation']}:{row['turn']}"
            if key in inventory: raise ValueError("duplicate stored original in reader trace")
            inventory[key] = dict(conversation=row["conversation"], turn=row["turn"],
                                  aliases=sorted({row["episodic_id"], row["batch_id"]}))
            for alias in inventory[key]["aliases"]:
                if alias in aliases: raise ValueError("ambiguous source alias")
                aliases[alias] = key
        if row["event"] != "as_of_probe": continue
        key = row["probe_id"]
        if key in seen_probes: raise ValueError("duplicate reader occurrence")
        seen_probes.add(key)
        current = probes[key] if probes is not None else row
        if (current["conversation"], current["split_turn"]) != (row["conversation"], row["split_turn"]):
            raise ValueError("reader cutoff differs from original inventory")
        stages = {"full": current["preflight"], **current["controls"]}
        if set(stages) != set(ARMS): raise ValueError("reader arm roster incomplete")
        own_inventory = {name for name, item in inventory.items() if item["conversation"] == row["conversation"]}
        if row["state_turns"] != len(own_inventory): raise ValueError("stored original denominator differs")
        observation = dict(probe_id=key, conversation=row["conversation"], cutoff=row["split_turn"],
            known_stored_originals=len(inventory), own_stored_originals=len(own_inventory),
            eligible_inventory="unobserved", arms={})
        for arm, stage in stages.items():
            arm_counts[arm]["question_occurrences"] += 1
            for original in inventory: frequencies[arm][original]["observed_opportunities"] += 1
            groups = dict(generated=[f for fs in stage["generated_by_leg"].values() for f in fs],
                          ranked=stage["ranked_candidates"], budgeted=stage["budgeted"], selected=stage["selected"])
            messages = stage.get("prompt_messages")
            visible = "\n".join(str(m.get("content", "")) for m in messages) if messages is not None else None
            rendered = [f for f in groups["selected"] if visible is not None and f.get("text") and f["text"] in visible]
            groups["rendered_selected"] = rendered
            missing = len(groups["selected"]) - len(rendered)
            arm_counts[arm]["selected_text_unconfirmed"] += missing
            per_stage, unknown = {}, set()
            for label, fragments in groups.items():
                ids = {s for f in fragments for s in source_ids(f)}
                known = {aliases[s] for s in ids if s in aliases}
                unknown.update(ids - aliases.keys())
                for original in known: frequencies[arm][original][label] += 1
                arm_counts[arm][f"{label}_fragments"] += len(fragments)
                for f in fragments: arm_counts[arm][f"{label}_leg:{f['leg']}"] += 1
                per_stage[label] = dict(fragments=len(fragments), unique_originals=len(known))
            rendered_versions = set()
            for f in rendered:
                identity = dict(type=f["type"], text_sha256=hashlib.sha256(f["text"].encode()).hexdigest(),
                                source_ids=sorted(source_ids(f)), edges=sorted(set(f.get("origin_edges") or [])))
                rendered_versions.add(digest(identity))
            for version in rendered_versions: versions[arm][version] += 1
            # Aggregate fragments credit edge lineage, not exact per-edge wording.
            for edge in {e for f in rendered for e in f.get("origin_edges") or []}: edge_counts[arm][edge] += 1
            notes = set(stage.get("source_note_ids") or [])
            for original in {aliases[s] for s in notes if s in aliases}:
                frequencies[arm][original]["rendered_note_lineage"] += 1
            unknown.update(notes - aliases.keys())
            rendered_originals = {aliases[s] for f in rendered for s in source_ids(f) if s in aliases}
            rendered_originals.update(aliases[s] for s in notes if s in aliases)
            for original in rendered_originals: frequencies[arm][original]["any_final_evidence_lineage"] += 1
            arm_counts[arm]["unknown_source_alias_occurrences"] += len(unknown)
            observation["arms"][arm] = dict(stages=per_stage, unknown_source_aliases=sorted(unknown),
                any_final_evidence_lineage=len(rendered_originals), selected_text_unconfirmed=missing,
                exact_derived_artifact_inventory="unobserved", answer_use="unmeasured")
        occurrences.append(observation)
    if probes is not None and seen_probes != set(probes): raise ValueError("reader occurrence roster differs")
    final = {}
    for arm in ARMS:
        originals = []
        for key, item in sorted(inventory.items()):
            counters = frequencies[arm][key]
            originals.append(dict(original=key, **item, counts=dict(counters),
                final_evidence_occurrences=counters["any_final_evidence_lineage"],
                eligible_opportunities="unobserved"))
        observed = [x for x in originals if x["counts"].get("observed_opportunities", 0)]
        final[arm] = dict(totals=dict(arm_counts[arm]), stored_originals=len(originals),
            observed_originals=len(observed), zero_evidence_reads_observed=sum(x["final_evidence_occurrences"] == 0 for x in observed),
            originals=originals, credited_graph_edge_frequency=dict(sorted(edge_counts[arm].items())),
            rendered_fragment_version_frequency=dict(sorted(versions[arm].items())))
    return dict(version="v3", policy=POLICY, score_of_record=False, arms=final, occurrences=occurrences,
        limits=["Final source lineage is exposure, not required-content support or answer use.",
                "Observed opportunities are sampled probes after storage, not eligibility or all live requests.",
                "Edge IDs are credited aggregate lineage, not exact rendered facts; signatures are versions, not stored row IDs.",
                "Derived artifact/slot/bookmark/recent inventory and eligible-source denominators remain unobserved.",
                "Do not sum arms or dependent repeats into independent memories or trials."])


def report(root, collection=None):
    producer, original = source_probes(root)
    header, probes = load_contexts(collection, original) if collection else (None, original)
    if header and (header["identity"]["producer"] != producer["producer_identity"]
                   or header["identity"]["producer_trace_sha256"] != producer["trace_sha256"]):
        raise ValueError("reader collection differs from producer")
    rows = [json.loads(line) for line in (root / "seed.jsonl").read_text().splitlines()]
    value = count_usage(rows, probes)
    value["identity"] = dict(producer=producer["producer_identity"], trace_sha256=producer["trace_sha256"],
        collection=header["identity_sha256"] if header else None, instrument=fingerprint(Path(__file__)))
    value["identity_sha256"] = digest(value["identity"])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", required=True, type=Path)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, collection, out = args.development_run.resolve(), args.collection.resolve() if args.collection else None, args.out.resolve()
    if (any(not p.is_relative_to((ROOT / "logs").resolve()) for p in [root, out, *([collection] if collection else [])])
            or any(out.is_relative_to(p) or p.is_relative_to(out) for p in [root, *([collection] if collection else [])])):
        raise ValueError("reader source/output must be separate private paths")
    value = report(root, collection)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_or_verify(out, value)
    print(json.dumps(dict(version="v3", occurrences=len(value["occurrences"]), score_of_record=False)))


if __name__ == "__main__": main()
