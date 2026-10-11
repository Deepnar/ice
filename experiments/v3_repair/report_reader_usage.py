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

from experiments.v3_repair.artifact_inventory import validate as validate_inventory
from experiments.v3_repair.compare_tuning_contexts import load_contexts
from experiments.v3_repair.final_prompt_artifacts import observe as observe_final_blocks
from experiments.v3_repair.prepare_inputs import digest, fingerprint, write_or_verify
from experiments.v3_repair.run_tuning_contexts import source_probes

POLICY = "ice-v3-source-exposure-frequency-1"
ARMS = ("full", "no_codex", "vector_only", "recent_only")


def source_ids(fragment):
    return {s for s in [fragment.get("source_row"), *(fragment.get("origin_batches") or [])] if s}


def count_usage(rows, probes=None, *, arms=ARMS, native_evidence_only=False):
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError("unknown/duplicate reader arm roster")
    aliases, inventory, arm_counts = {}, {}, {arm: Counter() for arm in arms}
    frequencies = {arm: defaultdict(Counter) for arm in arms}
    edge_counts = {arm: Counter() for arm in arms}
    versions = {arm: Counter() for arm in arms}
    artifact_counts = {arm: defaultdict(Counter) for arm in arms}
    artifact_versions = {arm: defaultdict(Counter) for arm in arms}
    block_versions = {arm: defaultdict(Counter) for arm in arms}
    stored_versions = {arm: defaultdict(Counter) for arm in arms}
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
        if set(stages) != set(arms): raise ValueError("reader arm roster incomplete")
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
            if native_evidence_only:
                # Native assembly puts retrieved evidence immediately before
                # its acknowledgement and the final question. Neither that
                # question nor an identical recent-history string is evidence.
                if (not isinstance(messages, list) or not messages or any(
                        not isinstance(m.get("content"), str) or m.get("role") not in {"system", "user", "assistant"}
                        for m in messages)):
                    raise ValueError("native final prompt messages missing/incomplete")
                evidence_cost = stage["context_ledger"]["blocks"]["evidence"]
                visible = (messages[-3]["content"] if evidence_cost > 0 and len(messages) >= 3
                    and messages[-3].get("role") == "user"
                    and messages[-3].get("content", "").startswith("=== RETRIEVED CONTEXT ===")
                    and messages[-2].get("role") == "assistant"
                    and messages[-2].get("content") == "Understood — I have the background context. What would you like to know?"
                    and messages[-1].get("role") == "user" else "")
            else:
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
                if native_evidence_only:
                    refs, unmapped = set(), 0
                    for f in fragments:
                        raw_refs = f.get("stored_artifact_ids") or []
                        if not raw_refs:
                            unmapped += 1
                        for ref in raw_refs:
                            if (not isinstance(ref, (list, tuple)) or len(ref) != 2
                                    or any(not isinstance(v, str) or not v for v in ref)):
                                raise ValueError("native artifact reference malformed")
                            refs.add(tuple(ref))
                    for ref in refs:
                        artifact_counts[arm][ref][label] += 1
                    per_stage[label].update(unique_observed_stored_artifacts=len(refs),
                                            unmapped_artifact_fragments=unmapped)
            rendered_versions = set()
            for f in rendered:
                identity = dict(type=f["type"], text_sha256=hashlib.sha256(f["text"].encode()).hexdigest(),
                                source_ids=sorted(source_ids(f)), edges=sorted(set(f.get("origin_edges") or [])))
                rendered_versions.add(digest(identity))
            for version in rendered_versions: versions[arm][version] += 1
            if native_evidence_only:
                rendered_artifacts = set()
                for f in rendered:
                    if not f.get("stored_artifact_ids"):
                        continue
                    version = hashlib.sha256(f["text"].encode()).hexdigest()
                    if version != f.get("rendered_text_sha256"):
                        raise ValueError("native rendered artifact version differs from text")
                    rendered_artifacts.update((tuple(ref), version) for ref in f["stored_artifact_ids"])
                for ref, version in rendered_artifacts:
                    artifact_versions[arm][ref][version] += 1
            catalog = (validate_inventory(stage["artifact_inventory"])
                       if native_evidence_only and "artifact_inventory" in stage else None)
            if catalog is not None:
                for ref, item in catalog.items():
                    artifact_counts[arm][ref]["live_catalog_opportunities"] += 1
                    stored_versions[arm][ref][item["stored_content_sha256"]] += 1
                observed_refs = {tuple(ref) for values in groups.values() for f in values
                                 for ref in f.get("stored_artifact_ids", [])}
                if observed_refs - catalog.keys():
                    raise ValueError("native retrieved artifact absent from pre-read live catalog")
            block_observation = "unobserved"
            if native_evidence_only:
                blocks = observe_final_blocks(stage)
                if blocks is not None:
                    final_refs = {ref for ref, _ in rendered_artifacts}
                    block_observation = {}
                    for name, details in blocks.items():
                        for ref in details["refs"]:
                            artifact_counts[arm][ref]["final_block:" + name] += 1
                        for ref, version in details["versions"]:
                            block_versions[arm][ref][name + ":" + version] += 1
                        final_refs.update(details["refs"])
                        block_observation[name] = {k: v for k, v in details.items() if k not in {"refs", "versions"}}
                        block_observation[name]["unique_observed_stored_artifacts"] = len(details["refs"])
                    if catalog is not None and final_refs - catalog.keys():
                        raise ValueError("native final context artifact absent from pre-read live catalog")
                    for ref in final_refs:
                        artifact_counts[arm][ref]["any_final_context"] += 1
                    arm_counts[arm]["final_block_observed_occurrences"] += 1
                else:
                    if catalog is not None:
                        raise ValueError("live inventory lacks native final context observation")
                    arm_counts[arm]["final_block_unobserved_occurrences"] += 1
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
            if native_evidence_only:
                observation["arms"][arm]["final_prompt_blocks"] = block_observation
                observation["arms"][arm]["live_artifact_inventory"] = len(catalog) if catalog is not None else "unobserved"
        occurrences.append(observation)
    if probes is not None and seen_probes != set(probes): raise ValueError("reader occurrence roster differs")
    final = {}
    for arm in arms:
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
        if native_evidence_only:
            final[arm]["observed_stored_artifact_frequency"] = [dict(table=ref[0], key=ref[1],
                counts=dict(counts), rendered_version_frequency=dict(sorted(artifact_versions[arm][ref].items())),
                final_block_version_frequency=dict(sorted(block_versions[arm][ref].items())),
                stored_payload_revision_opportunities=dict(sorted(stored_versions[arm][ref].items())),
                zero_context_reads_at_observed_live_opportunities=(counts.get("any_final_context", 0) == 0
                    if counts.get("live_catalog_opportunities", 0) else "unobserved"),
                eligible_opportunities="unobserved", live_inventory="captured_at_sampled_read_start"
                    if counts.get("live_catalog_opportunities", 0) else "unobserved")
                for ref, counts in sorted(artifact_counts[arm].items())]
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
