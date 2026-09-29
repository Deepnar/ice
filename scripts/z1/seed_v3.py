#!/usr/bin/env python3
"""ICE v3 sequential recorded-response replay with private failure traces.

Use a fresh, dedicated database (DATABASE_URL and ICE_TEST_DATABASE must name
the same non-ice_db database). Historical replies are stored *after* each real
preflight; they are never regenerated. This reconstructs memory state and
captures as-of prompts. It is not an answer-quality score.

  uv run python scripts/z1/seed_v3.py --check
  uv run python scripts/z1/seed_v3.py --out logs/z1-v3-seed.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import func, text
from sqlalchemy.engine import make_url

from scripts.z1 import production_parity as pp
from scripts.z1.run_meta import file_digest, run_meta
from src.api.config import settings
from src.api.db import SessionLocal
from src.ingestion.importer import _store_turn
from src.memory.models import Base, Conversation, EpisodicMemory, MemorySlot
from src.memory.tokens import count_messages

CORPUS = Path("data/simulation/simulation_full.jsonl")
UNIFIED = Path("experiments/curation_files/unified_probes.json")
TYPED = Path("experiments/curation_files/typed_probes.json")
DERIVED = Path("experiments/curation_files/derived_gt.json")
GENERATED = Path("experiments/curation_files/generated_probes.json")
EXPECTED = {"bb558b5f": 1119, "ecc64aab": 251, "355a5709": 101}
MARK = "z1seed"


def canonical_probe_id(probe: dict) -> str:
    """Catalog IDs repeat across section files; bind identity to the full row."""
    identity = {key: probe.get(key) for key in
                ("source", "probe_id", "conversation", "source_file",
                 "split_turn", "question")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return f"catalog:{digest}"


def load_native_checkpoint_probes(unified, conversations, recent_gap):
    """Use existing checkpoint questions with independently derived source turns.

    Most curated/mature questions have no source-turn label. Never infer it
    from their target leg or expected answer. The old derived file covers a
    subset; only nonfuture, in-corpus, already-long-term pairs enter here, and
    their machine-derived gold still requires source review before scoring.
    """
    derived = defaultdict(list)
    for row in json.loads(DERIVED.read_text())["probes"]:
        derived[row["question"]].append(row)
    selected = []
    for row in unified:
        if row.get("source") == "typed":
            continue
        matches = derived[row["question"]]
        if (len(matches) != 1 or matches[0].get("source") != row["source"]
                or not matches[0].get("gold_turns")):
            continue
        slug = row["conversation"]
        split = row.get("split_turn")
        remap = row.get("remapped_from")
        prior_slug = remap["conversation"] if remap else slug
        offset = int(remap["offset"]) if remap else 0
        if matches[0]["conversation"] != prior_slug:
            continue
        gold = [int(turn) + offset for turn in matches[0]["gold_turns"]]
        if (not isinstance(split, int) or split > len(conversations[slug])
                or min(gold) < 1 or max(gold) > split
                or split - max(gold) < recent_gap):
            continue
        probe = dict(row)
        probe["catalog_probe_id"] = row["probe_id"]
        probe["probe_id"] = canonical_probe_id(row)
        probe["gold_turns"] = gold
        probe["probe_type"] = f"native_{row['source']}_{row.get('target_leg') or 'general'}"
        probe["source_split_turn"] = max(gold)
        probe["cutoff_kind"] = "native_checkpoint"
        probe["label_status"] = "needs_gold_source_confirmation"
        selected.append(probe)
    return selected


def load_unlabeled_native_checkpoints(unified, conversations, mapped_ids):
    """Capture original section-time prompts without assigning false gold."""
    selected = []
    for row in unified:
        if row.get("source") == "typed":
            continue
        slug, split = row["conversation"], row.get("split_turn")
        if not isinstance(split, int) or not 1 <= split <= len(conversations[slug]):
            continue
        identity = canonical_probe_id(row)
        if identity in mapped_ids:
            continue
        probe = dict(row)
        probe.update(probe_id=identity, catalog_probe_id=row["probe_id"],
                     probe_type=f"unlabeled_{row['source']}_{row.get('target_leg') or 'general'}",
                     gold_turns=[], source_split_turn=None,
                     cutoff_kind="native_unlabeled_checkpoint",
                     label_status="needs_source_turn_and_answer_review")
        selected.append(probe)
    return selected


def load_generated_checkpoint_probes(unified, conversations, recent_gap):
    """Reuse old source-first questions only after the corrected ambiguity guard.

    A candidate needs exact quoted source evidence, a real later section
    checkpoint outside recent history, and no equal-or-better lexical rival
    before that checkpoint. These are conservative *screening* conditions;
    source support and intervening changes still need review.
    """
    from scripts.z1.derive_retrieval_gt import build_idf
    from scripts.z1.generate_probes import better_elsewhere

    checkpoints = defaultdict(set)
    for probe in unified:
        slug, split = probe["conversation"], probe.get("split_turn")
        if (probe.get("source") != "typed" and slug in conversations
                and isinstance(split, int) and split <= len(conversations[slug])):
            checkpoints[slug].add(split)
    old_rows = {slug: [{"turn_number": row["turn_number"],
                        "user_input": row["prompt"],
                        "ai_response": row["response"]} for row in turns]
                for slug, turns in conversations.items()}
    idf_at_checkpoint = {}
    selected = []
    seen = set()
    for row in json.loads(GENERATED.read_text())["probes"]:
        original_slug = row["conversation"]
        offset = 1038 if original_slug == "cca73c87" else 0
        slug = "bb558b5f" if offset else original_slug
        gold = int(row["gold_turn"]) + offset
        if slug not in conversations or not 1 <= gold <= len(conversations[slug]):
            continue
        source = conversations[slug][gold - 1]
        excerpt = (row.get("evidence") or "").strip()
        if not excerpt or excerpt.casefold() not in (
                source["prompt"] + "\n" + source["response"]).casefold():
            continue
        later = sorted(split for split in checkpoints[slug]
                       if split - gold >= recent_gap)
        if not later:
            continue
        cutoff = later[0]
        idf_key = (slug, cutoff)
        if idf_key not in idf_at_checkpoint:
            idf_at_checkpoint[idf_key] = build_idf(old_rows[slug][:cutoff])
        if better_elsewhere(row["question"], row["answer"], gold,
                            old_rows[slug][:cutoff],
                            idf_at_checkpoint[idf_key]) is not None:
            continue
        in_user = excerpt.casefold() in source["prompt"].casefold()
        in_assistant = excerpt.casefold() in source["response"].casefold()
        if not in_user and not in_assistant:
            continue
        role = "both" if in_user and in_assistant else (
            "user" if in_user else "assistant")
        key = f"{original_slug}:{row['gold_turn']}:{row['question']}"
        probe_id = "generated:" + hashlib.sha256(key.encode()).hexdigest()[:16]
        if probe_id in seen:
            raise ValueError("duplicate generated long-term probe identity")
        seen.add(probe_id)
        selected.append({"probe_id": probe_id, "source": "generated",
                         "probe_type": f"generated_{role}",
                         "conversation": slug, "split_turn": cutoff,
                         "source_split_turn": gold, "gold_turns": [gold],
                         "question": row["question"],
                         "expected_answer": row["answer"],
                         "source_excerpt": excerpt, "source_role": role,
                         "cutoff_kind": "generated_at_checkpoint",
                         "label_status": "needs_intervening_turn_review"})
    return selected


def load_plan(*, probe_timing="delayed", include_native=True,
              include_typed=False, include_generated=True,
              capture_unlabeled_native=False):
    """Reject bad sources and schedule questions outside direct recent history.

    The default panel uses existing questions at later section checkpoints.
    Unified split_turn is the latest gold turn for all typed probes, so typed
    questions are optional diagnostics. Their delayed cutoff is the first turn
    where gold leaves recent history. Every answer needs source and intervening
    turn review before scoring.
    """
    if probe_timing not in {"delayed", "immediate"}:
        raise ValueError("unknown probe timing")
    conversations = {slug: [] for slug in EXPECTED}
    for line in CORPUS.open():
        row = json.loads(line)
        slug = row["conversation_id"][:8]
        if slug not in conversations:
            continue
        stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError(f"{slug}: source timestamp lacks timezone")
        if not row.get("prompt") or not row.get("response"):
            raise ValueError(f"{slug}: blank historical prompt or reply")
        conversations[slug].append({"turn_number": len(conversations[slug]) + 1,
                                    "timestamp": stamp,
                                    "prompt": row["prompt"],
                                    "response": row["response"]})
    for slug, rows in conversations.items():
        if len(rows) != EXPECTED[slug]:
            raise ValueError(f"{slug}: expected {EXPECTED[slug]} turns, got {len(rows)}")
        if any(a["timestamp"] >= b["timestamp"] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"{slug}: source timestamps not strictly ordered")

    unified = json.loads(UNIFIED.read_text())["probes"]
    by_question = defaultdict(list)
    for probe in unified:
        by_question[probe["question"]].append(probe)
    typed = json.loads(TYPED.read_text())["probes"]
    at_turn = defaultdict(list)
    ineligible_typed = []
    typed_scheduled = 0
    recent_gap = int(settings.recent_window_max_turns)
    for old in typed:
        matches = by_question[old["question"]]
        if len(matches) != 1 or matches[0].get("source") != "typed":
            raise ValueError("typed probe has no unique unified mapping")
        probe = dict(matches[0])
        probe["catalog_probe_id"] = probe["probe_id"]
        probe["probe_id"] = canonical_probe_id(matches[0])
        if probe.get("probe_type") != old.get("probe_type"):
            raise ValueError("typed probe type changed during unification")
        remap = probe.get("remapped_from")
        offset = int(remap["offset"]) if remap else 0
        prior_slug = remap["conversation"] if remap else probe["conversation"]
        if prior_slug != old["conversation"] or [int(t) + offset for t in old["gold_turns"]] != probe["gold_turns"]:
            raise ValueError("typed probe gold mapping changed during unification")
        slug = probe["conversation"]
        split = probe.get("split_turn")
        if not isinstance(split, int) or not 1 <= split <= len(conversations[slug]):
            raise ValueError("typed probe has no valid historical cutoff")
        if any(t > split for t in probe["gold_turns"]):
            raise ValueError("typed gold lies after question cutoff")
        if split != max(probe["gold_turns"]):
            raise ValueError("typed cutoff assumption changed; re-audit label timing")
        probe["anchor_entity"] = old.get("anchor_entity")
        probe["superseded_by"] = old.get("superseded_by")
        probe["source_split_turn"] = split
        probe["cutoff_kind"] = probe_timing
        probe["label_status"] = ("needs_intervening_turn_review"
                                 if probe_timing == "delayed" else "source_turn_only")
        cutoff = split + recent_gap if probe_timing == "delayed" else split
        if cutoff > len(conversations[slug]):
            ineligible_typed.append(probe["probe_id"])
            continue
        probe["split_turn"] = cutoff
        typed_scheduled += 1
        if include_typed:
            at_turn[(slug, cutoff)].append(probe)
    if typed_scheduled + len(ineligible_typed) != 444:
        raise ValueError("typed catalog is incomplete")
    native = (load_native_checkpoint_probes(unified, conversations, recent_gap)
              if include_native or capture_unlabeled_native else [])
    if include_native:
        for probe in native:
            at_turn[(probe["conversation"], probe["split_turn"])].append(probe)
    if capture_unlabeled_native:
        mapped_ids = {probe["probe_id"] for probe in native}
        for probe in load_unlabeled_native_checkpoints(unified, conversations,
                                                        mapped_ids):
            at_turn[(probe["conversation"], probe["split_turn"])].append(probe)
    if include_generated:
        for probe in load_generated_checkpoint_probes(unified, conversations,
                                                      recent_gap):
            at_turn[(probe["conversation"], probe["split_turn"])].append(probe)
    ids = [p["probe_id"] for group in at_turn.values() for p in group]
    if len(ids) != len(set(ids)):
        raise ValueError("scheduled probes do not have unique identities")
    quarantined = [p.get("probe_id") for p in unified if p.get("source") != "typed"
                   and (not isinstance(p.get("split_turn"), int)
                        or p["split_turn"] > len(conversations[p["conversation"]]))]
    return conversations, at_turn, quarantined, ineligible_typed


def private_output(path: str) -> Path:
    out = Path(path).resolve()
    root = (Path(__file__).resolve().parents[2] / "logs").resolve()
    if not out.is_relative_to(root):
        raise ValueError("replay traces contain private conversation text; output must be under logs/")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def isolated_database() -> None:
    database = make_url(settings.database_url).database
    if (not os.environ.get("ICE_TEST_DATABASE")
            or os.environ["ICE_TEST_DATABASE"] != database or database == "ice_db"):
        raise RuntimeError("Set DATABASE_URL and ICE_TEST_DATABASE to one dedicated non-ice_db database")


def fragment(f, *, include_text=False):
    row = {"leg": f.leg or f.source_type, "type": f.source_type,
           "tokens": f.token_count, "score": f.score,
           "source_row": str(f.source_batch_id) if f.source_batch_id else None,
           "origin_batches": [str(v) for v in (f.origin_batch_ids or ())],
           "origin_edges": [str(v) for v in (f.origin_edge_ids or ())]}
    if include_text:
        row["text"] = f.text
    return row


def prepare_with_trace(db, pre, classifier, *, arm="full"):
    """Shared final preparation plus generated-leg and budget-stage evidence."""
    from src.api import memory_preparation as mp
    from src.api.prompt_assembler import bookmarked_turn_texts
    from src.model_registry.registry import get_model_context_window
    from src.model_registry.runtime_probe import serving_window
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator

    if arm not in {"full", "vector_only", "recent_only"}:
        raise ValueError(f"unknown prompt arm: {arm}")

    produced, budgeted = {}, []
    methods = {"bm25": "_bm25_episodic", "vector": "_vector_episodic",
               "codex_graph": "_codex_graph", "codex_claims": "_codex_claims",
               "procedural": "_procedural_lookup", "batch_summary": "_batch_summary_lookup",
               "cold": "_cold_lookup"}

    def factory(session, embedder):
        orchestrator = HybridRetrievalOrchestrator(session, embedder)
        if arm != "full":
            disabled = (set(methods) - {"vector"} if arm == "vector_only"
                        else set(methods))
            for label in disabled:
                setattr(orchestrator, methods[label], lambda *_a, **_k: [])
        for label, name in methods.items():
            original = getattr(orchestrator, name)

            def counted(*args, _original=original, _label=label, **kwargs):
                rows = _original(*args, **kwargs)
                produced[_label] = [fragment(f) for f in rows]
                return rows

            setattr(orchestrator, name, counted)
        original_retrieve = orchestrator.retrieve

        def captured(*args, **kwargs):
            rows = original_retrieve(*args, **kwargs)
            budgeted.extend(fragment(f) for f in rows)
            return rows

        orchestrator.retrieve = captured
        return orchestrator

    raw_window = get_model_context_window(pre.model_name)
    window = (serving_window(pre.model_name, raw_window)
              if settings.context_use_serving_window else raw_window)
    summary_patch = (patch.object(mp, "conversation_summary_block", return_value=None)
                     if arm != "full" else nullcontext())
    source_gate_patch = (patch.object(settings, "memory_source_gate_enabled", False)
                         if arm == "recent_only" else nullcontext())
    with (patch.object(mp, "HybridRetrievalOrchestrator", factory),
          summary_patch, source_gate_patch):
        memory = pp.prepare(
            db, pre, classifier,
            memory_slots=(db.query(MemorySlot).filter_by(is_active=True).all()
                          if arm != "recent_only" else []),
            bookmarked_texts=(bookmarked_turn_texts(db, pre.conversation_id)
                              if arm != "recent_only" else []),
            serving_window=window)
    if not memory.prepared.ledger.fits():
        raise RuntimeError("final prompt exceeds serving window")
    if arm == "recent_only" and memory.fragments:
        raise RuntimeError("recent-only control still contains retrieved fragments")
    if arm == "vector_only" and any(f.leg not in ("vector", "bm25+vector")
                                    for f in memory.fragments):
        raise RuntimeError("vector-only control contains another retrieval leg")
    return memory, produced, budgeted


def stage_record(pre, memory, produced, budgeted, *, include_prompt):
    selected = [fragment(f, include_text=True) for f in memory.fragments]
    messages = memory.prepared.messages
    result = {"gate": pp.provenance_fields(pre, memory),
              "topic_tags": pre.classification.topic_tags,
              "intent_tags": pre.classification.intent_tags,
              "generated_by_leg": produced, "budgeted": budgeted,
              "selected": selected, "evicted": memory.prepared.removed,
              "prompt_tokens": count_messages(messages),
              "selected_tokens": sum(f["tokens"] for f in selected),
              "answer_use": "unknown_until_new_answer_is_generated"}
    if include_prompt:
        result["prompt_messages"] = messages
    return result


def lineage_at_cutoff(stage, seen_source_times, cutoff):
    """Audit source identities at the actual question time, not final state."""
    unresolved, future = set(), set()
    with_source = 0
    for fragment_row in stage["selected"]:
        ids = ([fragment_row["source_row"]] if fragment_row["source_row"] else [])
        ids += fragment_row["origin_batches"]
        if ids:
            with_source += 1
        for source_id in ids:
            when = seen_source_times.get(source_id)
            if when is None:
                unresolved.add(source_id)
            elif when > cutoff:
                future.add(source_id)
    result = {"selected_with_source_identity": with_source,
              "selected_total": len(stage["selected"]),
              "unresolved_source_ids": sorted(unresolved),
              "future_source_ids": sorted(future)}
    if future:
        raise RuntimeError("future source reached an as-of prompt")
    return result


def gold_fragment_coverage(stage, gold_ids):
    """Count gold turns visible at each fragment stage, never infer answer use."""
    def ids(row):
        return ({row["source_row"]} if row["source_row"] else set()) | set(
            row["origin_batches"])

    def hits(rows):
        returned = set().union(*(ids(row) for row in rows)) if rows else set()
        return sum(bool(source_ids & returned) for source_ids in gold_ids.values())

    return {"gold_turns": len(gold_ids),
            "generated": hits([row for rows in stage["generated_by_leg"].values()
                               for row in rows]),
            "generated_by_leg": {leg: hits(rows) for leg, rows in
                                 stage["generated_by_leg"].items()},
            "budgeted": hits(stage["budgeted"]),
            "selected": hits(stage["selected"]),
            "interpretation": "fragment provenance only; recent context and answer use are separate"}


def due_maintenance(db, sink, source_time, conversation_id, last_run):
    """Exercise real periodic writers against as-of state, with stated cadence.

    Source-time gaps select when to run; job internals still use the actual
    machine clock. The trace declares that limitation. Decay/archive jobs are
    excluded here because they would age 2025 turns using the 2026 wall clock
    while we are still replaying 2025, silently erasing future as-of evidence.
    Their separate post-seed test must be reported before a whole-stack claim.
    """
    from src.workers.batch_summarizer import batch_summarize
    from src.workers.clustering import run_cluster_assignment, run_cluster_merge
    from src.workers.conversation_summary import run_conversation_summaries
    from src.workers.maintenance_agent import run_maintenance_agent
    from src.workers.reflection import run_reflection

    jobs = {
        "cluster_assignment": lambda: run_cluster_assignment(
            db, conversation_ids=[str(conversation_id)]),
        "cluster_merge": lambda: run_cluster_merge(
            db, conversation_ids=[str(conversation_id)]),
        "conversation_summary": lambda: run_conversation_summaries(
            db, conversation_ids=[conversation_id]),
        "batch_summarize": batch_summarize,
        "reflection": run_reflection,
        "maintenance_agent": lambda: run_maintenance_agent(db),
    }
    for name, fn in jobs.items():
        interval = settings.maintenance_intervals.get(name)
        previous = last_run.get(name)
        if not interval or (previous is not None
                            and (source_time - previous).total_seconds() < interval):
            continue
        try:
            result = fn()
            db.expire_all()
            sink.write(json.dumps({"event": "maintenance", "job": name,
                                   "source_time_trigger": source_time.isoformat(),
                                   "result": result}, default=str) + "\n")
            sink.flush()
            last_run[name] = source_time
        except Exception as exc:
            db.rollback()
            sink.write(json.dumps({"event": "maintenance_failed", "job": name,
                                   "source_time_trigger": source_time.isoformat(),
                                   "error_type": type(exc).__name__,
                                   "error": str(exc)[:500]}) + "\n")
            sink.flush()
            raise


class _FrozenClassifier:
    def __init__(self, result, expected_prompt, expected_conv):
        self.result, self.prompt, self.conv = result, expected_prompt, str(expected_conv)

    def classify(self, prompt, conversation_id=None):
        if prompt != self.prompt or str(conversation_id) != self.conv:
            raise ValueError("writer classification differs from preflight")
        return self.result


def run(args, conversations, probes) -> int:
    isolated_database()
    output = private_output(args.out)
    if output.exists():
        raise RuntimeError("trace output already exists; use a new run name")
    db = SessionLocal()
    try:
        nonempty = {name: db.execute(text(f'SELECT count(*) FROM "{name}"')).scalar()
                    for name in Base.metadata.tables}
        if any(nonempty.values()):
            raise RuntimeError("replay requires an empty dedicated database; found populated ORM tables")
        from src.classifier.classifier import PyTorchClassifier
        from src.memory.embedder import get_embedder
        from src.workers.bg_client_factory import get_bg_model_name
        from src.workers.post_flight import evaluate_turn

        embedder = get_embedder()
        classifier = PyTorchClassifier(model_path=settings.classifier_model_path,
                                       schema_path=settings.label_schema_path)
        background_model = get_bg_model_name()
        meta = run_meta(script=__file__, args=vars(args),
                        settings_keys=["recent_window_max_turns"],
                        inputs=[file_digest(CORPUS), file_digest(UNIFIED),
                                file_digest(TYPED), file_digest(DERIVED),
                                file_digest(GENERATED)],
                        extra={"version": "v3", "expected_turns": EXPECTED,
                               "recorded_answers_not_generated": True,
                               "clock_policy": "source timestamps stored; runtime wall clock not frozen",
                               "periodic_jobs": "source-time due checks; real wall-clock internals",
                               "unexercised_during_replay": [
                                   "decay_episodic", "decay_codex", "decay_procedural",
                                   "cold_archive", "project_document_paths"]})
        with output.open("x") as sink:
            sink.write(json.dumps({"event": "run", "meta": meta}) + "\n")
            last_maintenance = {}
            completed = Counter()
            probe_count = 0
            seen_source_times = {}
            turn_source_ids = {}
            selected_conversations = ({args.conversation: conversations[args.conversation]}
                                      if args.conversation else conversations)
            for slug, turns in selected_conversations.items():
                conv = Conversation(id=uuid.uuid4(), memory_scope_type="auto",
                                    created_at=turns[0]["timestamp"])
                db.add(conv)
                db.commit()
                bound = turns[:args.limit] if args.limit else turns
                for turn in bound:
                    number = turn["turn_number"]
                    key = f"{MARK}-{slug}-{number}"
                    stage = "preflight"
                    try:
                        pre = pp.build(db, turn["prompt"], conv.id, classifier, embedder)
                        memory, generated, budgeted = prepare_with_trace(db, pre, classifier)
                        preflight = stage_record(pre, memory, generated, budgeted,
                                                 include_prompt=True)
                        preflight["lineage"] = lineage_at_cutoff(
                            preflight, seen_source_times, turn["timestamp"])
                        before = db.query(EpisodicMemory).filter_by(conversation_id=conv.id).count()
                        record = {"event": "turn", "conversation": slug, "turn": number,
                                  "recorded_at": turn["timestamp"].isoformat(),
                                  "source_question_sha256": hashlib.sha256(turn["prompt"].encode()).hexdigest(),
                                  "before_turns": before,
                                  "preflight": preflight}
                        sink.write(json.dumps(record, default=str) + "\n")
                        sink.flush()
                        stage = "write"
                        frozen = _FrozenClassifier(pre.classification, turn["prompt"], conv.id)
                        stored = _store_turn(db, conv.id, turn["prompt"], turn["response"],
                                             turn["timestamp"], key, "fresh",
                                             datetime.now(timezone.utc), "original",
                                             frozen, embedder)
                        if stored is None:
                            raise RuntimeError("duplicate turn in fresh replay")
                        batch_id, prompt, response = stored
                        stage = "post_flight"
                        evaluate_turn(batch_id=str(batch_id), prompt=prompt,
                                      response=response, conversation_id=str(conv.id),
                                      model_used=background_model)
                        db.expire_all()
                        row = db.query(EpisodicMemory).filter_by(idempotency_key=key).one()
                        if row.source_spans is None or row.ts_provenance != "original":
                            raise RuntimeError("writer lost source roles or original timestamp")
                        seen_source_times[str(row.id)] = turn["timestamp"]
                        seen_source_times[str(row.batch_id)] = turn["timestamp"]
                        turn_source_ids[(slug, number)] = {str(row.id), str(row.batch_id)}
                        source_counts = {
                            "claims": db.execute(text("SELECT count(*) FROM codex_claims WHERE source_batch=:b"),
                                                 {"b": row.batch_id}).scalar(),
                            "edges_first_source": db.execute(text(
                                "SELECT count(*) FROM codex_edges WHERE source_batch=:b"),
                                {"b": row.batch_id}).scalar(),
                            "chunks": db.execute(text(
                                "SELECT count(*) FROM episodic_chunks WHERE turn_id=:id"),
                                {"id": row.id}).scalar(),
                        }
                        sink.write(json.dumps({"event": "written", "conversation": slug,
                                               "turn": number, "episodic_id": str(row.id),
                                               "batch_id": str(batch_id),
                                               "session_id": str(row.session_id),
                                               "lossless": row.lossless_flag,
                                               "inject_raw": row.inject_raw,
                                               "summary_coverage": row.summary_coverage,
                                               "summary_support": row.representation_verification,
                                               "source_counts": source_counts},
                                              default=str) + "\n")
                        if not args.no_maintenance:
                            stage = "maintenance"
                            due_maintenance(db, sink, turn["timestamp"], conv.id,
                                            last_maintenance)
                        for probe in probes.get((slug, number), ()):
                            stage = "as_of_probe"
                            question = probe["question"]
                            ppre = pp.build(db, question, conv.id, classifier, embedder)
                            control_pre = {arm: copy.deepcopy(ppre) for arm in
                                           ("vector_only", "recent_only")}
                            pmemory, pgenerated, pbudgeted = prepare_with_trace(
                                db, ppre, classifier)
                            probe_stage = stage_record(ppre, pmemory, pgenerated,
                                                       pbudgeted, include_prompt=True)
                            probe_stage["lineage"] = lineage_at_cutoff(
                                probe_stage, seen_source_times, turn["timestamp"])
                            gold_ids = {gold_turn: turn_source_ids[(slug, gold_turn)]
                                        for gold_turn in probe["gold_turns"]}
                            probe_stage["gold_fragment_coverage"] = (
                                gold_fragment_coverage(probe_stage, gold_ids))
                            controls = {}
                            if not args.no_probe_controls:
                                for arm, arm_pre in control_pre.items():
                                    cmemory, cgenerated, cbudgeted = prepare_with_trace(
                                        db, arm_pre, classifier, arm=arm)
                                    control_stage = stage_record(
                                        arm_pre, cmemory, cgenerated, cbudgeted,
                                        include_prompt=True)
                                    control_stage["lineage"] = lineage_at_cutoff(
                                        control_stage, seen_source_times,
                                        turn["timestamp"])
                                    control_stage["gold_fragment_coverage"] = (
                                        gold_fragment_coverage(control_stage, gold_ids))
                                    controls[arm] = control_stage
                            gold_sources = [{"turn": gold_turn,
                                             "recorded_at": turns[gold_turn - 1]["timestamp"].isoformat(),
                                             "prompt": turns[gold_turn - 1]["prompt"],
                                             "response": turns[gold_turn - 1]["response"],
                                             "source_ids": sorted(ids)}
                                            for gold_turn, ids in gold_ids.items()]
                            sink.write(json.dumps({
                                "event": "as_of_probe", "probe_id": probe.get("probe_id"),
                                "catalog_probe_id": probe.get("catalog_probe_id"),
                                "type": probe["probe_type"], "conversation": slug,
                                "split_turn": number,
                                "source_split_turn": probe["source_split_turn"],
                                "cutoff_kind": probe["cutoff_kind"],
                                "label_status": probe["label_status"],
                                "question": question,
                                "expected_answer": probe.get("expected_answer"),
                                "gold_turns": probe["gold_turns"],
                                "gold_sources": gold_sources,
                                "source_excerpt": probe.get("source_excerpt"),
                                "source_role": probe.get("source_role"),
                                "anchor_entity": probe.get("anchor_entity"),
                                "superseded_by": probe.get("superseded_by"),
                                "state_turns": db.query(EpisodicMemory).filter_by(
                                    conversation_id=conv.id).count(),
                                "preflight": probe_stage, "controls": controls},
                                default=str) + "\n")
                            probe_count += 1
                        sink.flush()
                        completed[slug] += 1
                    except Exception as exc:
                        db.rollback()
                        sink.write(json.dumps({"event": "failed", "conversation": slug,
                                               "turn": number, "stage": stage,
                                               "error_type": type(exc).__name__,
                                               "error": str(exc)[:500]}) + "\n")
                        sink.flush()
                        raise
            expected_probes = sum(map(len, probes.values()))
            complete_corpus = (dict(completed) == EXPECTED and not args.no_maintenance
                               and not args.no_probe_controls and args.limit == 0
                               and args.probe_panel != "immediate"
                               and probe_count == expected_probes)
            table_counts = {name: db.execute(text(f'SELECT count(*) FROM "{name}"')).scalar()
                            for name in Base.metadata.tables}
            sink.write(json.dumps({"event": "complete", "complete_selected_corpus": complete_corpus,
                                   "turns_by_conversation": dict(completed),
                                   "as_of_probes": probe_count,
                                   "expected_as_of_probes": expected_probes,
                                   "probe_panel": args.probe_panel,
                                   "table_counts": table_counts,
                                   "answer_quality_scored": False}) + "\n")
        print(f"v3 {'full selected-corpus' if complete_corpus else 'development/partial'} replay: {output} "
              f"({sum(completed.values())} recorded replies; {probe_count} as-of prompts; "
              "no new answer score)")
        return 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate corpus/probe mapping without models or DB")
    parser.add_argument("--out", default="logs/z1-v3-seed.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="development prefix per conversation; not a full seed")
    parser.add_argument("--conversation", choices=sorted(EXPECTED),
                        help="development diagnostic for one conversation; not a full seed")
    parser.add_argument("--no-maintenance", action="store_true",
                        help="development diagnostic only; leaves summary/cluster legs incomplete")
    parser.add_argument("--no-probe-controls", action="store_true",
                        help="development diagnostic only; omits matched as-of controls")
    parser.add_argument("--probe-panel",
                        choices=["existing", "typed_delayed", "all", "immediate"],
                        default="existing",
                        help="existing native/generated long-term; delayed typed optional")
    args = parser.parse_args()
    timing = "immediate" if args.probe_panel == "immediate" else "delayed"
    conversations, probes, quarantined, ineligible = load_plan(
        probe_timing=timing,
        include_typed=args.probe_panel in {"typed_delayed", "all", "immediate"},
        include_native=args.probe_panel in {"existing", "all"},
        include_generated=args.probe_panel in {"existing", "all"},
        capture_unlabeled_native=args.probe_panel in {"existing", "all"})
    native_count = sum(p["cutoff_kind"] == "native_checkpoint"
                       for group in probes.values() for p in group)
    generated_count = sum(p["cutoff_kind"] == "generated_at_checkpoint"
                          for group in probes.values() for p in group)
    unlabeled_count = sum(p["cutoff_kind"] == "native_unlabeled_checkpoint"
                          for group in probes.values() for p in group)
    print(f"full source: {sum(map(len, conversations.values()))} turns / {len(conversations)} conversations; "
          f"typed {timing} probes: {sum(map(len, probes.values())) - native_count - generated_count - unlabeled_count}; "
          f"native long-term checkpoints: {native_count}; "
          f"generated source-first checkpoints: {generated_count}; "
          f"unscored native section prompts: {unlabeled_count}; "
          f"no delayed window: {len(ineligible)}; "
          f"quarantined other cutoffs: {len(quarantined)}")
    if args.check:
        return 0
    return run(args, conversations, probes)


if __name__ == "__main__":
    raise SystemExit(main())
