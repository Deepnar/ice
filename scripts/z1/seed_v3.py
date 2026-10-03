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
import importlib
import json
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url

from scripts.z1 import production_parity as pp
from scripts.z1.historical_clock import historical_clock
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
# The text-export histories have a constructed five-minute simulation clock.
# Only the provider-export history has verified original message timestamps.
SOURCE_TIMESTAMP_PROVENANCE = {"bb558b5f": "synthetic_raw_import",
                               "ecc64aab": "synthetic_raw_import",
                               "355a5709": "original"}
MARK = "z1seed"
PROMPT_ARMS = ("full", "no_codex", "vector_only", "recent_only")
REPLAY_SETTINGS = [
    "recent_window_max_turns", "classifier_model_path", "label_schema_path",
    "embedding_model_name", "background_model_mode", "background_model_name",
    "codex_extraction_model", "codex_extraction_mode", "background_ner_model",
    "memory_source_gate_enabled", "memory_source_rescue_enabled",
    "context_use_serving_window", "context_budget_min", "context_budget_max",
    "context_budget_floor", "context_generation_reserve",
    "retrieval_rrf_k", "retrieval_leg_base_weights", "retrieval_leg_profiles",
    "retrieval_leg_topic_overrides", "retrieval_rerank_enabled",
    "retrieval_rerank_model", "retrieval_rerank_revision",
    "retrieval_rerank_candidates", "retrieval_coverage_enabled",
    "retrieval_set_floor_enabled", "retrieval_bm25_candidate_limit",
    "retrieval_vector_candidate_limit", "retrieval_chunk_candidate_limit",
    "retrieval_collapse_enabled", "retrieval_max_frags_per_turn",
    "maintenance_intervals", "batch_summary_age_days",
    "runtime_cycles_cap", "decay_daily_unaccessed", "decay_daily_accessed",
    "decay_daily_creative", "codex_decay_daily", "codex_retention_floor",
    "procedural_stale_days", "procedural_min_reinforcement",
    "compaction_event_threshold",
    "retrieval_strengthen_writes", "decay_strengthen_amount",
    "codex_retention_increment", "codex_retention_cap",
]
MEMORY_JOBS = (
    "cluster_assignment", "cluster_merge", "conversation_summary", "batch_summarize",
    "reflection", "maintenance_agent", "decay_episodic", "decay_codex",
    "decay_procedural", "compaction",
)
SEMANTIC_TABLES = (
    "episodic_memory", "cold_storage", "context_clusters", "episodic_cluster_links",
    "codex_entities", "codex_edges", "codex_claim_links", "procedural_memory",
    "batch_summaries", "batch_notes", "conversation_summaries", "conversation_notes",
    "memory_slots", "session_summaries", "review_queue",
)
JOB_OBSERVATION_TABLES = {
    "decay_episodic": ("episodic_memory", "cold_storage"),
    "decay_codex": ("codex_edges",),
    "decay_procedural": ("procedural_memory",),
    "compaction": ("codex_events", "codex_snapshots"),
}


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
              capture_unlabeled_native=False, development_repeat_review=None):
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
                                    "ts_provenance": SOURCE_TIMESTAMP_PROVENANCE[slug],
                                    "prompt": row["prompt"],
                                    "response": row["response"]})
    for slug, rows in conversations.items():
        if len(rows) != EXPECTED[slug]:
            raise ValueError(f"{slug}: expected {EXPECTED[slug]} turns, got {len(rows)}")
        if any(a["timestamp"] >= b["timestamp"] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"{slug}: source timestamps not strictly ordered")
    ordered_slugs = list(conversations)
    for earlier, later in zip(ordered_slugs, ordered_slugs[1:]):
        if conversations[earlier][-1]["timestamp"] >= conversations[later][0]["timestamp"]:
            raise ValueError("selected conversation histories overlap or are out of order")

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
    if development_repeat_review:
        from scripts.z1.development_repeats import packet_hash, repeat_probes
        path = Path(development_repeat_review)
        packet_hash(path)
        extra = repeat_probes(json.loads(path.read_text()), conversations, at_turn,
                              [file_digest(p) for p in (CORPUS, UNIFIED, TYPED, DERIVED, GENERATED)], recent_gap)
        for probe in extra:
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


def visible_source_note_ids(note_options, messages):
    """Resolve the single summary option that survived final prompt eviction."""
    rendered = "\n".join(str(message.get("content", "")) for message in messages)
    visible = [(len(note), ids) for note, ids in note_options.items()
               if note and note in rendered]
    longest = max((size for size, _ in visible), default=0)
    matches = [ids for size, ids in visible if size == longest]
    if len(matches) > 1:
        raise RuntimeError("ambiguous source-note option in final prompt")
    return matches[0] if matches else []


def prepare_with_trace(db, pre, classifier, *, arm="full", source_time=None,
                       graph_exposure=False):
    """Shared final preparation plus generated-leg and budget-stage evidence."""
    from src.api import memory_preparation as mp
    from src.api import prompt_assembler as pa
    from src.api.prompt_assembler import bookmarked_turn_texts
    from src.model_registry.registry import get_model_context_window
    from src.model_registry.runtime_probe import serving_window
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator

    if arm not in PROMPT_ARMS:
        raise ValueError(f"unknown prompt arm: {arm}")

    produced, ranked_candidates, budgeted = {}, [], []
    note_options = {}
    methods = {"bm25": "_bm25_episodic", "vector": "_vector_episodic",
               "codex_graph": "_codex_graph", "codex_claims": "_codex_claims",
               "procedural": "_procedural_lookup", "batch_summary": "_batch_summary_lookup",
               "cold": "_cold_lookup"}

    def factory(session, embedder):
        orchestrator = HybridRetrievalOrchestrator(session, embedder)
        if arm != "full":
            disabled = ({"codex_graph", "codex_claims"} if arm == "no_codex"
                        else set(methods) - {"vector"} if arm == "vector_only"
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
        original_budget = orchestrator._enforce_token_budget
        retrieval_calls = 0

        def capture_ranking(fragments, *args, **kwargs):
            # The input is after fusion, bonuses, optional reranking, dedup and
            # coverage. It is the meaningful pre-budget top-k list; the result
            # is an admission order shaped by leg diversity, not a rank list.
            ranked_candidates.extend(fragment(f) for f in fragments)
            return original_budget(fragments, *args, **kwargs)

        orchestrator._enforce_token_budget = capture_ranking

        def captured(*args, **kwargs):
            nonlocal retrieval_calls
            retrieval_calls += 1
            if retrieval_calls != 1:
                raise RuntimeError("one preparation unexpectedly retrieved more than once")
            rows = original_retrieve(*args, **kwargs)
            budgeted.extend(fragment(f) for f in rows)
            return rows

        orchestrator.retrieve = captured
        return orchestrator

    raw_window = get_model_context_window(pre.model_name)
    window = (serving_window(pre.model_name, raw_window)
              if settings.context_use_serving_window else raw_window)
    original_notes = mp.conversation_summary_block

    def capture_notes(*args, **kwargs):
        if arm not in {"full", "no_codex"}:
            return None
        choices = original_notes(*args, **kwargs)
        if kwargs.get("include_source_ids") and choices:
            note_options.update({text: list(ids) for text, ids in choices})
        return choices
    source_gate_patch = (patch.object(settings, "memory_source_gate_enabled", False)
                         if arm == "recent_only" else nullcontext())
    if source_time is not None:
        if source_time.tzinfo is None:
            raise ValueError("historical prompt time must be timezone-aware")

        class SourceDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return (source_time.astimezone(tz) if tz is not None
                        else source_time.replace(tzinfo=None))

        prompt_clock = patch.object(pa, "datetime", SourceDateTime)
        retrieval_clock = patch.object(sys.modules[HybridRetrievalOrchestrator.__module__],
                                       "datetime", SourceDateTime)
    else:
        prompt_clock = retrieval_clock = nullcontext()
    with (patch.object(mp, "HybridRetrievalOrchestrator", factory),
          patch.object(mp, "conversation_summary_block", capture_notes),
          source_gate_patch, prompt_clock, retrieval_clock):
        memory = pp.prepare(
            db, pre, classifier,
            memory_slots=(db.query(MemorySlot).filter_by(is_active=True).all()
                          if arm in {"full", "no_codex"} else []),
            bookmarked_texts=(bookmarked_turn_texts(db, pre.conversation_id)
                              if arm in {"full", "no_codex"} else []),
            serving_window=window)
        if graph_exposure:
            from src.memory.usage import record_graph_access
            record_graph_access(db, memory.fragments, stage="prompt_prepared")
    if not memory.prepared.ledger.fits():
        raise RuntimeError("final prompt exceeds serving window")
    if arm == "recent_only" and memory.fragments:
        raise RuntimeError("recent-only control still contains retrieved fragments")
    if arm == "vector_only" and any(f.leg not in ("vector", "bm25+vector")
                                    for f in memory.fragments):
        raise RuntimeError("vector-only control contains another retrieval leg")
    note_source_ids = visible_source_note_ids(note_options, memory.prepared.messages)
    return memory, produced, ranked_candidates, budgeted, note_source_ids


def stage_record(pre, memory, produced, ranked_candidates, budgeted,
                 note_source_ids, *, include_prompt):
    selected = [fragment(f, include_text=True) for f in memory.fragments]
    messages = memory.prepared.messages
    result = {"gate": pp.provenance_fields(pre, memory),
              "topic_tags": pre.classification.topic_tags,
              "intent_tags": pre.classification.intent_tags,
              "generated_by_leg": produced, "ranked_candidates": ranked_candidates,
              "budgeted": budgeted,
              "selected": selected, "evicted": memory.prepared.removed,
              "source_note_ids": note_source_ids,
              "exposure_writes_enabled": settings.retrieval_strengthen_writes,
              "prompt_block_counts": memory.prepared.item_counts,
              "context_ledger": memory.prepared.ledger.snapshot(),
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
    for source_id in stage.get("source_note_ids") or []:
        when = seen_source_times.get(source_id)
        if when is None:
            unresolved.add(source_id)
        elif when > cutoff:
            future.add(source_id)
    result = {"selected_with_source_identity": with_source,
              "selected_total": len(stage["selected"]),
              "visible_source_note_ids": len(stage.get("source_note_ids") or []),
              "unresolved_source_ids": sorted(unresolved),
              "future_source_ids": sorted(future)}
    if future:
        raise RuntimeError("future source reached an as-of prompt")
    return result


def gold_fragment_coverage(stage, gold_ids, source_turn_by_id=None):
    """Count source identity and pre-budget fragment rank, never answer use."""
    def ids(row):
        return ({row["source_row"]} if row["source_row"] else set()) | set(
            row["origin_batches"])

    def hits(rows):
        returned = set().union(*(ids(row) for row in rows)) if rows else set()
        return sum(bool(source_ids & returned) for source_ids in gold_ids.values())

    ranked = stage.get("ranked_candidates", [])
    # A gold turn may produce several fragments; use its FIRST ranked fragment.
    # A graph fragment may cite several turns; each gets that fragment's rank.
    first_rank = {turn: next((rank for rank, row in enumerate(ranked, 1)
                              if source_ids & ids(row)), None)
                  for turn, source_ids in gold_ids.items()}
    known_rank = [rank for rank in first_rank.values() if rank is not None]
    source_turn_by_id = source_turn_by_id or {}
    distinct_turn_rank = {}
    seen_turns = set()
    for row in ranked:
        new_turns = ({source_turn_by_id[source_id] for source_id in ids(row)
                      if source_id in source_turn_by_id} - seen_turns)
        # Several sources in one fragment have one tied retrieval position.
        for turn_key in new_turns:
            distinct_turn_rank[turn_key] = len(seen_turns) + 1
        seen_turns.update(new_turns)
    gold_turn_rank = {}
    for turn, source_ids in gold_ids.items():
        keys = {source_turn_by_id[source_id] for source_id in source_ids
                if source_id in source_turn_by_id}
        gold_turn_rank[str(turn)] = min((distinct_turn_rank[key] for key in keys
                                        if key in distinct_turn_rank), default=None)
    known_turn_rank = [rank for rank in gold_turn_rank.values() if rank is not None]
    note_ids = set(stage.get("source_note_ids") or [])
    selected_ids = (set().union(*(ids(row) for row in stage["selected"]))
                    if stage["selected"] else set())
    note_hits = sum(bool(source_ids & note_ids) for source_ids in gold_ids.values())
    prompt_hits = sum(bool(source_ids & (note_ids | selected_ids))
                      for source_ids in gold_ids.values())
    return {"gold_turns": len(gold_ids),
            "generated": hits([row for rows in stage["generated_by_leg"].values()
                               for row in rows]),
            "generated_by_leg": {leg: hits(rows) for leg, rows in
                                 stage["generated_by_leg"].items()},
            "ranked_candidate_fragments": len(ranked),
            "ranked": len(known_rank),
            "rank_at_5": sum(rank <= 5 for rank in known_rank),
            "rank_at_10": sum(rank <= 10 for rank in known_rank),
            "first_rank_by_gold_turn": {str(turn): rank for turn, rank in first_rank.items()},
            "ranked_distinct_source_turns": len(seen_turns),
            "source_turn_rank_at_5": sum(rank <= 5 for rank in known_turn_rank),
            "source_turn_rank_at_10": sum(rank <= 10 for rank in known_turn_rank),
            "first_source_turn_rank_by_gold_turn": gold_turn_rank,
            "budgeted": hits(stage["budgeted"]),
            "selected": hits(stage["selected"]),
            "source_note": note_hits,
            "selected_or_source_note": prompt_hits,
            "interpretation": "first matching fragment and distinct source-turn ranks before budget; multi-source fragments tie, unsourced fragments lack turn rank; neither rank proves answer use or correctness"}


def memory_state(db):
    """Observable job effects, including updates that do not change row counts."""
    metrics = {
        "warm_turns": "SELECT count(*) FROM episodic_memory",
        "cold_turns": "SELECT count(*) FROM cold_storage",
        "archived_warm_turns": "SELECT count(*) FROM episodic_memory WHERE is_archived",
        "warm_decay_total": "SELECT COALESCE(sum(decay_score),0) FROM episodic_memory",
        "graph_strength_total": "SELECT COALESCE(sum(strength),0) FROM codex_edges",
        "active_procedural": "SELECT count(*) FROM procedural_memory WHERE is_active",
        "graph_snapshots": "SELECT count(*) FROM codex_snapshots",
        "compacted_graph_events": "SELECT count(*) FROM codex_events WHERE compacted",
        "batch_notes": "SELECT count(*) FROM batch_notes",
        "conversation_notes": "SELECT count(*) FROM conversation_notes",
        "clusters": "SELECT count(*) FROM context_clusters",
        "cluster_links": "SELECT count(*) FROM episodic_cluster_links",
        "pending_reviews": "SELECT count(*) FROM review_queue WHERE status='pending'",
    }
    query = " UNION ALL ".join(f"SELECT '{name}', ({sql})::double precision"
                               for name, sql in metrics.items())
    return dict(db.execute(text(query)).all())


def semantic_state(db, job):
    """Private inspection state; omit vectors and already frozen original bodies."""
    state = {}
    for name in JOB_OBSERVATION_TABLES.get(job, SEMANTIC_TABLES):
        table = Base.metadata.tables[name]
        columns = [col for col in table.columns
                   if col.name not in {"embedding", "raw_text"}]
        rows = db.execute(select(*columns)).mappings().all()
        values = {}
        for row in rows:
            # Normalize UUIDs/dates/nested arrays before comparing SQL reads.
            value = json.loads(json.dumps(dict(row), default=str))
            key = json.dumps([value[col.name] for col in table.primary_key],
                             separators=(",", ":"))
            if key in values:
                raise RuntimeError("duplicate semantic observation key: " + name)
            values[key] = value
        state[name] = values
    return state


def semantic_changes(before, after):
    """Keep exact changed outputs, including deletion and composite-key membership."""
    if set(before) != set(after):
        raise RuntimeError("semantic observation table set changed during job")
    changes = []
    for table in sorted(before):
        for key in sorted(before[table].keys() | after[table].keys()):
            old, new = before[table].get(key), after[table].get(key)
            if old != new:
                changes.append({"table": table, "key": json.loads(key),
                                "before": old, "after": new})
    return changes


def original_turn_count(db, conversation_id):
    return db.execute(text("""
        SELECT count(*) FROM (
          SELECT id FROM episodic_memory WHERE conversation_id=:conversation
          UNION SELECT id FROM cold_storage WHERE conversation_id=:conversation
        ) originals
    """), {"conversation": conversation_id}).scalar()


def source_storage_at_cutoff(db, conversation_id):
    """Freeze original locations so later native labels do not inspect the final store."""
    rows = db.execute(text("""
        SELECT id::text, batch_id::text, 'warm' AS tier, is_archived
          FROM episodic_memory WHERE conversation_id=:conversation
        UNION ALL
        SELECT id::text, batch_id::text, 'cold' AS tier, TRUE AS is_archived
          FROM cold_storage WHERE conversation_id=:conversation
    """), {"conversation": conversation_id}).all()
    state = {}
    for row in rows:
        if row.id in state:
            raise RuntimeError("original source exists in both warm and cold storage")
        state[row.id] = {"batch_id": row.batch_id, "tier": row.tier,
                         "is_archived": row.is_archived}
    return state


@contextmanager
def probe_observation(db):
    """Do not let development questions mutate the historical conversation."""
    db.commit()
    try:
        db.execute(text("SET TRANSACTION READ ONLY"))
        with patch.object(settings, "retrieval_strengthen_writes", False):
            yield
    finally:
        db.rollback()
        db.expire_all()


def due_maintenance(db, sink, source_time, conversation_id, last_run):
    """Exercise real periodic writers against as-of state, with stated cadence.

    Share the production registry, cadence, overdue ordering and cycle cap.
    The caller scopes clocks to source time. Jobs run serially at source turn
    boundaries, not through production's asynchronous scheduler and leases.
    """
    from src.workers.runtime import JOBS, missed_cycles, overdue_seconds

    due = []
    for name in MEMORY_JOBS:
        interval = settings.maintenance_intervals.get(name)
        previous = last_run.get(name)
        if not interval:
            continue
        overdue = overdue_seconds(source_time, interval, None, previous)
        if overdue > 0:
            due.append((overdue, name))
    for _, name in sorted(due, reverse=True):
        spec = JOBS[name]
        module, function = spec.path.split(":")
        fn = getattr(importlib.import_module(module), function)
        interval = settings.maintenance_intervals[name]
        previous = last_run.get(name)
        elapsed = (source_time - previous).total_seconds() if previous else interval
        kwargs = ({"cycles": missed_cycles(elapsed, interval, settings.runtime_cycles_cap)}
                  if spec.pass_cycles else {})
        try:
            observation_start = time.perf_counter()
            before = memory_state(db)
            semantic_before = semantic_state(db, name)
            observation_ms = (time.perf_counter() - observation_start) * 1000
            job_start = time.perf_counter()
            result = fn(db, **kwargs) if spec.needs_db else fn(**kwargs)
            job_elapsed_ms = (time.perf_counter() - job_start) * 1000
            db.expire_all()
            observation_start = time.perf_counter()
            after = memory_state(db)
            semantic_after = semantic_state(db, name)
            changes = semantic_changes(semantic_before, semantic_after)
            observation_ms += (time.perf_counter() - observation_start) * 1000
            sink.write(json.dumps({"event": "maintenance", "job": name,
                                   "trigger_conversation": str(conversation_id),
                                   "source_time_trigger": source_time.isoformat(),
                                   "call_kwargs": kwargs, "before": before,
                                   "after": after,
                                   "semantic_observed_tables": sorted(semantic_before),
                                   "semantic_changes": changes,
                                   "observer_elapsed_ms": observation_ms,
                                   "job_elapsed_ms": job_elapsed_ms,
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
    resume = getattr(args, "resume", False)
    every = getattr(args, "checkpoint_every", 10)
    if every < 1:
        raise ValueError("checkpoint interval must be positive")
    if output.exists() and not resume:
        raise RuntimeError("trace output already exists; use a new run name")
    if resume and not output.exists():
        raise ValueError("resume requires the original replay trace")
    db = SessionLocal()
    checkpoints = None
    try:
        nonempty = {name: db.execute(text(f'SELECT count(*) FROM "{name}"')).scalar()
                    for name in Base.metadata.tables}
        if any(nonempty.values()) and not resume:
            raise RuntimeError("replay requires an empty dedicated database; found populated ORM tables")
        db.rollback()
        from src.classifier.classifier import PyTorchClassifier
        from src.memory.embedder import get_embedder
        from src.workers.bg_client_factory import get_bg_model_name
        from src.workers.post_flight import evaluate_turn
        from scripts.z1.replay_checkpoint import background_model_digests

        embedder = get_embedder()
        classifier = PyTorchClassifier(model_path=settings.classifier_model_path,
                                       schema_path=settings.label_schema_path)
        background_model = get_bg_model_name()
        repeat_path = getattr(args, "development_repeat_review", None)
        from scripts.z1.development_repeats import POLICY, packet_hash
        meta = run_meta(script=__file__, args=vars(args),
                        settings_keys=REPLAY_SETTINGS,
                        inputs=[file_digest(CORPUS), file_digest(UNIFIED),
                                file_digest(TYPED), file_digest(DERIVED),
                                file_digest(GENERATED),
                                file_digest(settings.classifier_model_path),
                                file_digest(settings.label_schema_path),
                                *([file_digest(repeat_path)] if repeat_path else [])],
                        extra={"version": "v3", "expected_turns": EXPECTED,
                               "development_repeat_review_sha256": packet_hash(Path(repeat_path)) if repeat_path else None,
                               "development_repeat_policy": POLICY if repeat_path else None,
                               "planned_probes": [[p["probe_id"], slug, cutoff]
                                                  for (slug, cutoff), group in probes.items()
                                                  for p in group],
                               "resolved_background_model": background_model,
                               "writer_model_digests": background_model_digests(settings, background_model),
                               "recorded_answers_not_generated": True,
                               "recorded_reply_model": "unknown; existing corpus reply is supplied after preflight",
                               "prompt_arms": list(PROMPT_ARMS),
                               "probe_state_policy": "read-only transaction, exposure writes disabled, rollback after each diagnostic probe",
                               "clock_policy": "source-time Python clocks/ORM defaults and explicit SQL NOW() within isolated turn; real elapsed model/network timers",
                               "timestamp_provenance_by_conversation": SOURCE_TIMESTAMP_PROVENANCE,
                               "periodic_jobs": list(MEMORY_JOBS),
                               "maintenance_schedule": "shared registry/cadence/overdue-order/cycle cap; serial source-turn boundaries",
                               "unexercised_during_replay": [
                                   "asynchronous_runtime", "idle_session_burst",
                                   "project_document_paths", "HTTP_streaming",
                                   "session_model_stickiness"]})
        from scripts.z1.replay_checkpoint import Checkpoints, run_identity
        recovery_root = private_output(getattr(args, "checkpoint_dir", None)
                                       or str(output.with_suffix(".recovery")))
        checkpoints = Checkpoints(recovery_root, output, run_identity(meta, args, settings))
        if not resume and checkpoints.pointer.exists():
            raise ValueError("recovery directory already belongs to a run; use its original trace and --resume")
        if resume:
            if checkpoints.complete_unchanged(EXPECTED, nonempty):
                print("v3 complete replay already verified; trace bytes retained unchanged")
                return 0
        restored = checkpoints.recover() if resume else {}
        last_maintenance = {key: datetime.fromisoformat(value)
                            for key, value in restored.get("last_maintenance", {}).items()}
        completed = Counter(restored.get("completed", {}))
        probe_count = restored.get("probe_count", 0)
        seen_source_times = {key: datetime.fromisoformat(value)
                             for key, value in restored.get("seen_source_times", {}).items()}
        turn_source_ids = {(row[0], row[1]): set(row[2])
                           for row in restored.get("turn_source_ids", [])}
        source_turn_by_id = dict(restored.get("source_turn_by_id", {}))
        conversation_ids = dict(restored.get("conversation_ids", {}))

        def checkpoint_state():
            return {"last_maintenance": last_maintenance, "completed": dict(completed),
                    "probe_count": probe_count, "seen_source_times": seen_source_times,
                    "turn_source_ids": [[slug, turn, sorted(ids)]
                                        for (slug, turn), ids in turn_source_ids.items()],
                    "source_turn_by_id": source_turn_by_id,
                    "conversation_ids": conversation_ids}

        with output.open("a" if resume else "x") as sink:
            if not resume:
                sink.write(json.dumps({"event": "run", "meta": meta}) + "\n")
                checkpoints.capture(sink, checkpoint_state())
            else:
                sink.write(json.dumps({"event": "resume", "meta": meta,
                                       "completed_turns": dict(completed)}) + "\n")
            selected_conversations = ({args.conversation: conversations[args.conversation]}
                                      if args.conversation else conversations)
            for slug, turns in selected_conversations.items():
                if slug in conversation_ids:
                    conv = db.get(Conversation, uuid.UUID(conversation_ids[slug]))
                    if conv is None:
                        raise RuntimeError("checkpoint lost its conversation identity")
                else:
                    conv = Conversation(id=uuid.uuid4(), memory_scope_type="auto",
                                        created_at=turns[0]["timestamp"])
                    db.add(conv)
                    db.commit()
                    conversation_ids[slug] = str(conv.id)
                bound = turns[:args.limit] if args.limit else turns
                for turn in bound:
                    number = turn["turn_number"]
                    if number <= completed[slug]:
                        continue
                    key = f"{MARK}-{slug}-{number}"
                    stage = "preflight"
                    try:
                        with historical_clock(turn["timestamp"], db.get_bind()) as clock_stats:
                            pre = pp.build(db, turn["prompt"], conv.id, classifier, embedder,
                                           source_time=turn["timestamp"])
                            memory, generated, ranked, budgeted, note_ids = prepare_with_trace(
                                db, pre, classifier, source_time=turn["timestamp"],
                                graph_exposure=True)
                            preflight = stage_record(pre, memory, generated, ranked, budgeted,
                                                     note_ids,
                                                     include_prompt=True)
                            preflight["lineage"] = lineage_at_cutoff(
                                preflight, seen_source_times, turn["timestamp"])
                            before = original_turn_count(db, conv.id)
                            record = {"event": "turn", "conversation": slug, "turn": number,
                                      "recorded_at": turn["timestamp"].isoformat(),
                                      "ts_provenance": turn["ts_provenance"],
                                      "source_question_sha256": hashlib.sha256(turn["prompt"].encode()).hexdigest(),
                                      "before_turns": before,
                                      "preflight": preflight}
                            sink.write(json.dumps(record, default=str) + "\n")
                            sink.flush()
                            stage = "write"
                            frozen = _FrozenClassifier(pre.classification, turn["prompt"], conv.id)
                            stored = _store_turn(db, conv.id, turn["prompt"], turn["response"],
                                                 turn["timestamp"], key, "fresh",
                                                 turn["timestamp"], turn["ts_provenance"],
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
                            if row.source_spans is None or row.ts_provenance != turn["ts_provenance"]:
                                raise RuntimeError("writer lost source roles or timestamp provenance")
                            seen_source_times[str(row.id)] = turn["timestamp"]
                            seen_source_times[str(row.batch_id)] = turn["timestamp"]
                            turn_source_ids[(slug, number)] = {str(row.id), str(row.batch_id)}
                            source_turn_by_id[str(row.id)] = f"{slug}:{number}"
                            source_turn_by_id[str(row.batch_id)] = f"{slug}:{number}"
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
                                                   "ts_provenance": row.ts_provenance,
                                                   "lossless": row.lossless_flag,
                                                   "inject_raw": row.inject_raw,
                                                   "summary_coverage": row.summary_coverage,
                                                   "summary_text": row.summary_text,
                                                   "abstract_text": row.abstract_text,
                                                   "source_raw_sha256": hashlib.sha256(row.raw_text.encode()).hexdigest(),
                                                   "summary_support": row.representation_verification,
                                                   "source_counts": source_counts},
                                                  default=str) + "\n")
                            if not args.no_maintenance:
                                stage = "maintenance"
                                due_maintenance(db, sink, turn["timestamp"], conv.id,
                                                last_maintenance)
                            for probe in probes.get((slug, number), ()):
                                with probe_observation(db):
                                    stage = "as_of_probe"
                                    question = probe["question"]
                                    ppre = pp.build(db, question, conv.id, classifier, embedder,
                                                    source_time=turn["timestamp"])
                                    control_pre = {arm: copy.deepcopy(ppre) for arm in
                                                   PROMPT_ARMS[1:]}
                                    pmemory, pgenerated, prank, pbudgeted, pnote_ids = prepare_with_trace(
                                        db, ppre, classifier, source_time=turn["timestamp"])
                                    probe_stage = stage_record(ppre, pmemory, pgenerated, prank,
                                                               pbudgeted, pnote_ids,
                                                               include_prompt=True)
                                    probe_stage["lineage"] = lineage_at_cutoff(
                                        probe_stage, seen_source_times, turn["timestamp"])
                                    gold_ids = {gold_turn: turn_source_ids[(slug, gold_turn)]
                                                for gold_turn in probe["gold_turns"]}
                                    probe_stage["gold_fragment_coverage"] = (
                                        gold_fragment_coverage(probe_stage, gold_ids,
                                                               source_turn_by_id))
                                    controls = {}
                                    if not args.no_probe_controls:
                                        for arm, arm_pre in control_pre.items():
                                            cmemory, cgenerated, crank, cbudgeted, cnote_ids = prepare_with_trace(
                                                db, arm_pre, classifier, arm=arm,
                                                source_time=turn["timestamp"])
                                            control_stage = stage_record(
                                                arm_pre, cmemory, cgenerated, crank, cbudgeted,
                                                cnote_ids,
                                                include_prompt=True)
                                            control_stage["lineage"] = lineage_at_cutoff(
                                                control_stage, seen_source_times,
                                                turn["timestamp"])
                                            control_stage["gold_fragment_coverage"] = (
                                                gold_fragment_coverage(control_stage, gold_ids,
                                                                       source_turn_by_id))
                                            controls[arm] = control_stage
                                    gold_sources = [{"turn": gold_turn,
                                                     "recorded_at": turns[gold_turn - 1]["timestamp"].isoformat(),
                                                     "ts_provenance": turns[gold_turn - 1]["ts_provenance"],
                                                     "prompt": turns[gold_turn - 1]["prompt"],
                                                     "response": turns[gold_turn - 1]["response"],
                                                     "source_ids": sorted(ids)}
                                                for gold_turn, ids in gold_ids.items()]
                                    storage = source_storage_at_cutoff(db, conv.id)
                                    sink.write(json.dumps({
                                        "event": "as_of_probe", "probe_id": probe.get("probe_id"),
                                        "catalog_probe_id": probe.get("catalog_probe_id"),
                                        "type": probe["probe_type"], "conversation": slug,
                                        "split_turn": number,
                                        "question_time": turn["timestamp"].isoformat(),
                                        "question_time_provenance": turn["ts_provenance"],
                                        "source_split_turn": probe["source_split_turn"],
                                        "cutoff_kind": probe["cutoff_kind"],
                                        "label_status": probe["label_status"],
                                        "development_repeat": probe.get("development_repeat"),
                                        "catalog_expected_answer": probe.get("catalog_expected_answer", probe.get("expected_answer")),
                                        "question": question,
                                        "expected_answer": probe.get("expected_answer"),
                                        "gold_turns": probe["gold_turns"],
                                        "gold_sources": gold_sources,
                                        "source_storage_at_cutoff": storage,
                                        "source_excerpt": probe.get("source_excerpt"),
                                        "source_role": probe.get("source_role"),
                                        "anchor_entity": probe.get("anchor_entity"),
                                        "superseded_by": probe.get("superseded_by"),
                                        "state_turns": original_turn_count(db, conv.id),
                                        "preflight": probe_stage, "controls": controls},
                                        default=str) + "\n")
                                    probe_count += 1
                            sink.flush()
                            completed[slug] += 1
                            sink.write(json.dumps({"event": "clock", "conversation": slug,
                                                   "turn": number, **clock_stats}) + "\n")
                            sink.flush()
                        # Outside the historical clock: snapshot metadata uses real time.
                        if (sum(completed.values()) % every == 0
                                or probes.get((slug, number)) or number == len(bound)):
                            stage = "checkpoint"
                            db.rollback()
                            checkpoints.capture(sink, checkpoint_state(),
                                                evaluation_key=(f"{slug}-{number}"
                                                                if probes.get((slug, number)) else None))
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
            from scripts.z1.snapshot import fingerprints
            table_sha256 = fingerprints()
            sink.write(json.dumps({"event": "complete", "complete_selected_corpus": complete_corpus,
                                   "turns_by_conversation": dict(completed),
                                   "as_of_probes": probe_count,
                                   "expected_as_of_probes": expected_probes,
                                   "probe_panel": args.probe_panel,
                                   "table_counts": table_counts,
                                   "table_sha256": table_sha256,
                                   "answer_quality_scored": False}) + "\n")
        print(f"v3 {'full selected-corpus' if complete_corpus else 'development/partial'} replay: {output} "
              f"({sum(completed.values())} recorded replies; {probe_count} as-of prompts; "
              "no new answer score)")
        return 0
    finally:
        db.close()
        if checkpoints is not None:
            checkpoints.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate corpus/probe mapping without models or DB")
    parser.add_argument("--out", default="logs/z1-v3-seed.jsonl")
    parser.add_argument("--resume", action="store_true", help="restore the last durable store/trace checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="completed turns between recovery snapshots; also snapshots at question checkpoints")
    parser.add_argument("--checkpoint-dir", help="private recovery directory under logs/")
    parser.add_argument("--development-repeat-review", help="private reviewed recent-to-old development pairs")
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
        capture_unlabeled_native=args.probe_panel in {"existing", "all"},
        development_repeat_review=args.development_repeat_review)
    native_count = sum(p["cutoff_kind"] == "native_checkpoint"
                       for group in probes.values() for p in group)
    generated_count = sum(p["cutoff_kind"] == "generated_at_checkpoint"
                          for group in probes.values() for p in group)
    unlabeled_count = sum(p["cutoff_kind"] == "native_unlabeled_checkpoint"
                          for group in probes.values() for p in group)
    repeat_count = sum(bool(p.get("development_repeat")) for group in probes.values() for p in group)
    print(f"full source: {sum(map(len, conversations.values()))} turns / {len(conversations)} conversations; "
          f"typed {timing} probes: {sum(map(len, probes.values())) - native_count - generated_count - unlabeled_count - repeat_count}; "
          f"native long-term checkpoints: {native_count}; "
          f"generated source-first checkpoints: {generated_count}; "
          f"unscored native section prompts: {unlabeled_count}; "
          f"reviewed development repeat prompts: {repeat_count}; "
          f"no delayed window: {len(ineligible)}; "
          f"quarantined other cutoffs: {len(quarantined)}")
    if args.check:
        return 0
    try:
        return run(args, conversations, probes)
    finally:
        # This serial instrument has no runtime idle drain. Release owned
        # identities even on failure/Ctrl+C; never hide the original outcome.
        from src.workers.bg_client_factory import release_owned_models
        try:
            release_owned_models()
        except Exception as exc:
            print(f"WARNING: owned model cleanup failed: {type(exc).__name__}",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
