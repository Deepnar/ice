"""Shared final-prompt path for conservative source-backed memory refinement."""

from dataclasses import dataclass

import structlog

from src.api.config import settings
from src.api.memory_decision import estimate_recent_window_tokens
from src.api.prompt_assembler import conversation_summary_block
from src.api.prompt_budget import BudgetedPrompt, assemble_budgeted_prompt, recent_row_ids
from src.api.source_need import judge_source_need, source_action
from src.api.source_proof import freeze_source_question, prove_source_fact
from src.memory.models import ConversationSummary, EpisodicMemory
from src.memory.usage import evidence_after_eviction
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

logger = structlog.get_logger("ice.api.memory_preparation")


@dataclass
class PreparedMemory:
    prepared: BudgetedPrompt
    fragments: list
    retrieve: bool
    action: str
    source_checked: bool = False


def _source_rows(fragment):
    """The stored turn rows a piece came from: an episodic row, or a fact's sources."""
    rows = tuple(getattr(fragment, "fact_source_row_ids", ()) or ())
    if not rows and fragment.source_type == "episodic" and fragment.source_batch_id:
        rows = (fragment.source_batch_id,)
    return tuple(str(r) for r in rows)


def conversation_order(db, fragments):
    """Selected pieces in the order their turns happened (retrieval_evidence_order).

    Runs after the budget has chosen the pieces, so it changes only where each one is
    written, never which are kept (prompt eviction removes whole blocks). A piece dated by
    its earliest source turn; undated pieces (notes, procedures) first, in their given
    order. Z2 F10: best-first order put a later turn ahead of an earlier one and the answer
    reversed which came first (7 of 21 on B)."""
    ids = {r for f in fragments for r in _source_rows(f)}
    times = {str(i): t for i, t in db.query(EpisodicMemory.id, EpisodicMemory.timestamp)
             .filter(EpisodicMemory.id.in_(ids)).all()} if ids else {}

    undated, dated = [], []
    for index, fragment in enumerate(fragments):
        stamps = [times[r] for r in _source_rows(fragment) if times.get(r) is not None]
        (dated.append((min(stamps), index, fragment)) if stamps else undated.append(fragment))
    return undated + [fragment for _, _, fragment in sorted(dated, key=lambda d: d[:2])]


def _shown_recent_rows(dry, retrieval_budget):
    """Rows the final prompt's recent window will show, which retrieval then skips
    (Z2, 2026-10-19: 13.8% of evidence tokens on B turns 101-150 re-showed recent
    turns; skipping them at retrieval lets the budget refill with older memory).

    Only when the recent window cannot be evicted: it is shed BEFORE evidence on
    overflow (context_ledger.DEFAULT_EVICT_ORDER), so a skipped turn would vanish
    from the prompt. That is ruled out when the evidence-free prompt plus the whole
    retrieval budget already fits; otherwise nothing is skipped (said once here)."""
    from src.memory.tokens import with_margin
    rows = recent_row_ids(getattr(dry, "artifact_blocks", None))
    ledger = getattr(dry, "ledger", None)
    if not rows:
        return frozenset()
    try:
        safe = (retrieval_budget is not None and ledger.serving_window > 0
                and with_margin(ledger.total() + retrieval_budget, ledger.safety_margin) <= ledger.available())
    except AttributeError:
        safe = False
    if not safe:
        logger.warning("recent_rows_not_excluded", rows=len(rows), retrieval_budget=retrieval_budget,
                       reason="evidence could overflow the prompt and evict the recent window")
        return frozenset()
    return frozenset(rows)


def prepare_memory_context(*, db, classifier, classification, conversation_id,
                           user_message, scope, turn_count, total_tokens,
                           total_budget, serving_window, base_retrieve,
                           memory_slots, bookmarked_texts,
                           session_start_text=None, constraints_text=None,
                           bookmarked_artifacts=None):
    """Assemble before judging; candidate fetches are not final exposures.

    Used by the chat route and available to answer/replay instruments. The
    caller still owns graph-access recording and downstream answer generation.
    """
    prompt_embedding = None
    original_context = classification.context_reliance
    summary_exists = db.query(ConversationSummary.conversation_id).filter_by(
        conversation_id=conversation_id).first() is not None

    def embedding():
        nonlocal prompt_embedding
        if prompt_embedding is None:
            vector = classifier.embedder.encode(user_message, convert_to_tensor=False)
            prompt_embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        return prompt_embedding

    def assemble(fragments, recent_budget):
        if summary_exists and total_tokens > recent_budget:
            embedding()
        artifact_options = {}
        choices = conversation_summary_block(
            db, str(conversation_id), turn_count, total_tokens, recent_budget,
            prompt_embedding, include_options=True, include_source_ids=True,
            artifact_options=artifact_options) or []
        options = [text for text, _ in choices]
        source_ids = {text: ids for text, ids in choices}
        return assemble_budgeted_prompt(
            serving_window=serving_window or 0,
            generation_reserve=settings.context_generation_reserve,
            safety_margin=settings.token_count_safety_margin,
            memory_slots=memory_slots, retrieved_fragments=fragments,
            user_message=user_message, db_session=db,
            conversation_id=str(conversation_id), bookmarked_texts=bookmarked_texts,
            classification=classification, scope=scope,
            max_recent_tokens=recent_budget, session_start_text=session_start_text,
            conversation_summary_text=options[0] if options else None,
            constraints_text=constraints_text, conversation_summary_options=options,
            conversation_summary_source_ids=source_ids,
            conversation_summary_artifact_options=artifact_options,
            bookmarked_artifacts=bookmarked_artifacts)

    recent_budget = estimate_recent_window_tokens(turn_count, total_budget)
    base_prompt = assemble([], recent_budget)
    if not base_prompt.ledger.fits():
        return PreparedMemory(base_prompt, [], False, "required_overflow")
    action = "keep"
    source_checked = False
    if (not base_retrieve and settings.memory_source_gate_enabled
            and settings.memory_source_rescue_enabled):
        # Only a negative is reconsidered, against ALL of the answerer's visible
        # context; a positive is never cancelled (source_need's docstring, Z2 F1).
        verdict = judge_source_need(base_prompt.messages)
        source_checked = True
        action = source_action(verdict)
    retrieve = base_retrieve or action == "rescue"
    if not retrieve:
        logger.info("memory_source_refinement", action=action,
                    base_retrieve=base_retrieve, retrieve=False)
        return PreparedMemory(base_prompt, [], False, action, source_checked)

    classification.context_reliance = "Long_Term_Memory"
    orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)
    orchestrator.set_budget_from_turn_count(
        turn_count, total_tokens=total_tokens, classification=classification,
        total_budget=total_budget)
    final_recent = getattr(orchestrator, "recent_token_budget", recent_budget)
    shown = _shown_recent_rows(base_prompt if final_recent == recent_budget else assemble([], final_recent),
                               getattr(orchestrator, "max_retrieval_tokens", None))
    fragments = orchestrator.retrieve(
        classification=classification, conversation_id=str(conversation_id),
        prompt_embedding=embedding(), scope=scope, defer_exposure=True,
        **({"exclude_rows": shown} if shown else {}))
    if settings.retrieval_evidence_order == "conversation" and fragments:
        fragments = conversation_order(db, fragments)
    prepared = assemble(fragments, final_recent)
    selected = evidence_after_eviction(prepared.visible_fragments, prepared.removed)
    if action == "rescue":
        supported = False
        framed = None
        if selected and prepared.ledger.fits():
            # Only complete original turns qualify this candidate branch.
            # Generated/legacy graph prose must not prove its own assertion.
            for fragment in selected:
                if fragment.source_type != "episodic" or not fragment.covers_entire_source:
                    continue
                if framed is None:
                    framed = freeze_source_question(user_message)
                proof = prove_source_fact(user_message, fragment.text, framed=framed)
                if proof.status == "supported":
                    # One proven factual source cannot license unrelated
                    # candidates. Pack and credit only its surviving original.
                    prepared = assemble([fragment], orchestrator.recent_token_budget)
                    selected = evidence_after_eviction(prepared.visible_fragments,
                                                       prepared.removed)
                    supported = bool(selected and prepared.ledger.fits())
                    break
        if not supported:
            classification.context_reliance = original_context
            logger.warning("memory_source_rescue_withheld", candidates=len(fragments),
                           surviving=len(selected), reason="no_qualified_original_source_proof")
            return PreparedMemory(base_prompt, [], False, "rescue_withheld", source_checked)
    if prepared.ledger.fits():
        orchestrator.record_exposure(selected)
    logger.info("memory_source_refinement", action=action,
                base_retrieve=base_retrieve, retrieve=True,
                candidates=len(fragments), admitted=len(selected))
    return PreparedMemory(prepared, selected, True, action, source_checked)
