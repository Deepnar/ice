"""Shared final-prompt path for conservative source-backed memory refinement."""

from dataclasses import dataclass

import structlog

from src.api.config import settings
from src.api.memory_decision import estimate_recent_window_tokens
from src.api.prompt_assembler import conversation_summary_block
from src.api.prompt_budget import BudgetedPrompt, assemble_budgeted_prompt
from src.api.source_need import judge_source_need, source_action
from src.api.source_proof import prove_source_fact
from src.memory.models import ConversationSummary
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


def prepare_memory_context(*, db, classifier, classification, conversation_id,
                           user_message, scope, turn_count, total_tokens,
                           total_budget, serving_window, base_retrieve,
                           memory_slots, bookmarked_texts,
                           session_start_text=None, constraints_text=None):
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
        options = conversation_summary_block(
            db, str(conversation_id), turn_count, total_tokens, recent_budget,
            prompt_embedding, include_options=True) or []
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
            constraints_text=constraints_text, conversation_summary_options=options)

    recent_budget = estimate_recent_window_tokens(turn_count, total_budget)
    base_prompt = assemble([], recent_budget)
    if not base_prompt.ledger.fits():
        return PreparedMemory(base_prompt, [], False, "required_overflow")
    action = "keep"
    source_checked = False
    if (settings.memory_source_gate_enabled
            and (base_retrieve or settings.memory_source_rescue_enabled)):
        # A positive can be suppressed only by evidence in the current message
        # alone. A negative must consider ALL of the answerer's visible context.
        evidence = [base_prompt.messages[-1]] if base_retrieve else base_prompt.messages
        verdict = judge_source_need(evidence, current_only=base_retrieve)
        source_checked = True
        action = source_action(base_retrieve, verdict, user_message)
    retrieve = (base_retrieve and action != "skip") or action == "rescue"
    if not retrieve:
        logger.info("memory_source_refinement", action=action,
                    base_retrieve=base_retrieve, retrieve=False)
        return PreparedMemory(base_prompt, [], False, action, source_checked)

    classification.context_reliance = "Long_Term_Memory"
    orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)
    orchestrator.set_budget_from_turn_count(
        turn_count, total_tokens=total_tokens, classification=classification,
        total_budget=total_budget)
    fragments = orchestrator.retrieve(
        classification=classification, conversation_id=str(conversation_id),
        prompt_embedding=embedding(), scope=scope, defer_exposure=True)
    prepared = assemble(fragments, getattr(orchestrator, "recent_token_budget", recent_budget))
    selected = evidence_after_eviction(fragments, prepared.removed)
    if action == "rescue":
        supported = False
        if selected and prepared.ledger.fits():
            # Only complete original turns qualify this candidate branch.
            # Generated/legacy graph prose must not prove its own assertion.
            for fragment in selected:
                if fragment.source_type != "episodic" or not fragment.covers_entire_source:
                    continue
                proof = prove_source_fact(user_message, fragment.text)
                if proof.status == "supported":
                    supported = True
                    break
        if not supported:
            classification.context_reliance = original_context
            logger.warning("memory_source_rescue_withheld", candidates=len(fragments),
                           surviving=len(selected), reason="no_qualified_retrieved_quote")
            return PreparedMemory(base_prompt, [], False, "rescue_withheld", source_checked)
    if prepared.ledger.fits():
        orchestrator.record_exposure(selected)
    logger.info("memory_source_refinement", action=action,
                base_retrieve=base_retrieve, retrieve=True,
                candidates=len(fragments), admitted=len(selected))
    return PreparedMemory(prepared, selected, True, action, source_checked)
