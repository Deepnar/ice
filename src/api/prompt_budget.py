"""Assemble, account, evict optional blocks, then verify the actual result."""
import hashlib
from dataclasses import dataclass, field

import structlog

from src.api.context_ledger import ContextLedger
from src.api.prompt_assembler import assemble_prompt
from src.memory.tokens import count as count_tokens

logger = structlog.get_logger("ice.api.prompt_budget")


@dataclass
class BudgetedPrompt:
    messages: list
    ledger: ContextLedger
    removed: list[str]
    item_counts: dict
    visible_fragments: list
    artifact_blocks: dict = field(default_factory=dict)


def _covered_by_active_original(fragment, source_ids, conversation_id, summary_text):
    if str(fragment.conversation_id) != conversation_id:
        return False
    if (fragment.source_type == 'episodic' and fragment.covers_entire_source
            and str(fragment.source_batch_id) in source_ids):
        # The indexed source-mode note is the complete original rendered with
        # role attribution; the episodic renderer uses a different wrapper.
        return True
    return bool(fragment.source_note_row_id in source_ids
                and fragment.source_note_body
                and fragment.source_note_body in (summary_text or ''))


def _graph_quotes_covered(fragment, originals):
    """Suppress only mapped graph evidence covered by actual complete originals."""
    units = getattr(fragment, 'source_quote_requirements', ())
    if (fragment.source_type != 'codex' or not units
            or getattr(fragment, 'source_quote_text_sha256', None)
            != hashlib.sha256(fragment.text.encode('utf-8')).hexdigest()):
        return False
    for row_id, text in units:
        if not text or not any(source.source_type == 'episodic'
                and source.covers_entire_source and str(source.source_batch_id) == row_id
                and text in source.text for source in originals):
            return False
    return True


def _fact_sources_in_prompt(fragment, originals, recent_ids):
    """CODEX_V4 D8: a fact adds nothing when EVERY source turn it rests on is
    already in the prompt — a complete retrieved original or a recent-window
    turn. Ids are episodic ROW ids on both sides (an episodic fragment's
    source_batch_id holds the row id). One visible source missing keeps it."""
    rows = getattr(fragment, 'fact_source_row_ids', ())
    if getattr(fragment, 'leg', None) != 'facts' or not rows:
        return False
    present = set(recent_ids) | {str(o.source_batch_id) for o in originals
                                 if o.source_type == 'episodic' and o.covers_entire_source}
    return all(row in present for row in rows)


def _recent_row_ids(artifacts):
    return {ident for item in (artifacts.get('recent_turns') or {}).get('items', [])
            for table, ident in item.get('stored_artifact_ids', ()) if table == 'episodic_memory'}


def assemble_budgeted_prompt(*, serving_window, generation_reserve,
                             safety_margin=1.0, **kwargs):
    arguments = dict(kwargs)
    summary_options = arguments.pop('conversation_summary_options', None) or []
    summary_sources = arguments.pop('conversation_summary_source_ids', None) or {}
    summary_artifacts = arguments.pop('conversation_summary_artifact_options', None) or {}
    source_fragments = list(arguments.get('retrieved_fragments') or [])
    evidence_enabled = True
    fact_skip = frozenset()
    removed = []
    evictions = []
    if not serving_window:
        logger.warning("context_window_unknown", reason="Provider capacity unavailable; prompt fit is unmeasured")
    while True:
        active_sources = set(summary_sources.get(arguments.get('conversation_summary_text'), ()))
        active_conv = str(arguments.get('conversation_id') or '')
        arguments['retrieved_fragments'] = ([
            fragment for fragment in source_fragments
            if not _covered_by_active_original(
                fragment, active_sources, active_conv,
                arguments.get('conversation_summary_text'))
        ] if evidence_enabled else [])
        # Recompute after each whole-note/eviction choice. Earlier suppression
        # cannot survive disappearance or degradation of its covering original.
        visible = [fragment for fragment in arguments['retrieved_fragments']
                   if not _graph_quotes_covered(fragment, arguments['retrieved_fragments'])]
        arguments['retrieved_fragments'] = [fragment for fragment in visible
            if not set(getattr(fragment, 'origin_fact_ids', ())) & fact_skip]
        costs, counts, artifacts = {}, {}, {}
        arguments['conversation_summary_artifacts'] = summary_artifacts.get(arguments.get('conversation_summary_text'), ())
        messages = assemble_prompt(**arguments, block_tokens=costs, block_counts=counts, block_artifacts=artifacts)
        # Recomputed from THIS assembly every round: if the recent window is
        # evicted later, the facts it made redundant come back (TRAPS 125).
        skip_now = frozenset(fid for fragment in visible
                             if _fact_sources_in_prompt(fragment, visible, _recent_row_ids(artifacts))
                             for fid in fragment.origin_fact_ids)
        if skip_now != fact_skip:
            if skip_now - fact_skip:
                logger.info("facts_skipped_source_in_prompt", count=len(skip_now))
            fact_skip = skip_now
            continue
        ledger = ContextLedger(serving_window=serving_window,
            generation_reserve=generation_reserve, safety_margin=safety_margin)
        for name, tokens in costs.items():
            ledger.add(name, tokens)
        ledger.evictions = list(evictions)
        if ledger.fits():
            return BudgetedPrompt(messages, ledger, removed, counts,
                                  arguments['retrieved_fragments'], artifacts)
        plan = [name for name in ledger.evict_plan() if name not in removed]
        if not plan:
            return BudgetedPrompt(messages, ledger, removed, counts,
                                  arguments['retrieved_fragments'], artifacts)
        # Reassemble after EACH change. A source-note block has whole-note
        # alternatives, so a small overflow need not discard every note.
        name = plan[0]
        if name == 'conversation_summary' and summary_options:
            current = arguments.get('conversation_summary_text')
            try:
                index = summary_options.index(current)
            except ValueError:
                index = len(summary_options) - 1
            if index + 1 < len(summary_options):
                replacement = summary_options[index + 1]
                arguments['conversation_summary_text'] = replacement
                evictions.append({'block': 'conversation_summary_part',
                                  'tokens': max(0, count_tokens(current or '')
                                                - count_tokens(replacement))})
                continue
        if name == 'slots': arguments['memory_slots'] = []
        elif name == 'session_start': arguments['session_start_text'] = None
        elif name == 'bookmarks': arguments['bookmarked_texts'] = None
        elif name == 'conversation_summary': arguments['conversation_summary_text'] = None
        elif name == 'recent_turns': arguments['max_recent_tokens'] = 0
        elif name == 'evidence': evidence_enabled = False
        else: raise ValueError(f'Unknown evictable prompt block: {name}')
        removed.append(name)
        evictions.append({'block': name, 'tokens': costs.get(name, 0)})
