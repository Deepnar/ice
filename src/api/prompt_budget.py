"""Assemble, account, evict optional blocks, then verify the actual result."""
from dataclasses import dataclass

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


def assemble_budgeted_prompt(*, serving_window, generation_reserve,
                             safety_margin=1.0, **kwargs):
    arguments = dict(kwargs)
    summary_options = arguments.pop('conversation_summary_options', None) or []
    summary_sources = arguments.pop('conversation_summary_source_ids', None) or {}
    source_fragments = list(arguments.get('retrieved_fragments') or [])
    evidence_enabled = True
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
        costs, counts = {}, {}
        messages = assemble_prompt(**arguments, block_tokens=costs, block_counts=counts)
        ledger = ContextLedger(serving_window=serving_window,
            generation_reserve=generation_reserve, safety_margin=safety_margin)
        for name, tokens in costs.items():
            ledger.add(name, tokens)
        ledger.evictions = list(evictions)
        if ledger.fits():
            return BudgetedPrompt(messages, ledger, removed, counts,
                                  arguments['retrieved_fragments'])
        plan = [name for name in ledger.evict_plan() if name not in removed]
        if not plan:
            return BudgetedPrompt(messages, ledger, removed, counts,
                                  arguments['retrieved_fragments'])
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
