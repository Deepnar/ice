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


def assemble_budgeted_prompt(*, serving_window, generation_reserve,
                             safety_margin=1.0, **kwargs):
    arguments = dict(kwargs)
    summary_options = arguments.pop('conversation_summary_options', None) or []
    removed = []
    evictions = []
    if not serving_window:
        logger.warning("context_window_unknown", reason="Provider capacity unavailable; prompt fit is unmeasured")
    while True:
        costs, counts = {}, {}
        messages = assemble_prompt(**arguments, block_tokens=costs, block_counts=counts)
        ledger = ContextLedger(serving_window=serving_window,
            generation_reserve=generation_reserve, safety_margin=safety_margin)
        for name, tokens in costs.items():
            ledger.add(name, tokens)
        ledger.evictions = list(evictions)
        if ledger.fits():
            return BudgetedPrompt(messages, ledger, removed, counts)
        plan = [name for name in ledger.evict_plan() if name not in removed]
        if not plan:
            return BudgetedPrompt(messages, ledger, removed, counts)
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
        elif name == 'evidence': arguments['retrieved_fragments'] = []
        else: raise ValueError(f'Unknown evictable prompt block: {name}')
        removed.append(name)
        evictions.append({'block': name, 'tokens': costs.get(name, 0)})
