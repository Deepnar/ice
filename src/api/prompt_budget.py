"""Assemble, account, evict optional blocks, then verify the actual result."""
from dataclasses import dataclass

import structlog

from src.api.context_ledger import ContextLedger
from src.api.prompt_assembler import assemble_prompt

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
        for name in plan:
            if name == 'slots': arguments['memory_slots'] = []
            elif name == 'session_start': arguments['session_start_text'] = None
            elif name == 'bookmarks': arguments['bookmarked_texts'] = None
            elif name == 'conversation_summary': arguments['conversation_summary_text'] = None
            elif name == 'recent_turns': arguments['max_recent_tokens'] = 0
            elif name == 'evidence': arguments['retrieved_fragments'] = []
            else: raise ValueError(f'Unknown evictable prompt block: {name}')
        removed.extend(plan)
        evictions = list(ledger.evictions)
