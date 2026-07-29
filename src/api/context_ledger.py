"""C16 — one account of what is in the prompt, in one unit.

Before this, five things independently decided how much context there was and
none of them knew about the others: `derive_total_budget` (model-aware), the
recent-fraction ladder, `growth_cap` (turn-count), a hardcoded 10-turn limit,
and the model server's own `num_ctx` — which ICE never set. A sixth, a trimming
loop hardcoded to 4096, was broken in four ways and did nothing at all.

The effect was that the effective budget was decided by whichever hardcoded
ladder happened to be smallest, almost never by the model and never by the
content. `OVERHEAD_RESERVE = 1_800` was supposed to cover "system message +
slots + question" and enforced nothing: `VALID_SLOTS` permits 13 slots at 300
tokens each (3,900 on its own), the project session-start block has no cap at
all and embeds a raw git diffstat, and the user's own message was never counted
against anything.

This module is the account. Blocks are added with their REAL token counts, the
total is checked against the window the server actually allocated, and when it
does not fit, blocks are evicted in a stated order.

**Eviction order (user, 2026-07-29): static before evidence.** Boilerplate goes
before the memory that answers the question. With one exception, which the user
confirmed: **E8 constraints are exempt entirely.** They are static in shape but
they are user preferences — recorded do-not-touch rules — and losing one is the
exact failure the feature exists to prevent. They cost at most five lines.
"""

from typing import Optional

import structlog

from src.memory.tokens import count as count_tokens
from src.memory.tokens import with_margin

logger = structlog.get_logger("ice.api.ledger")

# Blocks that may be shed, cheapest-to-lose first. `constraints`, `user_message`
# and `system_prompt_essential` are deliberately absent: the first is a user
# preference, the second IS the question, and the third carries the clause that
# stops the model answering the previous turn.
DEFAULT_EVICT_ORDER = (
    "session_start",
    "slots",
    "bookmarks",
    "conversation_summary",
    "recent_turns",
    "evidence",
)

NEVER_EVICT = ("constraints", "user_message", "system_prompt")


class ContextLedger:
    """Accounting for one request's prompt."""

    def __init__(self, *, serving_window: int, generation_reserve: int,
                 safety_margin: float = 1.0, evict_order=None):
        self.serving_window = int(serving_window or 0)
        self.generation_reserve = int(generation_reserve)
        self.safety_margin = float(safety_margin)
        self.evict_order = tuple(evict_order or DEFAULT_EVICT_ORDER)
        self.blocks: dict[str, int] = {}
        self.evictions: list[dict] = []
        self.pressure: Optional[str] = None

    def add(self, name: str, text_or_tokens) -> int:
        """Record a block. Accepts text (counted) or a token count."""
        tokens = (int(text_or_tokens) if isinstance(text_or_tokens, int)
                  else count_tokens(text_or_tokens or ""))
        self.blocks[name] = self.blocks.get(name, 0) + tokens
        return tokens

    def total(self) -> int:
        return sum(self.blocks.values())

    def available(self) -> int:
        """What the prompt may occupy: the window, less room to answer in it.

        The margin covers the gap between the embedder's tokenizer (what ICE
        counts with) and the generation model's (what actually decides whether
        the prompt fits). Measured at 0.83-0.86 on Llama vocabularies, which is
        why it errs high: under-reading says a prompt fits when it does not.
        """
        if not self.serving_window:
            return 0
        room = self.serving_window - self.generation_reserve
        if room <= 0:
            # A window no larger than the reserve (tinyllama serves 2,048 and
            # the reserve IS 2,048). Returning 0 would make every prompt
            # infinitely over budget and eviction pointless, which reports
            # "extreme pressure" for a perfectly ordinary short question. Split
            # the window instead: half to the prompt, half to the answer.
            room = max(1, self.serving_window // 2)
        return room

    def fits(self) -> bool:
        if not self.serving_window:
            return True                       # unknown window: cannot judge
        return with_margin(self.total(), self.safety_margin) <= self.available()

    def overflow(self) -> int:
        if not self.serving_window:
            return 0
        return max(0, with_margin(self.total(), self.safety_margin)
                   - self.available())

    def evict_plan(self) -> list[str]:
        """Which blocks to shed, in order, to make the prompt fit.

        Returns the block names to drop. The caller does the dropping — the
        ledger does not own the content, only the arithmetic, so a block can be
        *shrunk* instead of dropped where the caller knows how.
        """
        need = self.overflow()
        if need <= 0:
            return []
        plan = []
        for name in self.evict_order:
            if need <= 0:
                break
            if name in NEVER_EVICT:
                continue
            size = self.blocks.get(name, 0)
            if size <= 0:
                continue
            plan.append(name)
            self.evictions.append({"block": name, "tokens": size})
            need -= size
        if need > 0:
            # Everything sheddable is gone and it still does not fit. The
            # question itself is the remainder — say so loudly rather than
            # silently handing the server a prompt it will truncate from the
            # front, which is where the memory lives.
            self.pressure = "extreme"
        return plan

    def snapshot(self) -> dict:
        return {
            "blocks": dict(self.blocks),
            "total": self.total(),
            "serving_window": self.serving_window,
            "generation_reserve": self.generation_reserve,
            "available": self.available(),
            "fits": self.fits(),
            "overflow": self.overflow(),
            "evictions": list(self.evictions),
            "pressure": self.pressure,
        }

    def log(self, log) -> dict:
        snap = self.snapshot()
        if snap["pressure"] == "extreme":
            log.warning("context_pressure_extreme", **{
                k: v for k, v in snap.items() if k != "blocks"})
        elif not snap["fits"]:
            log.warning("context_over_window", **{
                k: v for k, v in snap.items() if k != "blocks"})
        else:
            log.info("context_ledger", **snap)
        return snap


def effective_memory_budget(total_budget: int, user_message: str, *,
                            generation_reserve: int, floor: int) -> int:
    """How much room memory gets, once the user's own message is counted.

    The question was never in the arithmetic: on the turn where someone pastes
    10,000 words, ICE added its full memory allowance on top — the exact case
    where stopping matters most had no accounting at all.

    The floor is the user's call: memory shrinks, but never to nothing.
    """
    question = count_tokens(user_message or "")
    budget = total_budget - question - generation_reserve
    return max(int(floor), min(int(total_budget), int(budget)))
