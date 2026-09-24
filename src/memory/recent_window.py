"""Shared base recent-window curve for B2, summary creation and retrieval.

Retrieval adds density and classification adjustments after this base curve.
The other readers deliberately use only the base: summary creation has no
routed model or request classification, and B2 needs a stable prior before it
decides whether retrieval runs at all.
"""

from src.api.config import settings


def base_recent_fraction(turn_count: int) -> float:
    fraction = settings.context_recent_fraction_default
    for edge, value in settings.context_recent_fraction_ladder:
        if turn_count < edge:
            return value
    return fraction


def estimate_recent_window_tokens(turn_count: int, total_budget: float = None) -> float:
    """Base recent-window allowance, before request-specific adjustments."""
    total = total_budget if total_budget else settings.context_total_budget_fallback
    fraction = max(settings.context_recent_fraction_min,
                   min(settings.context_recent_fraction_max,
                       base_recent_fraction(turn_count)))
    return fraction * (total - settings.context_overhead_reserve)
