"""CODEX_V4 D7: every path that reranks applies the fact strength bonus after it.

The wide-net fallback reranked without it — inert while B = 0, a silent
difference between the two retrieval paths as soon as B is set.
"""
import inspect
import re

from src.retrieval.orchestrator import HybridRetrievalOrchestrator


def test_every_rerank_is_followed_by_the_strength_bonus():
    src = inspect.getsource(HybridRetrievalOrchestrator)
    reranks = [m.end() for m in re.finditer(r"fused, reranked = rerank\(", src)]
    assert len(reranks) == 2                       # retrieve() and _wide_net_fallback()
    for end in reranks:
        following = src[end:end + 300]
        assert "self._fact_strength_bonus(fused)" in following
