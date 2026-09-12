"""Small synthetic v3 reranker qualification, not an answer-quality benchmark.

Run manually: uv run python tests/test_reranker_quality.py
Uses the pinned cached model and real production scorer; no stored user data.
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.config import settings
from src.memory.tokens import count as count_tokens
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator
from src.retrieval.reranker import score_pairs

CASES = [
    (
        [
            "Which port does Atlas use?",
            "ok so what port does atlas use",
            "atlas port pls",
        ],
        [
            "Atlas listens on port 8391.",
            "Atlas uses Python and Redis.",
            "Boreal listens on port 8000.",
            "The user enjoys gardening.",
        ],
    ),
    (
        [
            "What did Mira stop using?",
            "like what did mira stop using",
            "what tool is mira no longer using?",
        ],
        [
            "Mira no longer uses Slack; she moved to Matrix.",
            "Mira likes dark themes.",
            "Sam stopped using Teams.",
            "Slack supports channels.",
        ],
    ),
    (
        [
            "When did Atlas switch databases?",
            "atlas changed databases when",
            "what date was the database change for atlas?",
        ],
        [
            "Atlas switched from SQLite to PostgreSQL on June 4, 2026.",
            "Atlas uses Python.",
            "Boreal switched databases on May 2, 2026.",
            "PostgreSQL is a relational database.",
        ],
    ),
    (
        [
            "How should I prepare before deploying Atlas?",
            "atlas deploy prep steps please",
            "like before i deploy atlas what should i do",
        ],
        [
            "Before deploying Atlas, run the migration checks and save a database backup.",
            "Atlas is deployed in Europe.",
            "Before deploying Boreal, update its certificates.",
            "The user prefers short replies.",
        ],
    ),
    (
        [
            "Which module defines Atlas authentication?",
            "atlas auth defined where",
            "where do we implement authentication for atlas?",
        ],
        [
            "Atlas authentication is implemented in src/security/auth.py.",
            "Atlas logs requests in src/logging.py.",
            "Boreal uses src/auth.py.",
            "Authentication verifies identity.",
        ],
    ),
]


def pipeline_controls():
    """Actual model + retrieve + budget; scoped leg output is synthetic."""
    query = CASES[0][0][0]
    answer, distractor = CASES[0][1][:2]
    controls = []
    for wide in (False, True):
        outcomes = []
        for enabled in (False, True):
            db = SimpleNamespace(
                execute=lambda *a, **kw: SimpleNamespace(fetchall=lambda: [])
            )
            o = HybridRetrievalOrchestrator(db, None)
            o.max_retrieval_tokens = count_tokens(answer)
            o._codex_graph = lambda *a, **kw: [
                ContextFragment(distractor, "codex", 2, count_tokens(distractor)),
                ContextFragment(answer, "codex", 1, count_tokens(answer)),
            ]
            for name in (
                "_relevant_cluster_ids",
                "_bm25_episodic",
                "_vector_episodic",
                "_procedural_lookup",
                "_batch_summary_lookup",
                "_cold_lookup",
            ):
                setattr(o, name, lambda *a, **kw: [])
            o._apply_bonuses = lambda f, *a: f
            c = SimpleNamespace(
                prompt=query,
                context_reliance="Long_Term_Memory",
                max_confidence=0.0 if wide else 1.0,
                intent_tags=[],
                topic_tags=[],
            )
            with (
                patch.object(settings, "retrieval_rerank_enabled", enabled),
                patch.object(
                    settings, "retrieval_wide_net_budget_floor", count_tokens(answer)
                ),
                patch.object(settings, "retrieval_rerank_min_score", None),
            ):
                outcomes.append([f.text for f in o.retrieve(c, None, [1.0])])
        assert answer not in outcomes[0] and outcomes[1] == [answer], outcomes
        controls.append(
            {"path": "wide" if wide else "normal", "selection_changed_to_answer": True}
        )
    return controls


def main():
    pairs = [
        (query, document)
        for queries, docs in CASES
        for query in queries
        for document in docs
    ]
    start = time.perf_counter()
    scores = score_pairs(pairs)
    elapsed = time.perf_counter() - start
    groups = [scores[i : i + 4] for i in range(0, len(scores), 4)]
    report = {
        "version": "v3",
        "kind": "synthetic model qualification, not end-to-end answers",
        "queries": len(groups),
        "pairs": len(pairs),
        "gold_rank_first": sum(g.index(max(g)) == 0 for g in groups),
        "gold_pass_zero_floor": sum(g[0] >= 0 for g in groups),
        "distractors_below_zero": sum(s < 0 for g in groups for s in g[1:]),
        "scores": groups,
        "elapsed_seconds_including_load": round(elapsed, 3),
    }
    report["production_selection_controls"] = pipeline_controls()
    import torch

    report["peak_cuda_allocated_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 3
    )
    report["ranking_qualified"] = report["gold_rank_first"] == len(groups)
    report["zero_floor_qualified"] = (
        report["gold_pass_zero_floor"] == len(groups)
        and report["distractors_below_zero"] == len(groups) * 3
    )
    print(json.dumps(report, indent=2))
    assert report["gold_rank_first"] == len(groups)
    # A failed rejection qualification is reported, not silently promoted.
    if settings.retrieval_rerank_min_score == 0.0:
        assert report["zero_floor_qualified"]


if __name__ == "__main__":
    main()
