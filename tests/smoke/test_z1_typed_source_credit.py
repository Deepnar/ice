"""A derived memory's batch provenance must credit its episodic source row."""

from scripts.z1.source_credit import (covered_gold_turns, fragment_source_ids,
                                      gold_source_ids)
from scripts.z1.score_typed import first_specific_gold_hit
from src.retrieval.orchestrator import ContextFragment


def _fragment(kind, *, row=None, batches=()):
    return ContextFragment(
        text="source-backed evidence", source_type=kind, score=1.0,
        token_count=4, source_batch_id=row, origin_batch_ids=batches)


def test_derived_legs_credit_only_the_matching_gold_turn():
    batch_by_row = {"row-a": "batch-a", "row-b": "batch-b"}
    for leg in ("codex", "procedural", "batch_summary"):
        evidence = _fragment(leg, batches=("batch-a",))
        assert fragment_source_ids(evidence) & gold_source_ids("row-a", batch_by_row)
        assert not fragment_source_ids(evidence) & gold_source_ids("row-b", batch_by_row)
    original = _fragment("episodic", row="row-a")
    assert fragment_source_ids(original) & gold_source_ids("row-a", batch_by_row)
    assert not fragment_source_ids(original) & gold_source_ids("row-b", batch_by_row)
    assert covered_gold_turns(["row-a", "row-b"], [original,
                             _fragment("codex", batches=("batch-a",))], batch_by_row) == 1


def test_gold_hit_keeps_final_prompt_rank_and_names_earning_leg():
    fragments = [_fragment("batch_summary", batches=("batch-a",)),
                 _fragment("episodic", row="row-b"),
                 _fragment("timeline", batches=("batch-a",))]
    assert first_specific_gold_hit(fragments, {"batch-a"}) == (3, "timeline")
    assert first_specific_gold_hit(fragments, {"batch-missing"}) == (None, None)
