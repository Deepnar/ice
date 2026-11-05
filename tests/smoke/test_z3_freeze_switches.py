"""The Z3 freeze (maintainer, 2026-11-04/05): R7's choices are ICE's defaults, and the parts D9 switched
off are off everywhere — no job queued or run, nothing written, nothing in a prompt. Code kept; deleted
only after FINAL."""
import pytest

from src.api.config import Settings


def test_frozen_defaults():
    s = Settings()
    assert s.turn_summaries_in_reads is False
    assert (s.memory_source_rescue_mode, s.memory_source_rescue_min_score) == ("reranker", 0.0)
    assert not (s.conversation_notes_enabled or s.batch_summaries_enabled or s.procedural_memory_enabled)
    assert s.retrieval_rerank_candidates == 256


def test_switched_off_jobs_are_never_queued_and_refuse_to_run(monkeypatch):
    from src.api.config import settings
    from src.workers import batch_summarizer, conversation_summary, runtime
    assert runtime.job_switched_off("batch_summarize") and runtime.job_switched_off("conversation_summary")
    assert runtime.job_switched_off("decay_procedural")                     # no pass over an off part's table
    from src.workers import procedural_decay
    assert procedural_decay.decay_procedural_patterns() == {"skipped": "procedural_memory_enabled is off"}
    assert not runtime.job_switched_off("reflection") and not runtime.job_switched_off("post_flight")
    # the job functions refuse before touching the database (the replay harness calls them directly)
    assert batch_summarizer.batch_summarize() == {"skipped": "batch_summaries_enabled is off"}
    assert conversation_summary.run_conversation_summaries(None)["skipped"] == "conversation_notes_enabled is off"
    monkeypatch.setattr(settings, "batch_summaries_enabled", True)
    monkeypatch.setattr(settings, "conversation_notes_enabled", True)
    assert not runtime.job_switched_off("batch_summarize") and not runtime.job_switched_off("conversation_summary")


def test_the_runtime_drops_a_switched_off_job_before_queueing_it():
    from src.workers import runtime
    rt = runtime.MaintenanceRuntime.__new__(runtime.MaintenanceRuntime)   # no threads, no database
    rt._stopping, rt._jobs, rt._queued_keys = False, {"batch_summarize": None, "conversation_summary": None}, set()
    rt.enqueue("batch_summarize")
    rt.enqueue("conversation_summary")
    assert rt._queued_keys == set()

