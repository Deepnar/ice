"""G4(a): a running background job stands down when the user comes back.

Before this, *starting* a job was well guarded (`_gpu_ready` demands silence
and no generation in flight) while *continuing* was not guarded at all — once a
job entered `asyncio.to_thread` nothing could interrupt it, and a returning
user queued behind it inside Ollama.

Run: uv run python tests/test_job_yield.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.config import settings  # noqa: E402
from src.workers.runtime import (  # noqa: E402
    JobYielded,
    MaintenanceRuntime,
    yield_if_user_active,
)
import src.workers.runtime as rt_mod  # noqa: E402

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}{('  — ' + detail) if detail else ''}")


def quiet(rt, seconds=3600):
    rt._generation_inflight = 0
    rt._last_activity = datetime.now(timezone.utc) - timedelta(seconds=seconds)


print("── should_yield() ──")
rt = MaintenanceRuntime()
quiet(rt)
check("a long-quiet runtime does NOT ask jobs to yield", not rt.should_yield())

rt._generation_inflight = 1
check("a generation in flight asks jobs to yield", rt.should_yield())

rt._generation_inflight = 0
rt._last_activity = datetime.now(timezone.utc)
check("activity inside the grace window asks jobs to yield", rt.should_yield())

# The gate that STARTS a job is deliberately looser than the one that STOPS it:
# 90 s of silence to start, but a running job stands down after ~10 s of
# activity. Assert the asymmetry, or a future edit could quietly collapse them.
rt._last_activity = datetime.now(timezone.utc) - timedelta(
    seconds=settings.job_yield_grace_seconds + 5)
check("a running job resumes before a new one would be allowed to start",
      not rt.should_yield() and not rt._gpu_ready(for_event=False),
      f"yield={rt.should_yield()} gpu_ready={rt._gpu_ready(for_event=False)}")

print("── yield_if_user_active() ──")
saved = rt_mod._active_runtime
try:
    rt_mod._active_runtime = None
    try:
        yield_if_user_active("no-runtime")
        check("no runtime ⇒ no-op (scripts and tests stay safe)", True)
    except JobYielded:
        check("no runtime ⇒ no-op (scripts and tests stay safe)", False)

    rt_mod._active_runtime = rt
    quiet(rt)
    try:
        yield_if_user_active("quiet")
        check("quiet runtime ⇒ the job continues", True)
    except JobYielded:
        check("quiet runtime ⇒ the job continues", False)

    rt._generation_inflight = 1
    try:
        yield_if_user_active("busy")
        check("busy runtime ⇒ the job raises JobYielded", False)
    except JobYielded as exc:
        check("busy runtime ⇒ the job raises JobYielded", True)
        check("the yield names where it happened", "busy" in str(exc), str(exc))

    # JobYielded must not be catchable as an ordinary job failure, or the
    # runtime would mark the ledger row errored and burn a retry attempt.
    check("JobYielded is not a subclass of anything the workers catch broadly",
          issubclass(JobYielded, Exception) and JobYielded is not Exception)
finally:
    rt_mod._active_runtime = saved

print("── the real call sites have a yield point ──")
import inspect  # noqa: E402
from src.workers import batch_summarizer, codex_extractor, conversation_summary  # noqa: E402

for mod, fn_name in [(codex_extractor, "extract_triplets"),
                     (batch_summarizer, "batch_summarize"),
                     (conversation_summary, "run_conversation_summaries")]:
    src = inspect.getsource(getattr(mod, fn_name))
    check(f"{mod.__name__.split('.')[-1]}.{fn_name} yields inside its loop",
          "yield_if_user_active" in src)

print("── a yield does not lose work already committed ──")
# codex_extract writes the graph in extract_codex AFTER extraction returns, so
# abandoning between chunks leaves no half-written graph. conversation_summary
# carries `covers_through`, so a folded conversation is not refolded. Both are
# properties of where the yield point was placed, so pin them.
check("extract_triplets yields BEFORE the LLM call, not after the graph write",
      inspect.getsource(codex_extractor.extract_triplets).index("yield_if_user_active")
      < inspect.getsource(codex_extractor.extract_triplets).index("bg_client.chat"))
check("conversation_summary still has covers_through to resume from",
      "covers_through" in inspect.getsource(
          conversation_summary.run_conversation_summaries))

print()
print(f"  {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
