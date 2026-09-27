"""Owned model release and real runtime transitions without inference or data writes."""
import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from src.api.config import settings
from src.retrieval import ner_utils as ner
from src.workers import bg_client_factory as bg
from src.workers.runtime import JOBS, JobSpec, MaintenanceRuntime


@pytest.fixture
def ownership(monkeypatch):
    monkeypatch.setattr(bg, "_owned_models", set())
    monkeypatch.setattr(bg, "_model_locks", {})
    monkeypatch.setattr(settings, "bg_release_after_drain", True)
    monkeypatch.setattr(settings, "background_model_mode", "shared")
    calls, released = [], []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="complete", model_extra={}), finish_reason="stop")])
    def client(**_kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(bg, "OpenAI", client)
    def release(model):
        released.append(model)
        return True
    monkeypatch.setattr(bg, "release_bg_model", release)
    return calls, released


def idle_runtime():
    runtime = MaintenanceRuntime(jobs={}, intervals={})
    runtime._started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    return runtime


@pytest.mark.parametrize("disable_reasoning", [True, False])
def test_actual_factory_calls_own_both_models_and_idle_pump_releases_once(
        ownership, monkeypatch, disable_reasoning):
    monkeypatch.setattr(settings, "bg_disable_reasoning", disable_reasoning)
    calls, released = ownership
    client = bg.get_bg_client()
    client.chat.completions.create(model="general-local", messages=[], max_tokens=20)
    client.chat.completions.create(model="extractor-local", messages=[], max_tokens=20)
    client.chat.completions.create(model="general-local", messages=[], max_tokens=20)
    assert bg._owned_models == {"general-local", "extractor-local"}
    assert ("extra_body" in calls[0]) is disable_reasoning
    cache_releases = []
    monkeypatch.setattr(ner, "release_background_ner", lambda: cache_releases.append(True))
    runtime = idle_runtime()
    asyncio.run(runtime._pump())
    asyncio.run(runtime._pump())
    assert released == ["extractor-local", "general-local"]
    assert cache_releases == [True] and runtime._bg_released
    assert not bg._owned_models


def test_dedicated_calls_never_become_ollama_unload_targets(ownership, monkeypatch):
    monkeypatch.setattr(settings, "background_model_mode", "dedicated")
    bg.get_bg_client().chat.completions.create(model="remote-only", messages=[])
    assert bg.release_owned_models()
    assert not ownership[1] and not bg._owned_models


def test_failed_call_still_owns_possible_loaded_model(ownership, monkeypatch):
    def fail(**_kwargs): raise RuntimeError("request failed after model load")
    inner = SimpleNamespace(create=fail)
    with pytest.raises(RuntimeError):
        bg._OwnedCompletions(inner).create(model="partially-loaded")
    assert bg._owned_models == {"partially-loaded"}
    assert bg.release_owned_models() and ownership[1] == ["partially-loaded"]


def test_active_native_call_defers_unload_then_releases(ownership):
    with bg.local_model_call("native-proof"):
        assert not bg.release_owned_models()
        assert not ownership[1]
    assert bg.release_owned_models() and ownership[1] == ["native-proof"]


def test_new_native_call_after_idle_drain_is_still_released(ownership, monkeypatch):
    monkeypatch.setattr(ner, "release_background_ner", lambda: False)
    runtime = idle_runtime()
    asyncio.run(runtime._pump())
    assert runtime._bg_released
    with bg.local_model_call("native-proof"): pass
    asyncio.run(runtime._pump())
    assert ownership[1] == ["native-proof"] and not bg._owned_models


def test_failed_release_retries_only_remaining_model(ownership, monkeypatch):
    for name in ("general-local", "extractor-local"):
        with bg.local_model_call(name): pass
    attempts = []
    def release(model):
        attempts.append(model)
        return model != "extractor-local" or attempts.count(model) > 1
    monkeypatch.setattr(bg, "release_bg_model", release)
    assert not bg.release_owned_models()
    assert bg._owned_models == {"extractor-local"}
    assert bg.release_owned_models()
    assert attempts == ["extractor-local", "general-local", "extractor-local"]


def test_unload_acknowledgement_requires_observed_disappearance(monkeypatch):
    polls = []
    monkeypatch.setattr(settings, "bg_release_after_drain", True)
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: dict(done_reason="unload")))
    def get(*_args, **_kwargs):
        polls.append(True)
        models = [dict(name="owned-local")] if len(polls) == 1 else []
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(models=models))
    monkeypatch.setattr(httpx, "get", get)
    assert bg.release_bg_model("owned-local") and len(polls) == 2


def test_unload_deadline_is_failure_when_model_remains(monkeypatch):
    monkeypatch.setattr(settings, "bg_release_after_drain", True)
    monkeypatch.setattr(settings, "bg_release_timeout_seconds", .01)
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: dict(done_reason="unload")))
    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: dict(models=[dict(name="owned-local")])))
    warnings = []
    monkeypatch.setattr(bg.logger, "warning", lambda event, **fields: warnings.append((event, fields)))
    assert not bg.release_bg_model("owned-local")
    assert warnings == [("bg_model_release_unconfirmed", {
        "model": "owned-local", "reason": "still_resident_at_deadline"})]


@pytest.mark.parametrize("busy", ["activity", "generation", "queue", "task", "disabled"])
def test_idle_pump_does_not_release_during_work(ownership, monkeypatch, busy):
    with bg.local_model_call("general-local"): pass
    runtime = idle_runtime()
    if busy == "activity": runtime.note_user_activity()
    elif busy == "generation": runtime.generation_started()
    elif busy == "queue": runtime._queue.append(("unused", {}, 0))
    elif busy == "task": runtime._tasks.add(object())
    else: monkeypatch.setattr(settings, "bg_release_after_drain", False)
    # Pending queue is intentionally left pending; no fake job is dispatched.
    monkeypatch.setattr(runtime, "_dispatch_events", lambda: None)
    asyncio.run(runtime._pump())
    assert not ownership[1] and bg._owned_models == {"general-local"}


def test_nuner_cache_is_released_and_next_call_can_reload(monkeypatch):
    model = SimpleNamespace()
    monkeypatch.setattr(ner, "_bg_ner", model)
    monkeypatch.setattr(ner, "_bg_ner_attempted", True)
    empty_calls = []
    monkeypatch.setattr(ner.torch.cuda, "empty_cache", lambda: empty_calls.append(True))
    assert ner.release_background_ner()
    assert ner._bg_ner is None and not ner._bg_ner_attempted
    assert empty_calls == [True]
    assert not ner.release_background_ner()


def test_nuner_release_waits_for_active_inference(monkeypatch):
    monkeypatch.setattr(ner, "_bg_ner", object())
    monkeypatch.setattr(ner, "_bg_ner_attempted", True)
    monkeypatch.setattr(ner.torch.cuda, "empty_cache", lambda: None)
    finished = threading.Event()
    with ner._bg_ner_lock:
        thread = threading.Thread(target=lambda: (ner.release_background_ner(), finished.set()))
        thread.start()
        assert not finished.wait(.05) and ner._bg_ner is not None
    thread.join(timeout=2)
    assert finished.is_set() and ner._bg_ner is None


def test_actual_nuner_extraction_holds_release_guard(monkeypatch):
    started, finish, released = threading.Event(), threading.Event(), threading.Event()
    def predict(*_args, **_kwargs):
        started.set()
        assert finish.wait(2)
        return []
    monkeypatch.setattr(ner, "_bg_ner", SimpleNamespace(predict_entities=predict))
    monkeypatch.setattr(ner, "_bg_ner_attempted", True)
    monkeypatch.setattr(ner, "_background_model_windows", lambda piece, _model: [piece])
    monkeypatch.setattr(ner.torch.cuda, "empty_cache", lambda: None)
    inference = threading.Thread(target=lambda: ner._extract_background("a source", labels=["person"]))
    inference.start()
    assert started.wait(2)
    cleanup = threading.Thread(target=lambda: (ner.release_background_ner(), released.set()))
    cleanup.start()
    assert not released.wait(.05)
    finish.set()
    inference.join(timeout=2)
    cleanup.join(timeout=2)
    assert released.is_set() and ner._bg_ner is None


@pytest.mark.parametrize("job", ["cluster_assignment", "cluster_merge", "chunk_pending_documents"])
def test_model_using_memory_jobs_share_gpu_lane(job):
    assert JOBS[job].lane == "gpu"


def test_waiting_gpu_job_rechecks_user_activity_without_run_or_retry(monkeypatch):
    async def run():
        runtime = MaintenanceRuntime(jobs={"later": JobSpec("unused:unused", "gpu")}, intervals={})
        runtime._started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        called = []
        monkeypatch.setattr(runtime, "_ledger_claim", lambda *_args: called.append("claim"))
        monkeypatch.setattr(runtime, "_call", lambda *_args: called.append("call"))
        await runtime._gpu_lane.acquire()
        runtime._spawn("later", {"source": "preserved"}, 2, "event")
        await asyncio.sleep(0)
        runtime.note_user_activity()
        runtime._gpu_lane.release()
        await asyncio.gather(*runtime._tasks)
        assert not called
        assert list(runtime._queue) == [("later", {"source": "preserved"}, 2)]
        assert not runtime._running_keys and runtime._queued_keys
        assert not runtime._retries
    asyncio.run(run())
