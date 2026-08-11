"""G29: job markers in the shared table must be namespaced by job.

`post_flight` and `procedural_extractor` both record "I have processed this
batch" in the same `idempotency_keys` table, and each built its key by hand.
They had diverged on the only part that separates them: `procedural_extractor`
hashed `f"procedural:{batch_id}"`, `post_flight` hashed the **bare** batch id.

An un-namespaced key is a first-come-first-served claim on a batch. The next job
written against a bare batch id would inherit post_flight's marker and skip work
it never did — silently, because a skipped idempotent job looks exactly like a
completed one.

Run:  uv run pytest tests/smoke/test_idempotency_keys.py -q
"""

import hashlib
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.workers.idempotency import job_key  # noqa: E402


def test_different_jobs_never_collide_on_one_batch():
    assert job_key("post_flight", "b1") != job_key("procedural", "b1")


def test_the_same_job_and_batch_is_stable():
    """Idempotency depends on this being a pure function of its inputs."""
    assert job_key("post_flight", "b1") == job_key("post_flight", "b1")


def test_a_bare_batch_id_can_no_longer_be_expressed():
    """The bug was a key that omitted the namespace, so omitting it is an error
    rather than a default."""
    with pytest.raises(ValueError):
        job_key("", "b1")


def test_the_procedural_key_is_unchanged():
    """The migration side: procedural markers already in the table must keep
    matching, or every previously-processed batch is re-extracted."""
    assert job_key("procedural", "b1") == hashlib.sha256(
        b"procedural:b1").hexdigest()


def test_no_worker_still_hashes_a_batch_id_by_hand():
    """Structural — the copies are gone, and a new one would reintroduce the
    collision this module exists to prevent."""
    import src.workers.post_flight as pf
    import src.workers.procedural_extractor as pe

    offenders = [m.__name__ for m in (pf, pe)
                 if "hashlib.sha256" in inspect.getsource(m)]
    assert not offenders, f"hand-rolled idempotency key in: {offenders}"
