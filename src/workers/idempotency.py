"""G29 — the one place a background job asks "have I already done this?".

Two jobs marked their work in the shared ``idempotency_keys`` table and each
wrote the ritual out by hand. They had already diverged on the part that
matters: ``procedural_extractor`` hashed ``f"procedural:{batch_id}"`` while
``post_flight`` hashed the **bare** ``batch_id``.

Why that is a defect and not a style difference: the table is shared, so the key
is the only thing separating one job's "done" from another's. An un-namespaced
key is the first-come-first-served slot for a given batch — the next job written
against a bare batch id would silently inherit post_flight's marker and skip
work it had never performed. Nothing enforced the convention, and the one job
that followed it did so by accident of being written second.

Note the asymmetry with ``EpisodicMemory.idempotency_key``, which is a different
mechanism wearing the same name: that column dedupes *rows* on the write path
(``main.py``, ``bookmarks.py``, ``importer._turn_idempotency_key``) and is keyed
on turn content. This module is only about the job-marker table.
"""

import hashlib


def job_key(job: str, batch_id) -> str:
    """The idempotency key for *job* having processed *batch_id*.

    `job` is mandatory and unprefixed keys are impossible to express — which is
    the entire point, since the bug was a key that omitted it.
    """
    if not job:
        raise ValueError("job namespace is required; an un-namespaced key "
                         "collides with every other job on the same batch")
    return hashlib.sha256(f"{job}:{batch_id}".encode()).hexdigest()
