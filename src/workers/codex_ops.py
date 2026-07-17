"""Codex surgery callables (the D1/D2 seam; future graph surgery lives here —
codex_extractor is big enough).

`merge_entities` is dispatched by review-queue approval (E0's `review.py`)
and, from D1 on, by the maintenance agent's Tier-0/Tier-2 paths. The real
implementation lands with D1 (spec D1_D2 D5: re-point edges, union aliases,
keep the longer description journaled, move events, regenerate the payload,
expire-with-alias — never hard-delete). Until then this stub is LOUD: no
`entity_merge` items exist before the agent writes them, and approving one
against a stub must fail visibly, not silently flip a status.
"""


def merge_entities(db, keep_id, absorb_id, agent_run_id=None) -> dict:
    raise NotImplementedError(
        "merge_entities is a D1 stub (specs/D1_D2_maintenance_agent.md D5) — "
        "the maintenance agent session builds the real merge"
    )
