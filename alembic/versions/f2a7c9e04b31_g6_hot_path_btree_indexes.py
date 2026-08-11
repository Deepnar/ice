"""G6: the hot-path btree indexes the orphan SQL script never covered

Revision ID: f2a7c9e04b31
Revises: d5c81a37e9b2
Create Date: 2026-08-11

**What G6 still owed, and what it did not.** The entry says
`scripts/database/create_indexes.sql` "isn't applied by Alembic". That was true
when written and is now only half true: **b6e2f9a41c73 (G23/C17) already creates
every `idx_<table>_embedding` HNSW index in a loop** — it had to, because
widening the vectors to 1024 dropped and rebuilt them — so the vector half is
migration-covered on a fresh clone and the orphan script is redundant. Verified
against the live DB 2026-08-11: eight HNSW indexes present, all of them
reachable from `alembic upgrade head` alone.

What was genuinely missing is the **btree** half, which no migration and no
script ever created:

* `episodic_memory.batch_id` — the entry names this one, and it is the worst
  of them: **21 query sites in `orchestrator.py` alone** filter or join on a
  batch id, including every scoped retrieval path (A5's batch-id sets, C6's
  conversation scoping, the codex leg's `source_batch` gates). Unindexed, each
  is a sequential scan of the whole turn table.
* `episodic_memory.conversation_id` — `_conv_scope_filter` runs on effectively
  every retrieval.
* `episodic_memory.decay_score` and `lossless_flag` — the decay sweep and the
  batch summariser filter on these over the full table.

**Why now rather than "when tuning gets slow" (the deferral this retires).**
Z1 is about to populate the store and then sweep settings against it. An index
added *after* that would mean every tuning run carried a sequential scan, so the
timings would describe the missing index rather than the setting under test —
and the sweep is the one thing that cannot be re-run cheaply.

`IF NOT EXISTS` throughout: this machine already carries the vector indexes from
a manual run of the orphan script, and a migration that fails on an existing
index would make a working developer database unupgradeable.
"""
from alembic import op

revision = "f2a7c9e04b31"
down_revision = "d5c81a37e9b2"
branch_labels = None
depends_on = None


# (index name, table, column) — btree only; the vector indexes belong to
# b6e2f9a41c73 and are deliberately not duplicated here.
_INDEXES = [
    ("ix_episodic_memory_batch_id", "episodic_memory", "batch_id"),
    ("ix_episodic_memory_conversation_id", "episodic_memory", "conversation_id"),
    ("ix_episodic_memory_decay_score", "episodic_memory", "decay_score"),
    ("ix_episodic_memory_lossless_flag", "episodic_memory", "lossless_flag"),
]


def upgrade() -> None:
    for name, table, col in _INDEXES:
        op.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({col})")


def downgrade() -> None:
    for name, _table, _col in _INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {name}")
