"""G50: backfill codex_entities.properties.merge_key + partial index

The write-time tier in `get_or_create_entity` matches on
`properties->>'merge_key'`. Existing rows have no such key, so without this
backfill the tier can never hit an entity created before it shipped — the store
would keep minting duplicates against 8,280 invisible rows.

⚠ The backfill CANNOT be pure SQL. `merge_key()` splits digit/letter runs and
collapses separator punctuation in Python; an approximation in SQL would write
a key that disagrees with the one the write path computes, which is worse than
no key at all — the tier would match the wrong rows. So this loops in batches
through the Python function itself.

Revision ID: a1c4e7b90d22
Revises: 505f12031434
"""
import sqlalchemy as sa
from alembic import op

revision = "a1c4e7b90d22"
down_revision = "505f12031434"
branch_labels = None
depends_on = None

BATCH = 1000


def upgrade() -> None:
    from src.workers.maintenance_agent import merge_key

    bind = op.get_bind()
    total = updated = 0
    while True:
        rows = bind.execute(sa.text("""
            SELECT id, canonical_name FROM codex_entities
            WHERE properties->>'merge_key' IS NULL
            ORDER BY id LIMIT :n
        """), {"n": BATCH}).fetchall()
        if not rows:
            break
        total += len(rows)
        for rid, name in rows:
            bind.execute(sa.text("""
                UPDATE codex_entities
                   SET properties = jsonb_set(
                         coalesce(properties, '{}'::jsonb),
                         '{merge_key}', to_jsonb(CAST(:k AS text)), true)
                 WHERE id = :id
            """), {"k": merge_key(name or ""), "id": rid})
            updated += 1
        if len(rows) < BATCH:
            break
    print(f"  g50: stamped merge_key on {updated} entities (scanned {total})")

    # Partial index: only live rows are ever looked up, and `merged_into` marks
    # the absorbed ones. Matches the write-time tier's WHERE clause exactly.
    op.create_index(
        "ix_codex_entities_merge_key",
        "codex_entities",
        [sa.text("(properties->>'merge_key')")],
        postgresql_where=sa.text("(properties->>'merged_into') IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_codex_entities_merge_key", table_name="codex_entities")
    # Leave the stamped keys in place: they are derived data, harmless to keep,
    # and re-deriving them costs a full pass. Dropping them would also make a
    # re-upgrade re-scan every row for nothing.
