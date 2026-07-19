"""G23/C17: store_meta + embedding un-truncation 384 -> 1024

Creates store_meta (store-level embedding identity + re-embed progress
stamps) and widens all NINE vector columns to vector(1024) with USING NULL —
data is re-encoded by the re-embed runner (scripts/ice_reembed.py), never
carried across widths. rag_chunks' NOT NULL drops for the refill window and
is restored here only when the table has no NULL embeddings afterwards
(i.e. it was empty); otherwise the runner restores it when its refill
completes.

Also creates the canonical HNSW cosine index set: the live DB had NO vector
indexes at all (G6's scripts/database/create_indexes.sql was never applied),
so this is their first landing — through a migration, as G6 requires.

Seeds the 'embedding' stamp at the post-migration reality (settings' model @
1024). Per-table 'reembed:<t>' stamps seed 'pending' only where rows exist
(they need the runner); empty tables stamp 'done' so fresh installs boot
without a phantom warning.

Revision ID: b6e2f9a41c73
Revises: a7c5e91d3f28
Create Date: 2026-07-19
"""
import json
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "b6e2f9a41c73"
down_revision = "a7c5e91d3f28"
branch_labels = None
depends_on = None

NEW_DIM = 1024
OLD_DIM = 384

# The nine vector-bearing tables as of this revision (runtime code never
# hardcodes this list — src/memory/store_meta.py introspects the catalog;
# a migration is a point-in-time statement, so a literal list is correct).
VECTOR_COLUMNS = [
    ("episodic_memory", "embedding"),
    ("episodic_chunks", "embedding"),
    ("context_clusters", "embedding"),
    ("codex_entities", "embedding"),
    ("procedural_memory", "embedding"),
    ("decisions", "embedding"),
    ("rag_chunks", "embedding"),
    ("batch_summaries", "embedding"),
    ("conversation_summaries", "embedding"),
]


def _retype_all(dim: int) -> None:
    for table, col in VECTOR_COLUMNS:
        op.execute(f"DROP INDEX IF EXISTS idx_{table}_embedding")
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {col} "
            f"TYPE vector({dim}) USING NULL")
        op.execute(
            f"CREATE INDEX idx_{table}_embedding ON {table} "
            f"USING hnsw ({col} vector_cosine_ops)")


def _restore_rag_not_null_if_clean() -> None:
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM rag_chunks WHERE embedding IS NULL)
            THEN
                ALTER TABLE rag_chunks ALTER COLUMN embedding SET NOT NULL;
            END IF;
        END $$
    """)


def upgrade() -> None:
    op.create_table(
        "store_meta",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", postgresql.JSONB(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("key"),
    )

    op.alter_column("rag_chunks", "embedding", nullable=True)
    _retype_all(NEW_DIM)
    _restore_rag_not_null_if_clean()

    # Seed the stamps. Model name comes from settings so the guard agrees
    # with the active config; dim is THIS migration's literal target.
    from src.api.config import settings
    conn = op.get_bind()
    now = datetime.now(timezone.utc)
    stamp = {"model": settings.embedding_model_name, "dim": NEW_DIM,
             "stamped_at": now.isoformat()}
    conn.execute(
        sa.text("INSERT INTO store_meta (key, value, updated_at) "
                "VALUES ('embedding', CAST(:v AS jsonb), :ts)"),
        {"v": json.dumps(stamp), "ts": now})
    for table, _col in VECTOR_COLUMNS:
        has_rows = conn.execute(
            sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")).scalar()
        status = "pending" if has_rows else "done"
        conn.execute(
            sa.text("INSERT INTO store_meta (key, value, updated_at) "
                    "VALUES (:k, CAST(:v AS jsonb), :ts)"),
            {"k": f"reembed:{table}",
             "v": json.dumps({"status": status, "stamped_at": now.isoformat()}),
             "ts": now})


def downgrade() -> None:
    op.alter_column("rag_chunks", "embedding", nullable=True)
    _retype_all(OLD_DIM)
    _restore_rag_not_null_if_clean()
    op.drop_table("store_meta")
