"""add session_id and is_private to episodic_memory (C6/G16)

Revision ID: e8f3a26c5d17
Revises: d4a1e8c37b90
Create Date: 2026-07-08

session_id: one sitting = one session (30-min-gap heuristic, resolved at write
time). NULL on pre-existing rows — they predate session tracking.
is_private: G16 incognito flag for none-scoped conversations. Backfilled from
the owning conversation's current memory_scope_type so existing none-scoped
turns become invisible immediately.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "e8f3a26c5d17"
down_revision = "d4a1e8c37b90"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "episodic_memory",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "episodic_memory",
        sa.Column("is_private", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_episodic_memory_session_id", "episodic_memory", ["session_id"])
    op.create_index("ix_episodic_memory_is_private", "episodic_memory", ["is_private"])
    # Backfill privacy from the owning conversation's current scope.
    op.execute("""
        UPDATE episodic_memory e
        SET is_private = TRUE
        FROM conversations c
        WHERE e.conversation_id = c.id AND c.memory_scope_type = 'none'
    """)


def downgrade():
    op.drop_index("ix_episodic_memory_is_private", table_name="episodic_memory")
    op.drop_index("ix_episodic_memory_session_id", table_name="episodic_memory")
    op.drop_column("episodic_memory", "is_private")
    op.drop_column("episodic_memory", "session_id")
