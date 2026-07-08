"""add episodic_chunks table (C2)

Revision ID: a3d47e91c256
Revises: f2a91b3c8d44
Create Date: 2026-07-08

Retrieval-grade chunks of document turns. ON DELETE CASCADE from the parent
turn (C10 deletion look-ahead). Vector(384) now; C17's 1024 re-embed covers
this table via G23's migration runner.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = "a3d47e91c256"
down_revision = "f2a91b3c8d44"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "episodic_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("episodic_memory.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
    )
    op.create_index("ix_episodic_chunks_turn_id", "episodic_chunks", ["turn_id"])


def downgrade():
    op.drop_index("ix_episodic_chunks_turn_id", table_name="episodic_chunks")
    op.drop_table("episodic_chunks")
