"""add abstract_text to episodic_memory (C3)

Revision ID: b9e4f7a2c810
Revises: a3d47e91c256
Create Date: 2026-07-08

One-line abstract — third level of the raw → summary → abstract hierarchy,
generated in the same LLM call as the grounded summary; consumed only by
budget degradation.
"""
from alembic import op
import sqlalchemy as sa

revision = "b9e4f7a2c810"
down_revision = "a3d47e91c256"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("episodic_memory", sa.Column("abstract_text", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("episodic_memory", "abstract_text")
