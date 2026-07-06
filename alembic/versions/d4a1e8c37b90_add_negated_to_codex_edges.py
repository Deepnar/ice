"""add negated (polarity) to codex_edges (roadmap A8)

Revision ID: d4a1e8c37b90
Revises: c9f4d21a8b30
Create Date: 2026-07-06

A8 — relation polarity. negated=True stores the negative of a relation
(X does NOT use Y, X distrusts Y) without doubling the controlled
vocabulary. Existing rows default to False (positive).
"""
from alembic import op
import sqlalchemy as sa

revision = "d4a1e8c37b90"
down_revision = "c9f4d21a8b30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("codex_edges",
                  sa.Column("negated", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("codex_edges", "negated")
