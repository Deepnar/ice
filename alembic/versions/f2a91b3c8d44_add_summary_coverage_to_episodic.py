"""add summary_coverage to episodic_memory (C1)

Revision ID: f2a91b3c8d44
Revises: e8f3a26c5d17
Create Date: 2026-07-08

Measured must-term retention of summary_text (grounded summarisation, C1).
NULL = no summary or a legacy pre-C1 summary.
"""
from alembic import op
import sqlalchemy as sa

revision = "f2a91b3c8d44"
down_revision = "e8f3a26c5d17"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("episodic_memory", sa.Column("summary_coverage", sa.Float(), nullable=True))


def downgrade():
    op.drop_column("episodic_memory", "summary_coverage")
