"""add extraction_confidence to codex_edges (roadmap A3)

Revision ID: b7e2a91c4f05
Revises: 3648cf73064a
Create Date: 2026-07-06

Numeric per-edge extraction trust (0-1), seeded by NER grounding at write
time and raised by corroborating re-extractions. Existing edges default to
1.0 (they predate grounding; do not retroactively punish them).
"""
from alembic import op
import sqlalchemy as sa

revision = "b7e2a91c4f05"
down_revision = "3648cf73064a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "codex_edges",
        sa.Column("extraction_confidence", sa.Float(), nullable=False, server_default="1.0"),
    )


def downgrade() -> None:
    op.drop_column("codex_edges", "extraction_confidence")
