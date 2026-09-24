"""v3 authoritative source boundaries and cold timestamp provenance.

Revision ID: c8f0d21a6e39
Revises: a1c4e7b90d22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c8f0d21a6e39"
down_revision = "a1c4e7b90d22"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("episodic_memory", sa.Column("source_spans", postgresql.JSONB(), nullable=True))
    op.add_column("cold_storage", sa.Column("source_spans", postgresql.JSONB(), nullable=True))
    op.add_column("cold_storage", sa.Column("ts_provenance", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("cold_storage", "ts_provenance")
    op.drop_column("cold_storage", "source_spans")
    op.drop_column("episodic_memory", "source_spans")
