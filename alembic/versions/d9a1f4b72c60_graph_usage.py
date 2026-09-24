"""v3 graph usage is distinct from source observations.

Revision ID: d9a1f4b72c60
Revises: c8f0d21a6e39
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "d9a1f4b72c60"
down_revision = "c8f0d21a6e39"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("codex_edges", sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("codex_edges", sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("codex_edges", sa.Column("observed_batches", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True))


def downgrade():
    op.drop_column("codex_edges", "observed_batches")
    op.drop_column("codex_edges", "last_accessed_at")
    op.drop_column("codex_edges", "usage_count")
