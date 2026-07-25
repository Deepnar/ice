"""B1: curated labels carry a schema version and multi-label context

Revision ID: c4d7e91a2b58
Revises: 69873bf8e0c8
Create Date: 2026-07-25

Context-reliance stopped being a single choice in schema v2 — it is four
independent signals — so a user correction is a SET of labels, not one string.
Old rows keep their ``corrected_context_reliance`` and are stamped
``schema_version = 1``; the fine-tune worker masks their context head rather than
collapsing v2 into v1 (see src/workers/fine_tune.py).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c4d7e91a2b58"
down_revision = "69873bf8e0c8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("curated_labels",
                  sa.Column("corrected_context_labels",
                            postgresql.ARRAY(sa.Text()),
                            nullable=True, server_default="{}"))
    op.add_column("curated_labels",
                  sa.Column("schema_version", sa.Integer(),
                            nullable=False, server_default="1"))


def downgrade() -> None:
    op.drop_column("curated_labels", "schema_version")
    op.drop_column("curated_labels", "corrected_context_labels")
