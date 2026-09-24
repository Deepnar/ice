"""Keep source-to-relation verdicts on Codex claim links."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "7e4c9d2a0b65"
down_revision = "c742a88eeb19"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("codex_claim_links", sa.Column(
        "relation_verification", postgresql.JSONB(), nullable=True))


def downgrade():
    op.drop_column("codex_claim_links", "relation_verification")
