"""v3 rolling summary source snapshot; no historical inference/backfill."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = 'a6d4e902b173'
down_revision = 'f3c8d5e02b19'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('conversation_summaries', sa.Column('source_manifest', JSONB(), nullable=True))


def downgrade():
    op.drop_column('conversation_summaries', 'source_manifest')
