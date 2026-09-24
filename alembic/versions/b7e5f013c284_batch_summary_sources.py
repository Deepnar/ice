"""v3 batch summary source snapshots; legacy caches require rebuilding."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = 'b7e5f013c284'
down_revision = 'a6d4e902b173'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('batch_summaries', sa.Column('source_manifest', JSONB(), nullable=True))


def downgrade():
    op.drop_column('batch_summaries', 'source_manifest')
