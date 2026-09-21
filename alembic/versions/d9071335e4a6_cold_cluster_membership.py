"""Preserve cold cluster visibility; NULL legacy membership remains unknown."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision = 'd9071335e4a6'
down_revision = 'c8f60224d395'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('cold_storage', sa.Column('cluster_ids', ARRAY(UUID(as_uuid=True)), nullable=True))
    op.add_column('cold_storage', sa.Column('cluster_id', UUID(as_uuid=True), nullable=True))


def downgrade():
    op.drop_column('cold_storage', 'cluster_id')
    op.drop_column('cold_storage', 'cluster_ids')
