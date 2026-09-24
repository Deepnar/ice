"""Preserve document and decay eligibility metadata on cold sources."""
from alembic import op
import sqlalchemy as sa

revision = 'c742a88eeb19'
down_revision = '77b3d428e591'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('cold_storage', sa.Column('is_document', sa.Boolean(), nullable=True))
    op.add_column('cold_storage', sa.Column('decay_score', sa.Float(), nullable=True))


def downgrade():
    op.drop_column('cold_storage', 'decay_score')
    op.drop_column('cold_storage', 'is_document')
