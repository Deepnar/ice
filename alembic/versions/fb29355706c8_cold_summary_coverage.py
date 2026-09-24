"""Preserve summary source membership through archival."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 'fb29355706c8'
down_revision = 'ea182446f5b7'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('cold_storage', sa.Column('batch_summary_id', UUID(as_uuid=True),
        sa.ForeignKey('batch_summaries.id', ondelete='SET NULL'), nullable=True))
    op.create_index('ix_cold_storage_batch_summary_id', 'cold_storage', ['batch_summary_id'])


def downgrade():
    op.drop_index('ix_cold_storage_batch_summary_id', table_name='cold_storage')
    op.drop_column('cold_storage', 'batch_summary_id')
