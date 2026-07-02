"""add episodic_cluster_links

Revision ID: 03733c594868
Revises: 7c35ee6e2e59
Create Date: 2026-06-23 20:19:05.332612

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '03733c594868'
down_revision: Union[str, Sequence[str], None] = '7c35ee6e2e59'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'episodic_cluster_links',
        sa.Column('episodic_id', sa.UUID(), sa.ForeignKey('episodic_memory.id'), nullable=False),
        sa.Column('cluster_id', sa.UUID(), sa.ForeignKey('context_clusters.id'), nullable=False),
        sa.PrimaryKeyConstraint('episodic_id', 'cluster_id')
    )
    op.alter_column('episodic_memory', 'cluster_id', nullable=True)


def downgrade() -> None:
    op.alter_column('episodic_memory', 'cluster_id', nullable=False)
    op.drop_table('episodic_cluster_links')