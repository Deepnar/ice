"""add tags column to context_clusters

Revision ID: 2e6904706153
Revises: 03733c594868
Create Date: 2026-06-23 22:48:06.782267

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2e6904706153'
down_revision: Union[str, Sequence[str], None] = '03733c594868'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('context_clusters', sa.Column('tags', sa.ARRAY(sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('context_clusters', 'tags')