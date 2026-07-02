"""add conversation_id to context_clusters

Revision ID: 3648cf73064a
Revises: 2e6904706153
Create Date: 2026-06-24 15:56:01.107218

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3648cf73064a'
down_revision: Union[str, Sequence[str], None] = '2e6904706153'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('context_clusters', sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(None, 'context_clusters', 'conversations', ['conversation_id'], ['id'])

def downgrade() -> None:
    op.drop_constraint(None, 'context_clusters', type_='foreignkey')
    op.drop_column('context_clusters', 'conversation_id')
