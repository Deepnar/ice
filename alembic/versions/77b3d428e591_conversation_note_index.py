"""Index independently sourced conversation-summary notes for query selection."""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = '77b3d428e591'
down_revision = 'fb29355706c8'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'conversation_notes',
        sa.Column('conversation_id', UUID(as_uuid=True),
                  sa.ForeignKey('conversation_summaries.conversation_id', ondelete='CASCADE'),
                  primary_key=True),
        sa.Column('ordinal', sa.Integer(), primary_key=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('mode', sa.Text(), nullable=False),
        sa.Column('recorded_range', sa.Text(), nullable=True),
        sa.Column('source_ids', JSONB(), nullable=False),
        sa.Column('batch_ids', JSONB(), nullable=False),
        sa.Column('embedding', Vector(1024), nullable=False),
    )
    op.create_index('idx_conversation_notes_embedding', 'conversation_notes',
                    ['embedding'], postgresql_using='hnsw',
                    postgresql_ops={'embedding': 'vector_cosine_ops'})


def downgrade():
    op.drop_index('idx_conversation_notes_embedding', table_name='conversation_notes')
    op.drop_table('conversation_notes')
