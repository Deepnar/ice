"""Index independently sourced batch-summary notes for query selection."""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = '80f1ac593d72'
down_revision = '7e4c9d2a0b65'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'batch_notes',
        sa.Column('summary_id', UUID(as_uuid=True),
                  sa.ForeignKey('batch_summaries.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('ordinal', sa.Integer(), primary_key=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('mode', sa.Text(), nullable=False),
        sa.Column('recorded_range', sa.Text(), nullable=True),
        sa.Column('source_ids', JSONB(), nullable=False),
        sa.Column('batch_ids', JSONB(), nullable=False),
        sa.Column('embedding', Vector(1024), nullable=False),
    )
    op.create_index('idx_batch_notes_embedding', 'batch_notes', ['embedding'],
                    postgresql_using='hnsw',
                    postgresql_ops={'embedding': 'vector_cosine_ops'})


def downgrade():
    op.drop_index('idx_batch_notes_embedding', table_name='batch_notes')
    op.drop_table('batch_notes')
