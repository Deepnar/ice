"""Preserve original retrieval chunks in cold storage."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

revision = 'ea182446f5b7'
down_revision = 'd9071335e4a6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('cold_chunks',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('turn_id', UUID(as_uuid=True), sa.ForeignKey('cold_storage.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1024), nullable=True))
    op.create_index('ix_cold_chunks_turn_id', 'cold_chunks', ['turn_id'])


def downgrade():
    op.drop_table('cold_chunks')
