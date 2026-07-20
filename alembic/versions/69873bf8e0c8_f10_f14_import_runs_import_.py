"""F10/F14: import_runs + import_conversations, episodic decay_immune_until + ts_provenance

Revision ID: 69873bf8e0c8
Revises: b6e2f9a41c73
Create Date: 2026-07-20 16:50:18.479356

Autogenerate also proposed dropping the manually-created indexes (HNSW etc.)
and re-aligning legacy column nullability — all pre-existing DB<->models
drift owned by earlier migrations, stripped from this one by hand.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '69873bf8e0c8'
down_revision: Union[str, Sequence[str], None] = 'b6e2f9a41c73'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'import_runs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('source_path', sa.Text(), nullable=False),
        sa.Column('source_format', sa.Text(), nullable=False),
        sa.Column('policy', sa.Text(), nullable=False),
        sa.Column('kind', sa.Text(), server_default='replay', nullable=False),
        sa.Column('status', sa.Text(), server_default='running', nullable=False),
        sa.Column('total_conversations', sa.Integer(), server_default='0', nullable=False),
        sa.Column('total_turns', sa.Integer(), server_default='0', nullable=False),
        sa.Column('done_conversations', sa.Integer(), server_default='0', nullable=False),
        sa.Column('done_turns', sa.Integer(), server_default='0', nullable=False),
        sa.Column('skipped_conversations', sa.Integer(), server_default='0', nullable=False),
        sa.Column('failed_turns', sa.Integer(), server_default='0', nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('report', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'import_conversations',
        sa.Column('content_hash', sa.Text(), nullable=False),
        sa.Column('import_id', sa.UUID(), nullable=True),
        sa.Column('conversation_id', sa.UUID(), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('n_turns', sa.Integer(), server_default='0', nullable=False),
        sa.Column('imported_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['import_id'], ['import_runs.id']),
        sa.PrimaryKeyConstraint('content_hash'),
    )
    op.create_index(op.f('ix_import_conversations_import_id'),
                    'import_conversations', ['import_id'], unique=False)
    op.add_column('episodic_memory',
                  sa.Column('ts_provenance', sa.Text(),
                            server_default='original', nullable=False))
    op.add_column('episodic_memory',
                  sa.Column('decay_immune_until', sa.DateTime(timezone=True),
                            nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('episodic_memory', 'decay_immune_until')
    op.drop_column('episodic_memory', 'ts_provenance')
    op.drop_index(op.f('ix_import_conversations_import_id'),
                  table_name='import_conversations')
    op.drop_table('import_conversations')
    op.drop_table('import_runs')
