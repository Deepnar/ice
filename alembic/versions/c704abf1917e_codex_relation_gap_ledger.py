"""codex relation gap ledger

Revision ID: c704abf1917e
Revises: 9b3e71c2fa48
Create Date: 2026-08-04 20:15:20.466967

⚠ HAND-TRIMMED. `--autogenerate` emitted 20 additional `drop_index` calls and
five `alter_column`s loosening NOT NULL. Every one of them was spurious: the
HNSW vector indexes and the partial/composite indexes were created by earlier
migrations and raw SQL rather than declared on the models, so autogenerate sees
them as drift and proposes deleting them — including all seven
`idx_*_embedding` indexes G23 built. Applying that would have silently removed
every vector index in the database. Only the CREATE for the new table is kept.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c704abf1917e'
down_revision: Union[str, Sequence[str], None] = '9b3e71c2fa48'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the relation-gap ledger. Nothing else — see the module note."""
    op.create_table(
        'codex_relation_gaps',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('proposed_relation', sa.Text(), nullable=False),
        sa.Column('raw_relation', sa.Text(), nullable=False),
        sa.Column('subject', sa.Text(), nullable=False),
        sa.Column('object', sa.Text(), nullable=False),
        sa.Column('negated', sa.Boolean(), nullable=True),
        sa.Column('batch_id', sa.UUID(), nullable=True),
        sa.Column('conversation_id', sa.UUID(), nullable=True),
        sa.Column('status', sa.Text(), nullable=True),
        sa.Column('suggested_relation', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_codex_relation_gaps_batch_id'),
                    'codex_relation_gaps', ['batch_id'], unique=False)
    op.create_index(op.f('ix_codex_relation_gaps_proposed_relation'),
                    'codex_relation_gaps', ['proposed_relation'], unique=False)
    op.create_index(op.f('ix_codex_relation_gaps_status'),
                    'codex_relation_gaps', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_codex_relation_gaps_status'),
                  table_name='codex_relation_gaps')
    op.drop_index(op.f('ix_codex_relation_gaps_proposed_relation'),
                  table_name='codex_relation_gaps')
    op.drop_index(op.f('ix_codex_relation_gaps_batch_id'),
                  table_name='codex_relation_gaps')
    op.drop_table('codex_relation_gaps')
