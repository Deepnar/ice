"""v3 attributed, independently searchable Codex sentences."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

revision = 'e2b7c4d91a08'
down_revision = 'd9a1f4b72c60'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('codex_claims',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('source_batch', UUID(as_uuid=True), nullable=False),
        sa.Column('episodic_id', UUID(as_uuid=True), nullable=False),
        sa.Column('conversation_id', UUID(as_uuid=True), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('raw_sha256', sa.Text(), nullable=False),
        sa.Column('start', sa.Integer(), nullable=False),
        sa.Column('end', sa.Integer(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('sentence', sa.Text(), nullable=False),
        sa.Column('sentence_sha256', sa.Text(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1024), nullable=True),
        sa.Column('verification', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('source_batch','raw_sha256','start','end','sentence_sha256', name='uq_codex_claim_span'))
    op.create_table('codex_claim_links',
        sa.Column('claim_id', UUID(as_uuid=True), sa.ForeignKey('codex_claims.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('edge_id', UUID(as_uuid=True), sa.ForeignKey('codex_edges.id', ondelete='CASCADE'), primary_key=True))
    op.create_index('ix_codex_claims_source_batch','codex_claims',['source_batch'])
    op.create_index('ix_codex_claims_episodic_id','codex_claims',['episodic_id'])
    op.execute("CREATE INDEX ix_codex_claims_lexical ON codex_claims USING gin (to_tsvector('english', text))")


def downgrade():
    op.drop_table('codex_claim_links')
    op.drop_table('codex_claims')
