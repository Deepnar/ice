"""CODEX_V4 facts: verified fact sentences, their sources and entity links."""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = 'c0d4e7f1a2b3'
down_revision = '80f1ac593d72'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'codex_facts',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('text_sha256', sa.Text(), nullable=False),
        sa.Column('speaker', sa.Text(), nullable=False),
        sa.Column('modality', sa.Text(), nullable=False),
        sa.Column('subject', sa.Text(), nullable=False, server_default=''),
        sa.Column('relation', sa.Text(), nullable=False, server_default=''),
        sa.Column('object', sa.Text(), nullable=False, server_default=''),
        sa.Column('negated', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('embedding', Vector(1024), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('support_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('usage', sa.Float(), nullable=False, server_default='0'),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('mentioned_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=False),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('superseded_by', UUID(as_uuid=True), sa.ForeignKey('codex_facts.id'), nullable=True),
        sa.Column('flags', JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_codex_facts_text_sha256', 'codex_facts', ['text_sha256'])
    op.execute("CREATE INDEX ix_codex_facts_lexical ON codex_facts "
               "USING gin (to_tsvector('english', text))")
    op.create_index('ix_codex_facts_embedding', 'codex_facts', ['embedding'],
                    postgresql_using='hnsw',
                    postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.execute("CREATE INDEX ix_codex_facts_current ON codex_facts (mentioned_at) "
               "WHERE valid_until IS NULL")
    op.create_table(
        'codex_fact_sources',
        sa.Column('fact_id', UUID(as_uuid=True),
                  sa.ForeignKey('codex_facts.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('source_batch', UUID(as_uuid=True), primary_key=True),
        sa.Column('window_sha256', sa.Text(), primary_key=True),
        sa.Column('episodic_id', UUID(as_uuid=True), nullable=True),
        sa.Column('conversation_id', UUID(as_uuid=True), nullable=True),
        sa.Column('window', JSONB(), nullable=False),
        sa.Column('quote', sa.Text(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('verification', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_codex_fact_sources_episodic_id', 'codex_fact_sources', ['episodic_id'])
    op.create_index('ix_codex_fact_sources_conversation_id', 'codex_fact_sources',
                    ['conversation_id'])
    op.create_index('ix_codex_fact_sources_source_batch', 'codex_fact_sources', ['source_batch'])
    op.create_table(
        'codex_fact_entities',
        sa.Column('fact_id', UUID(as_uuid=True),
                  sa.ForeignKey('codex_facts.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('entity_id', UUID(as_uuid=True),
                  sa.ForeignKey('codex_entities.id'), primary_key=True),
        sa.Column('role', sa.Text(), primary_key=True),
    )
    op.create_index('ix_codex_fact_entities_entity_id', 'codex_fact_entities', ['entity_id'])


def downgrade():
    op.drop_table('codex_fact_entities')
    op.drop_table('codex_fact_sources')
    op.drop_table('codex_facts')
