"""C12: documents registry, conversation kind, RAG tables dropped

Revision ID: 5fe5ad26480b
Revises: a1f6b8d94c22
Create Date: 2026-07-28 21:01:32.288297

Hand-trimmed from the autogenerate output: it also wanted to drop every index
created OUTSIDE the ORM models (the HNSW vector indexes from G23/C17 and the
partial/composite indexes from Track T and the E-core). Those are deliberate
and must survive — only the C12 changes are below.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5fe5ad26480b'
down_revision: Union[str, Sequence[str], None] = 'a1f6b8d94c22'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── conversations gain a kind ───────────────────────────────────────────
    # 'chat' | 'document' | 'transcript'. Non-chat conversations are the
    # ingested content; they are opt-in in retrieval (C12 D2), resolved in
    # services/scoping.py and enforced by C6's existing exclusion filter.
    op.add_column('conversations',
                  sa.Column('kind', sa.Text(), nullable=False,
                            server_default='chat'))
    op.create_index('ix_conversations_kind', 'conversations', ['kind'],
                    postgresql_where=sa.text("kind <> 'chat'"))

    # ── the document registry ───────────────────────────────────────────────
    op.create_table(
        'documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('conversation_id', sa.UUID(), nullable=False),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('file_type', sa.Text(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False, server_default='document'),
        sa.Column('origin', sa.Text(), nullable=False, server_default='upload'),
        sa.Column('sha256', sa.Text(), nullable=False),
        sa.Column('byte_size', sa.BigInteger(), nullable=False,
                  server_default='0'),
        sa.Column('n_sections', sa.Integer(), nullable=False,
                  server_default='0'),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('source_path', sa.Text(), nullable=True),
        sa.Column('project_id', sa.UUID(), nullable=True),
        sa.Column('knowledge_shared', sa.Boolean(), nullable=False,
                  server_default=sa.text('false')),
        sa.Column('shared_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Text(), nullable=False, server_default='pending'),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'],
                                ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('sha256'),
    )
    op.create_index('ix_documents_conversation_id', 'documents',
                    ['conversation_id'])

    # Rows are never deleted on disable: first_enabled_at is the evidence the
    # knowledge-promotion latch reads (two rows = a second conversation has
    # reached for this document).
    op.create_table(
        'document_links',
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('conversation_id', sa.UUID(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False,
                  server_default=sa.text('true')),
        sa.Column('first_enabled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'],
                                ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'],
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('document_id', 'conversation_id'),
    )
    op.create_index('ix_document_links_conversation', 'document_links',
                    ['conversation_id'],
                    postgresql_where=sa.text('enabled'))

    # C12 D7: a promoted paste points at what it became. The turn is never
    # rewritten — what the user pasted stays verbatim.
    op.add_column('episodic_memory',
                  sa.Column('promoted_document_id', sa.UUID(), nullable=True))
    op.create_foreign_key('fk_episodic_promoted_document', 'episodic_memory',
                          'documents', ['promoted_document_id'], ['id'])

    # ── the v1 RAG store dies ───────────────────────────────────────────────
    # rag_chunks' only reader was a globally-unscoped lookup behind a
    # five-English-noun gate; its only writer was a watchdog script nothing
    # started. Documents are memory now, retrieved by the ordinary legs.
    op.drop_table('rag_chunks')
    op.drop_table('rag_documents')


def downgrade() -> None:
    op.create_table(
        'rag_documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('file_type', sa.Text(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.execute("CREATE TABLE rag_chunks ("
               "id UUID PRIMARY KEY, "
               "document_id UUID NOT NULL REFERENCES rag_documents(id), "
               "chunk_index INTEGER NOT NULL, "
               "chunk_text TEXT NOT NULL, "
               "embedding vector(1024) NOT NULL)")

    op.drop_constraint('fk_episodic_promoted_document', 'episodic_memory',
                       type_='foreignkey')
    op.drop_column('episodic_memory', 'promoted_document_id')
    op.drop_index('ix_document_links_conversation', table_name='document_links')
    op.drop_table('document_links')
    op.drop_index('ix_documents_conversation_id', table_name='documents')
    op.drop_table('documents')
    op.drop_index('ix_conversations_kind', table_name='conversations')
    op.drop_column('conversations', 'kind')
