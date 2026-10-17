"""CODEX_V4 D9: drop the v2/v3 conversation graph's storage.

Facts (`codex_facts`, c0d4e7f1a2b3) replaced conversation edges, attributed
claims, the relation-gap ledger and compaction snapshots; nothing reads or
writes those any more. What goes:

  * `codex_claim_links`, `codex_claims`, `codex_relation_gaps`, `codex_snapshots`
    are dropped;
  * conversation-sourced `codex_edges` rows are deleted — the table stays for
    the code graph (`static_analysis`) and project facts (`derived`);
  * pending legacy graph-conflict review items (`codex_contradiction`,
    `codex_reconciliation`) are marked `archived` — their approve handler is gone.

⚑ Take a `pg_dump` of the database BEFORE upgrading; its path goes in the
CLEANUP ledger. The downgrade recreates the four tables EMPTY and does not
restore deleted edges or review statuses — the dump is the way back.

Revision ID: d5e8a1c3b7f2
Revises: c0d4e7f1a2b3
Create Date: 2026-10-16
"""
import logging

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "d5e8a1c3b7f2"
down_revision = "c0d4e7f1a2b3"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

_LEGACY_REVIEW = ("codex_contradiction", "codex_reconciliation")


def upgrade() -> None:
    conn = op.get_bind()
    counts = {table: conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar()
              for table in ("codex_claim_links", "codex_claims", "codex_relation_gaps",
                            "codex_snapshots")}
    op.drop_table("codex_claim_links")
    op.drop_table("codex_claims")
    op.drop_table("codex_relation_gaps")
    op.drop_table("codex_snapshots")
    counts["codex_edges (conversation)"] = conn.execute(sa.text(
        "DELETE FROM codex_edges WHERE source = 'conversation'")).rowcount
    counts["review_queue archived"] = conn.execute(sa.text(
        "UPDATE review_queue SET status = 'archived' "
        "WHERE status = 'pending' AND item_type = ANY(:types)"),
        {"types": list(_LEGACY_REVIEW)}).rowcount
    for what, n in counts.items():
        logger.info("codex_v4 legacy removal: %s: %d row(s)", what, n)


def downgrade() -> None:
    # Empty shapes only (from e2b7c4d91a08 + 7e4c9d2a0b65, c704abf1917e and
    # 675b74e56988). Rows come back from the pre-upgrade pg_dump, nowhere else.
    op.create_table(
        "codex_snapshots",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("entity_id", sa.UUID(), nullable=False),
        sa.Column("snapshot_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_event_id", sa.UUID(), nullable=False),
        sa.Column("full_state", JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["entity_id"], ["codex_entities.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "codex_relation_gaps",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("proposed_relation", sa.Text(), nullable=False),
        sa.Column("raw_relation", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("negated", sa.Boolean(), nullable=True),
        sa.Column("batch_id", sa.UUID(), nullable=True),
        sa.Column("conversation_id", sa.UUID(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("suggested_relation", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_codex_relation_gaps_batch_id", "codex_relation_gaps", ["batch_id"])
    op.create_index("ix_codex_relation_gaps_proposed_relation", "codex_relation_gaps",
                    ["proposed_relation"])
    op.create_index("ix_codex_relation_gaps_status", "codex_relation_gaps", ["status"])
    op.create_table(
        "codex_claims",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_batch", UUID(as_uuid=True), nullable=False),
        sa.Column("episodic_id", UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", UUID(as_uuid=True),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("raw_sha256", sa.Text(), nullable=False),
        sa.Column("start", sa.Integer(), nullable=False),
        sa.Column("end", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("sentence", sa.Text(), nullable=False),
        sa.Column("sentence_sha256", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("verification", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("source_batch", "raw_sha256", "start", "end", "sentence_sha256",
                            name="uq_codex_claim_span"),
    )
    op.create_table(
        "codex_claim_links",
        sa.Column("claim_id", UUID(as_uuid=True),
                  sa.ForeignKey("codex_claims.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("edge_id", UUID(as_uuid=True),
                  sa.ForeignKey("codex_edges.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("relation_verification", JSONB(), nullable=True),
    )
    op.create_index("ix_codex_claims_source_batch", "codex_claims", ["source_batch"])
    op.create_index("ix_codex_claims_episodic_id", "codex_claims", ["episodic_id"])
    op.execute("CREATE INDEX ix_codex_claims_lexical ON codex_claims "
               "USING gin (to_tsvector('english', text))")
