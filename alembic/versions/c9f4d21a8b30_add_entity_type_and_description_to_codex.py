"""add entity_type and description to codex_entities (roadmap A7)

Revision ID: c9f4d21a8b30
Revises: b7e2a91c4f05
Create Date: 2026-07-06

A7 — typed, rich-note codex nodes. entity_type: structural node type
(inferred from relations for conversational entities; deterministic for
code entities in E1b). description: the enriched "note body" (Obsidian-
style) that context_payload is assembled from. Both backfill safely with
defaults on existing rows.
"""
from alembic import op
import sqlalchemy as sa

revision = "c9f4d21a8b30"
down_revision = "b7e2a91c4f05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("codex_entities",
                  sa.Column("entity_type", sa.Text(), nullable=False, server_default="entity"))
    op.add_column("codex_entities",
                  sa.Column("description", sa.Text(), nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("codex_entities", "description")
    op.drop_column("codex_entities", "entity_type")
