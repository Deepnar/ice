"""C4/C9: conversation_summaries + three-tier memory slots

C4: conversation_summaries — ONE evolving summary per conversation
(PK = conversation_id, ondelete CASCADE so C10's conversation deletion takes
the row with it).
C9 (D5): memory_slots grows scope_tier/project_id/conversation_id; legacy
rows backfill to 'global'. Uniqueness is a NULLS NOT DISTINCT unique index
(pg16) — the plain composite constraint would let (name,'global',NULL,NULL)
duplicate. Note: memory_slots had NO uniqueness on slot_name before this
migration (only the id pkey), despite an old comment claiming otherwise.

Revision ID: a7c5e91d3f28
Revises: c4e9b7d15a20
Create Date: 2026-07-19
"""
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision = "a7c5e91d3f28"
down_revision = "c4e9b7d15a20"
branch_labels = None
depends_on = None

SLOT_UNIQUE_INDEX = "uq_memory_slots_name_tier_anchor"


def upgrade() -> None:
    op.create_table(
        "conversation_summaries",
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("covers_through", sa.DateTime(timezone=True), nullable=True),
        sa.Column("covers_turns", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding", Vector(dim=384), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"],
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("conversation_id"),
    )

    op.add_column("memory_slots", sa.Column(
        "scope_tier", sa.Text(), nullable=False, server_default="global"))
    op.add_column("memory_slots", sa.Column("project_id", sa.UUID(), nullable=True))
    op.add_column("memory_slots", sa.Column("conversation_id", sa.UUID(), nullable=True))
    op.create_foreign_key("fk_memory_slots_project", "memory_slots",
                          "projects", ["project_id"], ["id"])
    op.create_foreign_key("fk_memory_slots_conversation", "memory_slots",
                          "conversations", ["conversation_id"], ["id"])
    # Legacy duplicates would break the unique index — keep the newest row of
    # any pre-existing same-name pair (none observed in the dev DB; belt and
    # suspenders for other installs).
    op.execute("""
        DELETE FROM memory_slots a USING memory_slots b
        WHERE a.slot_name = b.slot_name
          AND a.last_updated < b.last_updated
          AND a.scope_tier = 'global' AND b.scope_tier = 'global'
    """)
    op.execute(f"""
        CREATE UNIQUE INDEX {SLOT_UNIQUE_INDEX}
        ON memory_slots (slot_name, scope_tier, project_id, conversation_id)
        NULLS NOT DISTINCT
    """)


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {SLOT_UNIQUE_INDEX}")
    op.drop_constraint("fk_memory_slots_conversation", "memory_slots", type_="foreignkey")
    op.drop_constraint("fk_memory_slots_project", "memory_slots", type_="foreignkey")
    op.drop_column("memory_slots", "conversation_id")
    op.drop_column("memory_slots", "project_id")
    op.drop_column("memory_slots", "scope_tier")
    op.drop_table("conversation_summaries")
