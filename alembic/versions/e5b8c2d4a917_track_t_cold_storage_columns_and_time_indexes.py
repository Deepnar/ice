"""Track T: cold_storage resurrection columns + time-axis indexes (T3)

Revision ID: e5b8c2d4a917
Revises: a1c7e5f2d9b3
Create Date: 2026-07-12

cold_storage gains conversation_id / is_private / batch_id (D12): the old
cold move dropped the conversation link and the privacy flag, which made
resurrection impossible and would have leaked incognito turns into time
queries. Legacy rows (NULL conversation_id) become cite-only — searchable,
never resurrected. Legacy rows cannot be private (cold rows predate G16 —
incognito didn't exist when they froze), so the FALSE default is correct,
not a guess.

Indexes serve the new timescope window predicates on episodic/cold
timestamps and the valid_at(T) codex reads (also a G6 down-payment: indexes
land via Alembic, not the orphan SQL script).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "e5b8c2d4a917"
down_revision = "a1c7e5f2d9b3"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("cold_storage",
                  sa.Column("conversation_id", UUID(as_uuid=True), nullable=True))
    op.add_column("cold_storage",
                  sa.Column("is_private", sa.Boolean(), nullable=False,
                            server_default=sa.false()))
    op.add_column("cold_storage",
                  sa.Column("batch_id", UUID(as_uuid=True), nullable=True))

    op.create_index("ix_episodic_memory_timestamp", "episodic_memory", ["timestamp"])
    op.create_index("ix_cold_storage_timestamp", "cold_storage", ["timestamp"])
    op.create_index("ix_codex_edges_source_valid", "codex_edges",
                    ["source_id", "valid_until"])
    op.create_index("ix_codex_edges_target_valid", "codex_edges",
                    ["target_id", "valid_until"])

    # Expected ≈0 because of the pre-T3 archived-freeze bug (rows could never
    # reach the cold threshold); any existing rows are pre-incognito and
    # legally is_private=FALSE.
    count = op.get_bind().execute(sa.text("SELECT count(*) FROM cold_storage")).scalar()
    print(f"[e5b8c2d4a917] existing cold_storage rows (become cite-only): {count}")


def downgrade():
    op.drop_index("ix_codex_edges_target_valid", table_name="codex_edges")
    op.drop_index("ix_codex_edges_source_valid", table_name="codex_edges")
    op.drop_index("ix_cold_storage_timestamp", table_name="cold_storage")
    op.drop_index("ix_episodic_memory_timestamp", table_name="episodic_memory")
    op.drop_column("cold_storage", "batch_id")
    op.drop_column("cold_storage", "is_private")
    op.drop_column("cold_storage", "conversation_id")
