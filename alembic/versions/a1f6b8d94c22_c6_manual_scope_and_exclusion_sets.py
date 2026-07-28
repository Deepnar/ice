"""C6: manual-scope inclusion + exclusion sets on conversations

Revision ID: a1f6b8d94c22
Revises: c4d7e91a2b58
Create Date: 2026-07-28

C6's scope-semantics rework needs three id sets that had no columns:

  included_conversation_ids — the cross-chat set a `manual` conversation reads
                              (its own id is always implicit).
  excluded_conversation_ids — never retrieve from these conversations.
  excluded_cluster_ids      — never retrieve turns linked to these clusters.

ARRAY(UUID) to match the existing `cluster_ids` style (and A5's batch-set
primitive, which already consumes id lists) rather than a JSONB blob.

Also DROPS `custom_filter`. It was the v1 definition of `manual` scope — a
user-authored SQL WHERE fragment stapled onto every episodic query
(ARCHITECTURE.md §8.1, `docs/outdated/`). It was never read by retrieval and
its allowlist validator was never written. `manual` now means "the user picked
the conversations, and that pick beats the automatic cluster picker", which is
a different feature; keeping both would give one mode two contradictory
meanings. Verdict pre-recorded in specs/G_mechanical.md (G20 sweep) and
confirmed by the user 2026-07-28. Downgrade restores the column (empty).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a1f6b8d94c22"
down_revision = "c4d7e91a2b58"
branch_labels = None
depends_on = None

_ID_SET_COLUMNS = (
    "included_conversation_ids",
    "excluded_conversation_ids",
    "excluded_cluster_ids",
)


def upgrade():
    for name in _ID_SET_COLUMNS:
        op.add_column(
            "conversations",
            sa.Column(name, postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
                      nullable=True),
        )
    op.drop_column("conversations", "custom_filter")


def downgrade():
    op.add_column("conversations", sa.Column("custom_filter", sa.Text(),
                                             nullable=True))
    for name in reversed(_ID_SET_COLUMNS):
        op.drop_column("conversations", name)
