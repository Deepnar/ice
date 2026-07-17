"""drop sentinel tables (D2 — the maintenance agent replaces the Sentinel)

The sentinel's two real checks (pending-edge pileup, stale pending_items
slot) live on as maintenance-agent detectors 2 and 5; the rest of the rule
engine was stubs (audit verdict: removal). Rule rows are archived into the
migration log before the drop — they were seed rules, nothing user-authored.

Revision ID: f7a3d9c21e46
Revises: e5b8c2d4a917
Create Date: 2026-07-17
"""
import json
import logging

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "f7a3d9c21e46"
down_revision = "e5b8c2d4a917"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT name, description, is_active, trigger_type, "
        "       trigger_conditions, action_type, action_payload, "
        "       cooldown_seconds "
        "FROM sentinel_rules ORDER BY name")).mappings().all()
    for r in rows:
        logger.info("archived sentinel rule: %s",
                    json.dumps(dict(r), default=str))
    logger.info("sentinel_rules archived to log: %d row(s)", len(rows))

    op.drop_table("sentinel_events")
    op.drop_table("sentinel_rules")


def downgrade() -> None:
    # Shapes copied from 675b74e56988_initial_schema (rows are not restored —
    # they were seed data; re-run the archived seed script if ever needed).
    op.create_table(
        "sentinel_rules",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("trigger_type", sa.Text(), nullable=False),
        sa.Column("trigger_conditions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("action_type", sa.Text(), nullable=False),
        sa.Column("action_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("cooldown_seconds", sa.Integer(), nullable=True),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "sentinel_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("rule_id", sa.UUID(), nullable=False),
        sa.Column("fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trigger_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("action_taken", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["rule_id"], ["sentinel_rules.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
