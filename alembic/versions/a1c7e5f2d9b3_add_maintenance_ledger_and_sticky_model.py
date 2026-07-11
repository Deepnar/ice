"""add maintenance_ledger + conversations sticky-model columns (C7 D3/D9)

Revision ID: a1c7e5f2d9b3
Revises: b9e4f7a2c810
Create Date: 2026-07-11

maintenance_ledger: per-job schedule state for the in-process maintenance
runtime that replaces Celery beat (survives restarts, feeds overdue catch-up,
doubles as the duplicate-instance claim lock, later read by Track D's agent).

conversations.sticky_model / consecutive_shifts: session-stickiness state
moved out of main.py's in-memory SESSION_STATE dict (G8, without Redis).
"""
from alembic import op
import sqlalchemy as sa

revision = "a1c7e5f2d9b3"
down_revision = "b9e4f7a2c810"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "maintenance_ledger",
        sa.Column("job_name", sa.Text(), primary_key=True),
        sa.Column("last_started", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_finished", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status", sa.Text(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("runs", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("conversations",
                  sa.Column("sticky_model", sa.Text(), nullable=True))
    op.add_column("conversations",
                  sa.Column("consecutive_shifts", sa.Integer(),
                            nullable=False, server_default="0"))


def downgrade():
    op.drop_column("conversations", "consecutive_shifts")
    op.drop_column("conversations", "sticky_model")
    op.drop_table("maintenance_ledger")
