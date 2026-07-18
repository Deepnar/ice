"""E-coding core (E1/E1b): projects first-class + namespaced codex

New tables: projects, project_state, decisions (bi-temporal), tasks.
New columns: conversations.project_id, codex_entities.project_id/source,
codex_edges.source, procedural_memory.project_id.
daily_checklist ships as a VIEW over tasks (spec D1 — the drafted table was
dropped); architecture_clusters and development_patterns are deliberately
NEVER created (A7.4 communities / project-scoped procedural rows serve them).

Revision ID: c4e9b7d15a20
Revises: f7a3d9c21e46
Create Date: 2026-07-17
"""
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "c4e9b7d15a20"
down_revision = "f7a3d9c21e46"
branch_labels = None
depends_on = None

DAILY_CHECKLIST_VIEW = """
CREATE VIEW daily_checklist AS
SELECT t.id, t.project_id, p.slug AS project_slug, t.title, t.status,
       t.updated_at,
       (now() - t.updated_at > interval '14 days') AS stale
FROM tasks t
JOIN projects p ON p.id = t.project_id
WHERE t.status IN ('pending', 'active')
ORDER BY t.updated_at ASC
"""


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("roots", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("settings", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("slug"),
    )
    op.create_table(
        "project_state",
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("goal", sa.Text(), nullable=True),
        sa.Column("current_branch", sa.Text(), nullable=True),
        sa.Column("last_task_id", sa.UUID(), nullable=True),
        sa.Column("last_session_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_reconciled_commit", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("project_id"),
    )
    op.create_table(
        "decisions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("alternatives_rejected", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("files_affected", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("decision_type", sa.Text(), nullable=False, server_default="decision"),
        sa.Column("source_batch", sa.UUID(), nullable=True),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_by", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "tasks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("commit_hashes", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("files_changed", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column("conversations",
                  sa.Column("project_id", sa.UUID(), nullable=True))
    op.create_foreign_key("fk_conversations_project", "conversations",
                          "projects", ["project_id"], ["id"])
    op.add_column("codex_entities",
                  sa.Column("project_id", sa.UUID(), nullable=True))
    op.add_column("codex_entities",
                  sa.Column("source", sa.Text(), nullable=False,
                            server_default="conversation"))
    op.add_column("codex_edges",
                  sa.Column("source", sa.Text(), nullable=False,
                            server_default="conversation"))
    op.add_column("procedural_memory",
                  sa.Column("project_id", sa.UUID(), nullable=True))

    op.create_index("ix_decisions_project_valid", "decisions",
                    ["project_id", "valid_until"])
    op.create_index("ix_codex_entities_project_source", "codex_entities",
                    ["project_id", "source"])
    op.create_index("ix_tasks_project_status", "tasks",
                    ["project_id", "status"])

    op.execute(DAILY_CHECKLIST_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS daily_checklist")
    op.drop_index("ix_tasks_project_status", table_name="tasks")
    op.drop_index("ix_codex_entities_project_source", table_name="codex_entities")
    op.drop_index("ix_decisions_project_valid", table_name="decisions")
    op.drop_column("procedural_memory", "project_id")
    op.drop_column("codex_edges", "source")
    op.drop_column("codex_entities", "source")
    op.drop_column("codex_entities", "project_id")
    op.drop_constraint("fk_conversations_project", "conversations",
                       type_="foreignkey")
    op.drop_column("conversations", "project_id")
    op.drop_table("tasks")
    op.drop_table("decisions")
    op.drop_table("project_state")
    op.drop_table("projects")
