"""Preserve source representation and identity through cold storage."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = 'c8f60224d395'
down_revision = 'b7e5f013c284'
branch_labels = None
depends_on = None

COLUMNS = (
    ('summary_coverage', sa.Float()),
    ('representation_verification', JSONB()),
    ('abstract_text', sa.Text()),
    ('lossless_flag', sa.Boolean()),
    ('inject_raw', sa.Boolean()),
    ('session_id', UUID(as_uuid=True)),
    ('intent_tags', ARRAY(sa.Text())),
    ('context_reliance', sa.Text()),
    ('idempotency_key', sa.Text()),
)


def upgrade():
    for name, kind in COLUMNS:
        op.add_column('cold_storage', sa.Column(name, kind, nullable=True))


def downgrade():
    for name, _ in reversed(COLUMNS):
        op.drop_column('cold_storage', name)
