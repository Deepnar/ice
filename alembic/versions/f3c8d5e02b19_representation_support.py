"""v3 independent source-support verdicts for turn representations."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
revision = 'f3c8d5e02b19'
down_revision = 'e2b7c4d91a08'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('episodic_memory', sa.Column('representation_verification', JSONB(), nullable=True))

def downgrade():
    op.drop_column('episodic_memory', 'representation_verification')
