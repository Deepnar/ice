"""documents.source_text — a pasted document has no file to re-read

Revision ID: 7c4d19ab35e2
Revises: 5fe5ad26480b
Create Date: 2026-07-28 23:52:04.118307

One nullable column. Every existing row has a source_path, so NULL here means
"read the file", which is what the code did before this existed.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '7c4d19ab35e2'
down_revision: Union[str, Sequence[str], None] = '5fe5ad26480b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('documents',
                  sa.Column('source_text', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('documents', 'source_text')
