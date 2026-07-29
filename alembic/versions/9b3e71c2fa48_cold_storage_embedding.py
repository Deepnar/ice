"""cold_storage.embedding — the one archive table C17 never gave a vector

Revision ID: 9b3e71c2fa48
Revises: 7c4d19ab35e2
Create Date: 2026-07-29 17:40:02.114523

C17 moved nine vector columns to 1024 and `cold_storage` was not among them —
it had none. So the T3 cold leg has always retrieved by `raw_text ILIKE ANY(%kw%)`
against a hardcoded 40-word English stoplist, the last purely lexical query in
retrieval, and the one place C16's coverage selection could not reach.

The table holds ZERO rows right now, so this is a column add with no backfill.
Doing it later means doing it as a backfill.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '9b3e71c2fa48'
down_revision: Union[str, Sequence[str], None] = '7c4d19ab35e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('cold_storage', sa.Column('embedding', Vector(1024),
                                            nullable=True))
    # G23 invariant: every vector column carries a `reembed:<table>` stamp in
    # store_meta. C17 seeded one per column it created and this one owes the
    # same — without it an export/import round trip comes back with one MORE
    # store_meta row than it left with, because the importer's re-embed pass
    # creates the missing stamp. 'done' is correct: the table is empty, so
    # there is nothing to re-embed.
    op.execute("""
        INSERT INTO store_meta (key, value, updated_at)
        VALUES ('reembed:cold_storage',
                jsonb_build_object('status', 'done',
                                   'stamped_at', now()::text),
                now())
        ON CONFLICT (key) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM store_meta WHERE key = 'reembed:cold_storage'")
    op.drop_column('cold_storage', 'embedding')
