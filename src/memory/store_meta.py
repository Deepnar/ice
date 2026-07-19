"""G23 D1: store-level embedding identity + the fail-loud startup guard.

``store_meta`` is a one-row-per-key JSONB table. Key ``embedding`` holds the
store's embedding identity ``{model, dim, stamped_at}``; keys
``reembed:<table>`` hold the re-embed runner's per-table progress stamps
(kill-safe resume — src/memory/reembed.py).

The guard has ONE home: ``create_core()`` calls ``check_embedding_stamp()``
on every boot path (app lifespan and headless ice-mcp both pass through the
factory). Mismatch ⇒ refuse to boot naming the exact commands. A stamp that
matches but has pending re-embed tables boots WITH a loud warning — the
vector legs already filter ``embedding IS NOT NULL``, so a half-filled store
degrades recall, never crashes.
"""

import json
from datetime import datetime, timezone

import structlog
from sqlalchemy import text

logger = structlog.get_logger("ice.memory.store_meta")

EMBEDDING_KEY = "embedding"
REEMBED_PREFIX = "reembed:"
MIGRATE_CMD = "uv run alembic upgrade head"
REEMBED_CMD = "uv run python scripts/ice_reembed.py"


class EmbeddingStampMismatch(RuntimeError):
    """settings' embedder identity disagrees with what the store carries —
    booting would silently cosine-compare wrong vectors."""


def get_meta(db, key: str):
    row = db.execute(text("SELECT value FROM store_meta WHERE key = :k"),
                     {"k": key}).first()
    return row.value if row else None


def set_meta(db, key: str, value: dict) -> None:
    """Upsert one key. Does NOT commit — the caller owns the transaction
    (the re-embed runner batches a stamp update with its row updates)."""
    db.execute(text("""
        INSERT INTO store_meta (key, value, updated_at)
        VALUES (:k, CAST(:v AS jsonb), :ts)
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
    """), {"k": key, "v": json.dumps(value),
           "ts": datetime.now(timezone.utc)})


def discover_vector_columns(db) -> list[tuple[str, str, int]]:
    """Every pgvector column in the public schema, from the catalog — never
    a hardcoded list (D4). Returns [(table, column, declared_dim)]."""
    rows = db.execute(text("""
        SELECT c.relname AS table_name, a.attname AS column_name,
               a.atttypmod AS dim
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_type t ON t.oid = a.atttypid
        WHERE t.typname = 'vector' AND NOT a.attisdropped
          AND c.relkind = 'r'
          AND c.relnamespace = 'public'::regnamespace
        ORDER BY c.relname
    """)).fetchall()
    return [(r.table_name, r.column_name, r.dim) for r in rows]


def _any_embeddings_present(db) -> bool:
    for table, column, _dim in discover_vector_columns(db):
        hit = db.execute(text(
            f"SELECT 1 FROM {table} WHERE {column} IS NOT NULL LIMIT 1"
        )).first()
        if hit:
            return True
    return False


def stamp_embedding(db) -> dict:
    """Stamp the store with settings' embedding identity. Caller commits."""
    from src.api.config import settings
    stamp = {"model": settings.embedding_model_name,
             "dim": settings.embedding_dim,
             "stamped_at": datetime.now(timezone.utc).isoformat()}
    set_meta(db, EMBEDDING_KEY, stamp)
    return stamp


def pending_reembeds(db) -> list[str]:
    rows = db.execute(text(
        "SELECT key, value FROM store_meta WHERE key LIKE :p"),
        {"p": REEMBED_PREFIX + "%"}).fetchall()
    return sorted(r.key[len(REEMBED_PREFIX):] for r in rows
                  if (r.value or {}).get("status") != "done")


def check_embedding_stamp(db) -> dict:
    """The D1 guard. Raises EmbeddingStampMismatch when settings disagree
    with the store; bootstraps a stamp only when there is provably nothing
    to be wrong about (no store_meta row AND zero stored embeddings — fresh
    create_all databases); otherwise refuses unknown-provenance stores."""
    from src.api.config import settings
    stamp = get_meta(db, EMBEDDING_KEY)

    if stamp is None:
        if _any_embeddings_present(db):
            raise EmbeddingStampMismatch(
                "store_meta has no 'embedding' stamp but the store contains "
                "embeddings of unknown provenance. Refusing to guess. Run "
                f"'{MIGRATE_CMD}' (seeds the stamp), then '{REEMBED_CMD}' "
                "if the vectors predate the configured embedder.")
        stamp = stamp_embedding(db)
        db.commit()
        logger.info("embedding_stamp_bootstrapped", **stamp)
        return stamp

    if (stamp.get("model") != settings.embedding_model_name
            or stamp.get("dim") != settings.embedding_dim):
        raise EmbeddingStampMismatch(
            f"store carries embeddings for {stamp.get('model')}@"
            f"{stamp.get('dim')} but settings say "
            f"{settings.embedding_model_name}@{settings.embedding_dim}. "
            f"Vector search would be silently wrong. Run '{MIGRATE_CMD}' "
            f"then '{REEMBED_CMD}' to re-encode the store (resumable), or "
            "revert the settings change.")

    pending = pending_reembeds(db)
    if pending:
        logger.warning(
            "reembed_in_progress", tables=pending,
            note=("vector recall is degraded until the re-embed finishes — "
                  f"resume with '{REEMBED_CMD}'"))
    return stamp
