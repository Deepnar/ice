"""G23 D4: the introspective, resumable re-embed runner.

First consumer: C17's 384→1024 un-truncation. The machinery stays for every
future embedder swap (the store_meta stamp makes running it mandatory —
src/memory/store_meta.py), for F10 imports (vectors are excluded from
portable exports and re-encoded here), and for any re-chunk rebuild.

Vector columns are DISCOVERED from the catalog, never hardcoded: a vector
column that exists in the schema without a rule registered below is a HARD
ERROR naming the spec — a future vector-bearing table must register how its
source text is derived before it can ship.

Every rule was verified against its writer (spec §1 rev note 2):
  * episodic_memory   → the user half of raw_text (store_turn embeds ONLY
                        user_message; full raw_text when the "User: …" format
                        is absent — which is exactly right for MCP notes).
  * episodic_chunks   → chunk_text          * rag_chunks → chunk_text
  * codex_entities    → canonical_name — SKIPPING merge husks
                        (properties.merged_into), C10 deletion husks
                        (properties.deleted_reason) and non-conversation
                        sources (code-graph rows carry embedding=NULL by
                        design). Re-encoding a husk would silently resurrect
                        a deleted entity into vector matching.
  * procedural_memory → pattern_description * decisions → decision
  * batch_summaries / conversation_summaries → summary_text
  * context_clusters  → NOT encoded: centroids are recomputed as the
                        normalized mean of member embeddings, AFTER the
                        member table finishes (clustering's own function).

Kill-safe: per-table progress stamps ('reembed:<table>') advance with each
committed batch; a rerun resumes from the stamp. Rows whose source text is
empty keep embedding=NULL and are counted in the report.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import structlog
from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy import bindparam, text

from src.memory.store_meta import (
    REEMBED_PREFIX,
    discover_vector_columns,
    get_meta,
    set_meta,
    stamp_embedding,
)

logger = structlog.get_logger("ice.memory.reembed")

SPEC = "docs/specs/G23_C17_data_longevity.md"
PER_ROW_MS = 20  # CPU estimate for the up-front print only

_RAG_NOT_NULL_RESTORE = """
    DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM rag_chunks WHERE embedding IS NULL)
        THEN
            ALTER TABLE rag_chunks ALTER COLUMN embedding SET NOT NULL;
        END IF;
    END $$
"""


def _episodic_source(raw: str) -> str:
    raw = raw or ""
    if raw.startswith("User: "):
        head, sep, _ = raw.partition("\n\nAssistant:")
        if sep:
            return head[len("User: "):].strip()
    return raw.strip()


@dataclass(frozen=True)
class TableRule:
    kind: str                              # "text" | "centroid"
    select_sql: str = ""                   # must yield columns (id, src)
    id_col: str = "id"
    transform: Optional[Callable[[str], str]] = None
    post_sql: Optional[str] = None         # once, after the table completes


# Insertion order is execution order; context_clusters MUST stay last (its
# centroids are means over episodic_memory's fresh embeddings).
RULES: dict[str, TableRule] = {
    "episodic_memory": TableRule(
        kind="text",
        select_sql="SELECT id, raw_text AS src FROM episodic_memory",
        transform=_episodic_source),
    "episodic_chunks": TableRule(
        kind="text",
        select_sql="SELECT id, chunk_text AS src FROM episodic_chunks"),
    "codex_entities": TableRule(
        kind="text",
        select_sql="""SELECT id, canonical_name AS src FROM codex_entities
            WHERE source = 'conversation'
              AND NOT (properties ? 'merged_into')
              AND NOT (properties ? 'deleted_reason')"""),
    "procedural_memory": TableRule(
        kind="text",
        select_sql="SELECT id, pattern_description AS src FROM procedural_memory"),
    "decisions": TableRule(
        kind="text",
        select_sql="SELECT id, decision AS src FROM decisions"),
    "rag_chunks": TableRule(
        kind="text",
        select_sql="SELECT id, chunk_text AS src FROM rag_chunks",
        post_sql=_RAG_NOT_NULL_RESTORE),
    "batch_summaries": TableRule(
        kind="text",
        select_sql="SELECT id, summary_text AS src FROM batch_summaries"),
    "conversation_summaries": TableRule(
        kind="text", id_col="conversation_id",
        select_sql=("SELECT conversation_id AS id, summary_text AS src "
                    "FROM conversation_summaries")),
    "context_clusters": TableRule(kind="centroid"),
}


class UnregisteredVectorColumn(RuntimeError):
    """A vector column exists in the schema with no source-text rule."""


def _validate_registry(db, requested: Optional[set]) -> list[str]:
    """Cross-check catalog vs registry; return the ordered target tables
    (registry order, filtered to what exists in this schema)."""
    cols_by_table: dict[str, list[str]] = {}
    for table, column, _dim in discover_vector_columns(db):
        cols_by_table.setdefault(table, []).append(column)

    unregistered = sorted(set(cols_by_table) - set(RULES))
    multi = sorted(t for t, cs in cols_by_table.items() if len(cs) > 1)
    if unregistered or multi:
        raise UnregisteredVectorColumn(
            f"vector columns without a registered source-text rule: "
            f"unregistered tables={unregistered or 'none'}, "
            f"multi-vector tables={multi or 'none'}. Register a rule in "
            f"src/memory/reembed.py (see {SPEC}) before re-embedding.")

    targets = [t for t in RULES if t in cols_by_table]
    if requested is not None:
        unknown = requested - set(RULES)
        if unknown:
            raise ValueError(f"unknown tables requested: {sorted(unknown)}")
        targets = [t for t in targets if t in requested]
    return targets


def _count_eligible(db, rule: TableRule, table: str) -> int:
    if rule.kind == "centroid":
        return db.execute(text(f"SELECT count(*) FROM {table}")).scalar() or 0
    return db.execute(text(
        f"SELECT count(*) FROM ({rule.select_sql}) s")).scalar() or 0


def _run_text_table(db, embedder, table: str, rule: TableRule,
                    batch_size: int, stamp: dict) -> dict:
    update = text(
        f"UPDATE {table} SET embedding = :emb WHERE {rule.id_col} = :id"
    ).bindparams(bindparam("emb", type_=PgVector))
    done = int(stamp.get("rows_done", 0))
    empty = int(stamp.get("empty_source", 0))
    last_id = stamp.get("last_id")

    while True:
        if last_id is None:
            rows = db.execute(text(
                f"SELECT * FROM ({rule.select_sql}) s "
                f"ORDER BY s.id LIMIT :lim"), {"lim": batch_size}).fetchall()
        else:
            rows = db.execute(text(
                f"SELECT * FROM ({rule.select_sql}) s WHERE s.id > :last "
                f"ORDER BY s.id LIMIT :lim"),
                {"last": last_id, "lim": batch_size}).fetchall()
        if not rows:
            break

        texts, targets = [], []
        for row in rows:
            src = rule.transform(row.src) if rule.transform else (row.src or "").strip()
            if src:
                texts.append(src)
                targets.append(row.id)
            else:
                empty += 1
        if texts:
            vecs = embedder.encode(texts, convert_to_tensor=False,
                                   show_progress_bar=False)
            for row_id, vec in zip(targets, vecs):
                emb = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                db.execute(update, {"emb": emb, "id": row_id})
            done += len(texts)

        last_id = str(rows[-1].id)
        set_meta(db, REEMBED_PREFIX + table, {
            "status": "in_progress", "last_id": last_id,
            "rows_done": done, "empty_source": empty})
        db.commit()
        logger.info("reembed_batch", table=table, rows_done=done)

    return {"rows": done, "empty_source": empty}


def _run_centroids(db, table: str) -> dict:
    # Local import: clustering materializes the shared embedder at import
    # (already loaded by the time we get here).
    from src.workers.clustering import _recompute_centroid_from_members
    update = text(
        f"UPDATE {table} SET embedding = :emb WHERE id = :id"
    ).bindparams(bindparam("emb", type_=PgVector))
    ids = [r.id for r in db.execute(
        text(f"SELECT id FROM {table} ORDER BY id")).fetchall()]
    done = memberless = 0
    for i, cid in enumerate(ids):
        centroid = _recompute_centroid_from_members(db, cid)
        if centroid is None:
            db.execute(text(
                f"UPDATE {table} SET embedding = NULL WHERE id = :id"),
                {"id": cid})
            memberless += 1
        else:
            db.execute(update, {"emb": centroid, "id": cid})
            done += 1
        if (i + 1) % 50 == 0:
            db.commit()
    db.commit()
    return {"rows": done, "memberless": memberless}


def run(db, embedder=None, tables: Optional[str] = "all",
        batch_size: int = 256, force: bool = False) -> dict:
    """Re-encode every registered vector column whose stamp isn't 'done'.

    *tables*: "all" or a comma-separated subset. *force* restarts tables
    from scratch, ignoring existing stamps (an import or a same-dim embedder
    swap uses this — the stamps it inherits describe the SOURCE store).
    Returns a per-table report. Safe to kill at any point; rerun resumes.
    """
    requested = None if tables in (None, "all") else {
        t.strip() for t in tables.split(",") if t.strip()}
    targets = _validate_registry(db, requested)

    total = sum(_count_eligible(db, RULES[t], t) for t in targets)
    est_min = total * PER_ROW_MS / 1000 / 60
    msg = (f"re-embed: {total} rows across {len(targets)} tables, "
           f"~{est_min:.1f} min CPU estimate (resumable — safe to kill)")
    logger.info("reembed_start", rows=total, tables=targets)
    print(msg)

    if embedder is None:
        from src.memory.embedder import get_embedder
        embedder = get_embedder()

    report: dict = {}
    for table in targets:
        rule = RULES[table]
        stamp = (get_meta(db, REEMBED_PREFIX + table) or {}) if not force else {}
        if stamp.get("status") == "done":
            report[table] = {"skipped": "already done"}
            continue
        if rule.kind == "centroid":
            result = _run_centroids(db, table)
        else:
            result = _run_text_table(db, embedder, table, rule,
                                     batch_size, stamp)
            if rule.post_sql:
                db.execute(text(rule.post_sql))
        set_meta(db, REEMBED_PREFIX + table,
                 {"status": "done", **{k: v for k, v in result.items()}})
        db.commit()
        report[table] = result
        logger.info("reembed_table_done", table=table, **result)

    # Completion re-stamp: the store now matches settings' identity.
    stamp_embedding(db)
    db.commit()
    logger.info("reembed_complete", report=report)
    print(f"re-embed complete: {report}")
    return report
