"""G23 D2: portable JSONL export / staged import (state-copy).

Export = one JSONL per table + manifest.json (alembic head, embedding
identity, per-table counts). This is a STATE-COPY: ids preserved, derived
stores included, **vectors excluded by default** (they are derived and
heavy — the import re-encodes them with the re-embed runner). F10's replay
import is the "relive it" alternative — that path re-processes raw history
through the pipeline instead of copying state; the manifest's ``kind``
field names which one an archive is.

Tables walked from ``Base.metadata.sorted_tables`` (FK-safe order), so new
tables join the export automatically. Excluded: ``maintenance_ledger``
(machine-local scheduler state + the runtime_lease pseudo-row — importing
a foreign lease could stall the target machine's runtime arbitration).
Secrets are never exported: nothing here reads .env or settings values
(the backup module, not this one, snapshots config — and backups are
local-private, exports are shareable).

Import is staged: manifest validation (alembic heads must MATCH — a
state-copy never skips schema versions) → id-preserving inserts into an
empty store by default (``merge=True`` skips existing ids) → re-embed pass
(force=True: inherited stamps describe the SOURCE store) → codex
context_payload regeneration sweep.
"""

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import structlog
from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = structlog.get_logger("ice.memory.portability")

EXPORT_FORMAT = "ice-export-v1"
EXCLUDED_TABLES = {"maintenance_ledger"}


class PortabilityError(RuntimeError):
    pass


def _ser(value):
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_ser(v) for v in value]
    if hasattr(value, "tolist"):          # numpy vector (with_vectors=True)
        return value.tolist()
    return value


def _tables():
    from src.memory.models import Base
    return [t for t in Base.metadata.sorted_tables
            if t.name not in EXCLUDED_TABLES]


def _vector_cols(table) -> set:
    return {c.name for c in table.columns if isinstance(c.type, PgVector)}


def _alembic_head(db) -> str:
    return db.execute(text("SELECT version_num FROM alembic_version")).scalar()


def export_store(db, out_dir, with_vectors: bool = False) -> dict:
    """Write one JSONL per table + manifest.json into *out_dir*. Returns the
    manifest."""
    from src.memory.store_meta import EMBEDDING_KEY, get_meta

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for table in _tables():
        drop_cols = set() if with_vectors else _vector_cols(table)
        n = 0
        with open(out / f"{table.name}.jsonl", "w") as fh:
            result = db.execute(
                table.select().execution_options(yield_per=500))
            for row in result.mappings():
                d = {k: _ser(v) for k, v in row.items() if k not in drop_cols}
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
        counts[table.name] = n

    manifest = {
        "format": EXPORT_FORMAT,
        "kind": "state-copy",
        "note": ("State-copy of an ICE memory store: ids preserved, vectors "
                 + ("included" if with_vectors else
                    "excluded (re-embedded on import)")
                 + ". The F10 replay import is the alternative that re-lives "
                   "raw history through the pipeline instead."),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alembic_head": _alembic_head(db),
        "embedding": get_meta(db, EMBEDDING_KEY),
        "with_vectors": with_vectors,
        "excluded_tables": sorted(EXCLUDED_TABLES),
        "tables": counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("export_complete", out=str(out),
                rows=sum(counts.values()), tables=len(counts))
    return manifest


# ── import ──────────────────────────────────────────────────────────────

def _converter_for(column):
    """Parse JSONL strings back into the column's python type."""
    from sqlalchemy import ARRAY, Date, DateTime
    from sqlalchemy.dialects.postgresql import UUID as PgUUID
    ctype = column.type
    if isinstance(ctype, PgUUID):
        return lambda v: uuid.UUID(v) if v is not None else None
    if isinstance(ctype, DateTime):
        return lambda v: datetime.fromisoformat(v) if v is not None else None
    if isinstance(ctype, Date):
        return lambda v: date.fromisoformat(v) if v is not None else None
    if isinstance(ctype, ARRAY) and isinstance(ctype.item_type, PgUUID):
        return lambda v: [uuid.UUID(x) for x in v] if v is not None else None
    return lambda v: v


def _self_ref_cols(table) -> set:
    """Columns whose FK points back at the same table (episodic_memory.
    parent_message_id): sorted_tables can't order rows WITHIN a table, so
    these are stripped on insert and backfilled in a second pass."""
    cols = set()
    for c in table.columns:
        for fk in c.foreign_keys:
            if fk.column.table is table:
                cols.add(c.name)
    return cols


def import_store(db, in_dir, merge: bool = False, skip_reembed: bool = False,
                 embedder=None) -> dict:
    """Restore an export into this database. Default requires every imported
    table to be empty (id-preserving restore); *merge* skips existing ids
    instead. Ends with the re-embed pass (unless vectors were carried and
    match settings) and the codex context_payload sweep."""
    from src.api.config import settings

    src = Path(in_dir)
    manifest_path = src / "manifest.json"
    if not manifest_path.is_file():
        raise PortabilityError(f"no manifest.json in {src}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != EXPORT_FORMAT:
        raise PortabilityError(
            f"unknown export format {manifest.get('format')!r}")

    target_head = _alembic_head(db)
    if manifest["alembic_head"] != target_head:
        raise PortabilityError(
            f"export was taken at alembic head {manifest['alembic_head']} "
            f"but this database is at {target_head}. Bring the DB to the "
            "export's head (checkout the matching commit + 'uv run alembic "
            "upgrade head'), import there, then upgrade — a state-copy "
            "never skips schema versions.")

    vectors_carried = bool(manifest.get("with_vectors"))
    if vectors_carried:
        exp_dim = (manifest.get("embedding") or {}).get("dim")
        if exp_dim != settings.embedding_dim:
            raise PortabilityError(
                f"export carries {exp_dim}-dim vectors but this store is "
                f"configured for {settings.embedding_dim} — re-export "
                "without vectors, or import into a matching schema.")

    tables = [t for t in _tables() if (src / f"{t.name}.jsonl").is_file()]

    if not merge:
        for table in tables:
            if table.name == "store_meta":
                continue  # a migrated empty DB legitimately carries stamps
            existing = db.execute(text(
                f"SELECT count(*) FROM {table.name}")).scalar()
            if existing:
                raise PortabilityError(
                    f"table {table.name} is not empty ({existing} rows) — "
                    "default import restores into an empty store; pass "
                    "--merge to skip existing ids instead.")

    # rag_chunks may have NOT NULL armed (empty store post-migration);
    # imported rows arrive vectorless, so disarm BEFORE inserting — the
    # re-embed runner's post-step re-arms it once the refill completes.
    rag_file = src / "rag_chunks.jsonl"
    if (not vectors_carried and rag_file.is_file()
            and rag_file.stat().st_size > 0):
        db.execute(text(
            "ALTER TABLE rag_chunks ALTER COLUMN embedding DROP NOT NULL"))
        db.commit()

    report: dict = {"tables": {}}
    for table in tables:
        convs = {c.name: _converter_for(c) for c in table.columns}
        self_ref = _self_ref_cols(table)
        pk_cols = [c.name for c in table.primary_key.columns]
        backfill: list[dict] = []
        inserted = read = 0

        def _flush(batch):
            nonlocal inserted
            if not batch:
                return
            if table.name == "store_meta":
                ins = pg_insert(table)
                stmt = ins.on_conflict_do_update(
                    index_elements=["key"],
                    set_={"value": ins.excluded.value,
                          "updated_at": ins.excluded.updated_at})
            elif merge:
                stmt = pg_insert(table).on_conflict_do_nothing()
            else:
                stmt = table.insert()
            result = db.execute(stmt, batch)
            inserted += result.rowcount if result.rowcount >= 0 else len(batch)

        batch: list[dict] = []
        with open(src / f"{table.name}.jsonl") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                row = {k: convs[k](v) for k, v in raw.items() if k in convs}
                read += 1
                # Pop self-ref FKs from EVERY row (executemany needs
                # homogeneous keys); backfill only the non-NULL ones.
                stripped = {}
                for c in self_ref:
                    val = row.pop(c, None)
                    if val is not None:
                        stripped[c] = val
                if stripped:
                    backfill.append(
                        {**{pk: row[pk] for pk in pk_cols}, **stripped})
                batch.append(row)
                if len(batch) >= 500:
                    _flush(batch)
                    batch = []
        _flush(batch)

        for entry in backfill:
            sets = ", ".join(f"{c} = :{c}" for c in entry if c not in pk_cols)
            where = " AND ".join(f"{pk} = :{pk}" for pk in pk_cols)
            db.execute(text(
                f"UPDATE {table.name} SET {sets} WHERE {where}"), entry)
        db.commit()
        report["tables"][table.name] = {
            "read": read, "inserted": inserted, "skipped": read - inserted}
        logger.info("import_table", table=table.name, read=read,
                    inserted=inserted)

    embedding_matches = (
        vectors_carried
        and (manifest.get("embedding") or {}).get("model") == settings.embedding_model_name
        and (manifest.get("embedding") or {}).get("dim") == settings.embedding_dim)
    if skip_reembed or embedding_matches:
        report["reembed"] = "skipped" if skip_reembed else "vectors carried"
    else:
        from src.memory import reembed
        report["reembed"] = reembed.run(db, embedder=embedder, force=True)

    report["payloads_regenerated"] = _regen_payload_sweep(db)
    logger.info("import_complete", tables=len(tables))
    return report


def _regen_payload_sweep(db) -> int:
    """Rebuild codex context_payloads from the imported graph state (they
    were carried as-is, but links/backlinks are cheap to re-derive and this
    guarantees payload↔graph consistency)."""
    from src.memory.models import CodexEntity
    from src.workers.codex_extractor import _regenerate_context_payload

    n = 0
    entities = db.query(CodexEntity).filter(
        CodexEntity.source == "conversation").all()
    for ent in entities:
        props = ent.properties or {}
        if "merged_into" in props or "deleted_reason" in props:
            continue
        _regenerate_context_payload(ent, db)
        n += 1
        if n % 100 == 0:
            db.commit()
    db.commit()
    return n
