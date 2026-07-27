"""G23/C17 behavioral test — specs/G23_C17_data_longevity.md §4 checks 1–8.

1  export→import round-trip (counts, ids, manifest, no secrets)
2  re-embed on the imported store (dims 1024, husk/static stay NULL,
   centroid = normalized member mean — recomputed, not encoded)
3  retrieval parity 384 vs 1024 (top-3 overlap ≥2/3, known row top-1)
4  startup guard refuses a doctored stamp, names the commands
5  unregistered vector column ⇒ runner hard-errors naming the spec
6  kill-safe resume (per-table stamps honored, done tables skipped)
7  slice384 MRL equivalence: bit-identical to truncate_dim=384 output;
   classifier end-to-end sanity on the live checkpoint
8  backup archive restores into a SCRATCH database (never ice_db)

House pattern: live Postgres, uniquely-marked fixture rows, cleanup in
`finally` — NEVER truncates. Unlike the stub-embedder suites this one uses
the REAL embedder (equivalence/parity are the point): expect two model
loads and a few minutes of runtime. Scratch databases
(ice_longevity_scratch / ice_restore_scratch) are created and dropped by
the test itself.

Run: uv run python tests/test_longevity.py
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.orm import sessionmaker

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.embedder import get_embedder, slice384
from src.memory.models import (
    Base,
    BatchSummary,
    CodexEdge,
    CodexEntity,
    ContextCluster,
    Conversation,
    ConversationSummary,
    Decision,
    EpisodicChunk,
    EpisodicClusterLink,
    EpisodicMemory,
    IdempotencyKey,
    MemorySlot,
    ProceduralMemory,
    Project,
    RAGChunk,
    RAGDocument,
)
from src.memory import reembed
from src.memory.backup import run_backup
from src.memory.portability import export_store, import_store
from src.memory.store_meta import (
    EMBEDDING_KEY,
    REEMBED_PREFIX,
    EmbeddingStampMismatch,
    check_embedding_stamp,
    get_meta,
    set_meta,
    stamp_embedding,
)

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


HEX = uuid.uuid4().hex[:6]
MARK = "longevmark" + HEX.translate(str.maketrans("0123456789", "abcdefghij"))
SCRATCH_DB = "ice_longevity_scratch"
RESTORE_DB = "ice_restore_scratch"
COMPOSE = ["docker", "compose", "-f", "docker/docker-compose.yml"]

_base_url = settings.database_url.rsplit("/", 1)[0]
ADMIN_URL = _base_url + "/postgres"
SCRATCH_URL = _base_url + f"/{SCRATCH_DB}"


def _admin_exec(sql: str) -> None:
    eng = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    try:
        with eng.connect() as c:
            c.execute(text(sql))
    finally:
        eng.dispose()


def _normalize(vec):
    n = float(np.linalg.norm(vec))
    return [v / n for v in vec] if n > 0 else list(vec)


db = SessionLocal()
embedder = get_embedder()

# fixture id trackers (live DB)
conv_id = None
turn_ids: list = []
chunk_ids: list = []
cluster_id = None
entity_ids: list = []
edge_ids: list = []
pattern_id = None
project_id = None
decision_id = None
rag_doc_id = None
rag_chunk_ids: list = []
bs_id = None
slot_id = None
ikey = f"longevity-{HEX}"

scratch_engine = None
tmp_root = Path(tempfile.mkdtemp(prefix="ice_longevity_"))
export_dir = tmp_root / "export"
backup_archive = None

FIX_TEXTS = [
    "the postgres index rebuild fixed the slow database queries",
    "grandmothers lasagna recipe needs fresh basil and oven patience",
    "the marathon training plan added hill sprints on tuesdays",
    "quantum error correction codes protect fragile qubit states",
    "the watercolor workshop taught wet on wet blending techniques",
]
QUERY = "why were the database queries so slow before the index change"

try:
    now = datetime.now(timezone.utc)

    # ═══ seed fixtures in the live store (tracked, deleted in finally) ════
    conv = Conversation(memory_scope_type="auto")
    db.add(conv)
    db.commit()
    conv_id = conv.id

    for i, t in enumerate(FIX_TEXTS):
        user_part = f"{MARK} {t}"
        emb = embedder.encode(user_part, convert_to_tensor=False).tolist()
        turn = EpisodicMemory(
            conversation_id=conv_id, batch_id=uuid.uuid4(), timestamp=now,
            context_reliance="Long_Term_Memory",
            raw_text=f"User: {user_part}\n\nAssistant: ok",
            embedding=emb, decay_score=1.0,
            idempotency_key=f"longevity-{HEX}-{i}")
        db.add(turn)
        db.flush()
        turn_ids.append(turn.id)
    db.commit()

    for j in range(2):
        ch = EpisodicChunk(turn_id=turn_ids[0], chunk_index=j,
                           chunk_text=f"{MARK} chunk {FIX_TEXTS[j]}",
                           embedding=embedder.encode(
                               f"{MARK} chunk {FIX_TEXTS[j]}",
                               convert_to_tensor=False).tolist())
        db.add(ch)
        db.flush()
        chunk_ids.append(ch.id)

    member_embs = [db.get(EpisodicMemory, turn_ids[k]).embedding
                   for k in (0, 1)]
    centroid = _normalize(np.mean([list(e) for e in member_embs], axis=0))
    cl = ContextCluster(name=f"{MARK} cluster", conversation_id=conv_id,
                        embedding=centroid)
    db.add(cl)
    db.flush()
    cluster_id = cl.id
    for k in (0, 1):
        db.add(EpisodicClusterLink(episodic_id=turn_ids[k],
                                   cluster_id=cluster_id))
    db.commit()

    ent_a = CodexEntity(canonical_name=f"zorbelquat{HEX}",
                        embedding=embedder.encode(
                            f"zorbelquat{HEX}",
                            convert_to_tensor=False).tolist())
    ent_husk = CodexEntity(canonical_name=f"morvexled{HEX}", embedding=None,
                           properties={"deleted_reason": "test_husk"})
    ent_static = CodexEntity(canonical_name=f"statquex{HEX}", embedding=None,
                             source="static_analysis")
    ent_d = CodexEntity(canonical_name=f"farnowick{HEX}",
                        embedding=embedder.encode(
                            f"farnowick{HEX}",
                            convert_to_tensor=False).tolist())
    db.add_all([ent_a, ent_husk, ent_static, ent_d])
    db.flush()
    entity_ids = [ent_a.id, ent_husk.id, ent_static.id, ent_d.id]
    edge = CodexEdge(source_id=ent_a.id, target_id=ent_d.id,
                     relation="works_with", source_batch=uuid.uuid4())
    db.add(edge)
    db.flush()
    edge_ids.append(edge.id)

    pat_text = f"{MARK} when debugging the user pastes the traceback first"
    pat = ProceduralMemory(pattern_name=pat_text[:80],
                           pattern_description=pat_text,
                           embedding=embedder.encode(
                               pat_text, convert_to_tensor=False).tolist())
    db.add(pat)
    db.flush()
    pattern_id = pat.id

    proj = Project(name=f"zzlongevity{HEX}", slug=f"zzlongevity-{HEX}",
                   roots=["/tmp/zzlongevity"])
    db.add(proj)
    db.flush()
    project_id = proj.id
    dec_text = f"{MARK} use alembic for every schema change"
    dec = Decision(project_id=project_id, decision=dec_text,
                   embedding=embedder.encode(
                       dec_text, convert_to_tensor=False).tolist())
    db.add(dec)
    db.flush()
    decision_id = dec.id

    doc = RAGDocument(filename=f"{MARK}.md", file_type="md")
    db.add(doc)
    db.flush()
    rag_doc_id = doc.id
    for j in range(2):
        rc = RAGChunk(document_id=rag_doc_id, chunk_index=j,
                      chunk_text=f"{MARK} rag {FIX_TEXTS[j]}",
                      embedding=embedder.encode(
                          f"{MARK} rag {FIX_TEXTS[j]}",
                          convert_to_tensor=False).tolist())
        db.add(rc)
        db.flush()
        rag_chunk_ids.append(rc.id)

    bs = BatchSummary(conversation_id=conv_id, start_turn_index=0,
                      end_turn_index=1,
                      summary_text=f"{MARK} summary of database work",
                      embedding=embedder.encode(
                          f"{MARK} summary of database work",
                          convert_to_tensor=False).tolist())
    db.add(bs)
    db.flush()
    bs_id = bs.id
    db.add(ConversationSummary(
        conversation_id=conv_id,
        summary_text=f"{MARK} conversation about databases",
        embedding=embedder.encode(
            f"{MARK} conversation about databases",
            convert_to_tensor=False).tolist()))
    slot = MemorySlot(slot_name="user_profile", content=f"{MARK} profile",
                      updated_by="system")
    db.add(slot)
    db.flush()
    slot_id = slot.id
    db.add(IdempotencyKey(key=ikey))
    db.commit()

    # ═══ check 1: export → import round-trip ═════════════════════════════
    print("── check 1: export→import round-trip ──")
    manifest = export_store(db, export_dir)
    live_counts = {t: db.execute(text(f"SELECT count(*) FROM {t}")).scalar()
                   for t in manifest["tables"]}
    check("manifest head matches live DB",
          manifest["alembic_head"] == db.execute(
              text("SELECT version_num FROM alembic_version")).scalar())
    check("per-table export counts match live",
          manifest["tables"] == live_counts)
    check("maintenance_ledger excluded",
          "maintenance_ledger" not in manifest["tables"]
          and not (export_dir / "maintenance_ledger.jsonl").exists())

    secret_hit = False
    password = settings.database_url.split(":")[2].split("@")[0]
    for f in export_dir.iterdir():
        if password and password in f.read_text():
            secret_hit = True
    check("no secrets in export (db password grep)", not secret_hit)

    _admin_exec(f"DROP DATABASE IF EXISTS {SCRATCH_DB} WITH (FORCE)")
    _admin_exec(f"CREATE DATABASE {SCRATCH_DB}")
    scratch_engine = create_engine(SCRATCH_URL)
    with scratch_engine.begin() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(scratch_engine)
    with scratch_engine.begin() as c:
        c.execute(text("CREATE TABLE alembic_version "
                       "(version_num varchar(32) NOT NULL PRIMARY KEY)"))
        c.execute(text("INSERT INTO alembic_version VALUES (:v)"),
                  {"v": manifest["alembic_head"]})
    ScratchSession = sessionmaker(bind=scratch_engine)

    sdb = ScratchSession()
    report = import_store(sdb, export_dir)   # runs re-embed + payload sweep
    scratch_counts = {t: sdb.execute(
        text(f"SELECT count(*) FROM {t}")).scalar()
        for t in manifest["tables"]}
    check("imported counts equal exported counts",
          scratch_counts == manifest["tables"])
    check("episodic ids preserved",
          sdb.execute(text(
              "SELECT count(*) FROM episodic_memory WHERE id = ANY(:ids)"),
              {"ids": turn_ids}).scalar() == len(turn_ids))

    # ═══ check 2: re-embed on the imported store ═════════════════════════
    print("── check 2: re-embed coverage on the imported store ──")
    dims = set()
    for t in ("episodic_memory", "episodic_chunks", "codex_entities",
              "procedural_memory", "decisions", "rag_chunks",
              "batch_summaries", "conversation_summaries",
              "context_clusters"):
        dims |= {r[0] for r in sdb.execute(text(
            f"SELECT DISTINCT vector_dims(embedding) FROM {t} "
            f"WHERE embedding IS NOT NULL")).fetchall()}
    check("all scratch embeddings are 1024-dim", dims == {1024})
    check("no unexpectedly-NULL embeddings",
          sdb.execute(text(
              "SELECT count(*) FROM episodic_memory WHERE embedding IS NULL"
          )).scalar() == 0
          and sdb.execute(text(
              "SELECT count(*) FROM rag_chunks WHERE embedding IS NULL"
          )).scalar() == 0)
    husk_emb, static_emb = sdb.execute(text(
        "SELECT (SELECT embedding FROM codex_entities WHERE id = :h), "
        "       (SELECT embedding FROM codex_entities WHERE id = :s)"),
        {"h": entity_ids[1], "s": entity_ids[2]}).first()
    check("husk + static entities stay NULL (skip rule)",
          husk_emb is None and static_emb is None)

    s_members = sdb.execute(text(
        "SELECT e.embedding FROM episodic_memory e "
        "JOIN episodic_cluster_links l ON l.episodic_id = e.id "
        "WHERE l.cluster_id = :c"), {"c": cluster_id}).fetchall()
    expected = _normalize(np.mean(
        [[float(x) for x in str(r[0]).strip("[]").split(",")]
         for r in s_members], axis=0))
    actual_raw = sdb.execute(text(
        "SELECT embedding FROM context_clusters WHERE id = :c"),
        {"c": cluster_id}).scalar()
    actual = [float(x) for x in str(actual_raw).strip("[]").split(",")]
    check("centroid == normalized member mean (recomputed, not encoded)",
          np.allclose(expected, actual, atol=1e-4))
    check("rag NOT NULL re-armed after refill",
          sdb.execute(text(
              "SELECT is_nullable FROM information_schema.columns WHERE "
              "table_name='rag_chunks' AND column_name='embedding'"
          )).scalar() == "NO")

    # ═══ check 3: retrieval parity 384 vs 1024 ═══════════════════════════
    print("── check 3: retrieval parity (384 reference vs native 1024) ──")
    from sentence_transformers import SentenceTransformer
    ref384 = SentenceTransformer(settings.embedding_model_name,
                                 device="cpu", truncate_dim=384)
    fix_texts_marked = [f"{MARK} {t}" for t in FIX_TEXTS]
    ref_docs = ref384.encode(fix_texts_marked, convert_to_tensor=False)
    ref_q = ref384.encode(QUERY, convert_to_tensor=False)

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    ref_rank = sorted(range(5), key=lambda i: -_cos(ref_q, ref_docs[i]))
    ref_top3 = {turn_ids[i] for i in ref_rank[:3]}

    q1024 = embedder.encode(QUERY, convert_to_tensor=False).tolist()
    rows = db.execute(text(
        "SELECT id FROM episodic_memory WHERE conversation_id = :c "
        "ORDER BY embedding <=> :q LIMIT 3"
    ).bindparams(bindparam("q", type_=PgVector)),
        {"c": conv_id, "q": q1024}).fetchall()
    native_top3 = [r.id for r in rows]
    overlap = len(ref_top3 & set(native_top3))
    check(f"top-3 overlap >= 2/3 (got {overlap}/3)", overlap >= 2)
    check("known-correct row stays top-1",
          native_top3 and native_top3[0] == turn_ids[0]
          and ref_rank[0] == 0)

    # ═══ check 4: startup guard ══════════════════════════════════════════
    print("── check 4: startup guard (doctored stamp refuses) ──")
    check("guard passes on the live store",
          check_embedding_stamp(db).get("dim") == settings.embedding_dim)
    doctored = dict(get_meta(sdb, EMBEDDING_KEY) or {})
    doctored["dim"] = 999
    set_meta(sdb, EMBEDDING_KEY, doctored)
    sdb.commit()
    refused, msg = False, ""
    try:
        check_embedding_stamp(sdb)
    except EmbeddingStampMismatch as e:
        refused, msg = True, str(e)
    check("doctored stamp refused", refused)
    check("error names the migration + re-embed commands",
          "alembic upgrade head" in msg and "ice_reembed" in msg)
    stamp_embedding(sdb)
    sdb.commit()

    # ═══ check 5: unregistered vector column ⇒ hard error ════════════════
    print("── check 5: unregistered vector column hard-errors ──")
    sdb.execute(text("CREATE TABLE zz_scratch_vectors "
                     "(id uuid PRIMARY KEY, embedding vector(1024))"))
    sdb.commit()
    blew, blew_msg = False, ""
    try:
        reembed.run(sdb, embedder=embedder)
    except reembed.UnregisteredVectorColumn as e:
        blew, blew_msg = True, str(e)
    check("runner hard-errors on the unregistered column", blew)
    check("error names the spec", "G23_C17" in blew_msg)
    sdb.execute(text("DROP TABLE zz_scratch_vectors"))
    sdb.commit()

    # ═══ check 6: kill-safe resume ═══════════════════════════════════════
    print("── check 6: resume from per-table stamps ──")
    sorted_ids = [r.id for r in sdb.execute(text(
        "SELECT id FROM episodic_memory ORDER BY id")).fetchall()]
    sdb.execute(text("UPDATE episodic_memory SET embedding = NULL"))
    set_meta(sdb, REEMBED_PREFIX + "episodic_memory",
             {"status": "in_progress", "last_id": str(sorted_ids[1]),
              "rows_done": 2})
    sdb.commit()
    reembed.run(sdb, embedder=embedder, tables="episodic_memory")
    before_null = sdb.execute(text(
        "SELECT count(*) FROM episodic_memory "
        "WHERE id = ANY(:ids) AND embedding IS NULL"),
        {"ids": sorted_ids[:2]}).scalar()
    after_filled = sdb.execute(text(
        "SELECT count(*) FROM episodic_memory "
        "WHERE id = ANY(:ids) AND embedding IS NOT NULL"),
        {"ids": sorted_ids[2:]}).scalar()
    check("rows before the stamp untouched (resume honored)",
          before_null == 2)
    check("rows after the stamp completed", after_filled == len(sorted_ids) - 2)
    rerun = reembed.run(sdb, embedder=embedder, tables="codex_entities")
    check("done tables skipped on rerun",
          rerun.get("codex_entities", {}).get("skipped") == "already done")

    # ═══ check 7: slice384 MRL equivalence + classifier ══════════════════
    print("── check 7: slice384 equivalence (MRL proof) + classifier ──")
    probes = [QUERY, f"{MARK} short", FIX_TEXTS[3]]
    exact = True
    for p in probes:
        # .float(): encode returns bf16 tensors; the classifier itself
        # floats them the same way before the MLP.
        ref_vec = ref384.encode(p, convert_to_tensor=True).float()
        new_vec = slice384(embedder.encode(p, convert_to_tensor=True)).float()
        exact = exact and bool(
            np.array_equal(ref_vec.numpy(), new_vec.numpy()))
    check("slice384(native) == truncate_dim output, bit-identical", exact)

    from src.classifier.classifier import PyTorchClassifier
    clf = PyTorchClassifier(model_path=settings.classifier_model_path,
                            schema_path=settings.label_schema_path)
    result = clf._run_ml_classifier(
        "what did we decide about the postgres migration last week")
    # Generation-agnostic on purpose. The bit-identity proof above is the C17
    # claim and it stands whatever is installed; THIS check only says the live
    # classifier runs end-to-end at its own declared width. Since B1's promotion
    # (2026-07-27) the live checkpoint is v2 and reads the native 1024, so the
    # slice is no longer on the classifier's path at all — that is A9's retirement
    # arriving, not a regression. The micro-NER keeps consuming slice384, which is
    # why the equivalence proof above still matters.
    check("classifier runs end-to-end at its checkpoint's width",
          len(result.raw_probs) == clf.active_schema.total_width
          and clf.input_dim in (384, 1024)
          and result.context_reliance
          in ("Zero_Shot", "Long_Term_Memory", "Real_Time_Search"))
    check("classifier shares the process embedder (G13)",
          clf.embedder is embedder)

    # ═══ check 8: backup restores into a scratch DB ══════════════════════
    print("── check 8: backup restore into scratch DB (never ice_db) ──")
    backup_archive = run_backup(out_dir=tmp_root / "backups",
                                tag="longevity-test")
    check("archive exists and is plausible",
          backup_archive.is_file() and backup_archive.stat().st_size > 4096)
    with tarfile.open(backup_archive) as tar:
        tar.extractall(tmp_root / "restore_stage", filter="data")
    dump = next((tmp_root / "restore_stage").glob("*/ice_db.dump"))

    _admin_exec(f"DROP DATABASE IF EXISTS {RESTORE_DB} WITH (FORCE)")
    _admin_exec(f"CREATE DATABASE {RESTORE_DB}")
    with open(dump, "rb") as fh:
        subprocess.run(COMPOSE + ["exec", "-T", "postgres", "pg_restore",
                                  "-U", "ice", "-d", RESTORE_DB,
                                  "--no-owner"],
                       stdin=fh, check=True, capture_output=True)
    restore_eng = create_engine(_base_url + f"/{RESTORE_DB}")
    try:
        with restore_eng.connect() as c:
            restored = c.execute(text(
                "SELECT count(*) FROM episodic_memory")).scalar()
            live_now = db.execute(text(
                "SELECT count(*) FROM episodic_memory")).scalar()
            check("restored row count matches live at backup time",
                  restored == live_now)
    finally:
        restore_eng.dispose()

    sdb.close()

finally:
    try:
        db.rollback()
        if edge_ids:
            db.execute(text("DELETE FROM codex_edges WHERE id = ANY(:ids)"),
                       {"ids": edge_ids})
        if entity_ids:
            db.execute(text("DELETE FROM codex_entities WHERE id = ANY(:ids)"),
                       {"ids": entity_ids})
        if chunk_ids:
            db.execute(text("DELETE FROM episodic_chunks WHERE id = ANY(:ids)"),
                       {"ids": chunk_ids})
        if cluster_id is not None:
            db.execute(text("DELETE FROM episodic_cluster_links "
                            "WHERE cluster_id = :c"), {"c": cluster_id})
            db.execute(text("DELETE FROM context_clusters WHERE id = :c"),
                       {"c": cluster_id})
        if bs_id is not None:
            db.execute(text("DELETE FROM batch_summaries WHERE id = :i"),
                       {"i": bs_id})
        if conv_id is not None:
            db.execute(text("DELETE FROM conversation_summaries "
                            "WHERE conversation_id = :c"), {"c": conv_id})
        if turn_ids:
            db.execute(text("DELETE FROM episodic_memory WHERE id = ANY(:ids)"),
                       {"ids": turn_ids})
        if conv_id is not None:
            db.execute(text("DELETE FROM conversations WHERE id = :c"),
                       {"c": conv_id})
        if pattern_id is not None:
            db.execute(text("DELETE FROM procedural_memory WHERE id = :i"),
                       {"i": pattern_id})
        if decision_id is not None:
            db.execute(text("DELETE FROM decisions WHERE id = :i"),
                       {"i": decision_id})
        if project_id is not None:
            db.execute(text("DELETE FROM projects WHERE id = :i"),
                       {"i": project_id})
        if rag_chunk_ids:
            db.execute(text("DELETE FROM rag_chunks WHERE id = ANY(:ids)"),
                       {"ids": rag_chunk_ids})
        if rag_doc_id is not None:
            db.execute(text("DELETE FROM rag_documents WHERE id = :i"),
                       {"i": rag_doc_id})
        if slot_id is not None:
            db.execute(text("DELETE FROM memory_slots WHERE id = :i"),
                       {"i": slot_id})
        db.execute(text("DELETE FROM idempotency_keys WHERE key = :k"),
                   {"k": ikey})
        db.commit()
    finally:
        db.close()
        if scratch_engine is not None:
            scratch_engine.dispose()
        try:
            _admin_exec(f"DROP DATABASE IF EXISTS {SCRATCH_DB} WITH (FORCE)")
            _admin_exec(f"DROP DATABASE IF EXISTS {RESTORE_DB} WITH (FORCE)")
        except Exception as e:
            print(f"scratch-db cleanup warning: {e}")
        shutil.rmtree(tmp_root, ignore_errors=True)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
