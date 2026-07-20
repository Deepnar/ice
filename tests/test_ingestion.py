"""F10/F14 behavioral test — specs/F10_F14_import.md §4 checks 1–6.

F10: the format adapters (golden NormalizedConversations), the replay engine
(original timestamps, gap-derived sessions, full pipeline per turn, C4 summary
at the end), hash-idempotent resume, and the decay-policy math. F14: the raw
slicer (word-boundary slices, stubbed overlap seam, end dedup, synthetic-time
provenance).

House pattern: live Postgres, uniquely-marked fixture rows, stubbed LLM +
embedder (no model loads, no Ollama), cleanup in `finally` — NEVER truncates.
Snapshot-diff cleanup removes exactly the rows this run created in the derived
stores; the imported conversations/turns/import rows are deleted explicitly.

Run: uv run python tests/test_ingestion.py
"""
import hashlib
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sqlalchemy import text

# ── Stub the ONE embedder BEFORE any worker module captures it (G13: every
# worker does `embedder = get_embedder()` at import; pre-seeding the singleton
# means codex/procedural/clustering/post-flight all share this stub) ─────────
import src.memory.embedder as _emb_mod


class StubEmbedder:
    """Deterministic pseudo-embeddings varied by text (so clustering has
    something to separate) — never loads the real model."""
    def _vec(self, s):
        h = hashlib.sha256((s or "").encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(1024).astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def encode(self, texts, convert_to_tensor=False, **kwargs):
        if isinstance(texts, (list, tuple)):
            arr = np.stack([self._vec(t) for t in texts]) if texts \
                else np.zeros((0, 1024), dtype="float32")
            if convert_to_tensor:
                import torch
                return torch.from_numpy(arr)
            return arr
        v = self._vec(texts)
        if convert_to_tensor:
            import torch
            return torch.from_numpy(v)
        return v


_emb_mod._embedder = StubEmbedder()

from src.api.db import SessionLocal
from src.ingestion import formats, importer, raw_slicer
from src.ingestion.formats import NormalizedConversation, NormalizedTurn
from src.memory.models import (
    CodexEntity,
    ConversationSummary,
    EpisodicMemory,
    ImportConversation,
    ImportRun,
    ProceduralMemory,
)
from src.workers import codex_extractor, procedural_extractor
from src.workers import post_flight
from src.workers.decay import apply_decay

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


FIX = os.path.join(os.path.dirname(__file__), "fixtures", "ingestion")
HEX = uuid.uuid4().hex[:6]
MARK = "ingmark" + HEX.translate(str.maketrans("0123456789", "abcdefghij"))

db = SessionLocal()


# ── LLM stubs (module attrs — the house seam) ───────────────────────────────
class _Msg:
    def __init__(self, c):
        self.message = type("M", (), {"content": c})


class _Resp:
    def __init__(self, c):
        self.choices = [_Msg(c)]


class _Completions:
    def __init__(self, c):
        self._c = c

    def create(self, **kw):
        return _Resp(self._c)


class _FakeBG:
    def __init__(self, c):
        self.chat = type("C", (), {"completions": _Completions(c)})


def install_pipeline_stubs():
    # No NER / no summary LLM: canned key terms, no per-turn summary.
    post_flight.extract_key_terms = lambda t, e, max_chars=2500: {
        "entities": [MARK], "figures": [], "identifiers": []}
    post_flight.generate_summary = lambda *a, **k: (None, None, None)
    # Codex: one canned triplet → real graph rows appear.
    codex_extractor.extract_triplets = lambda t, model_override="", topic_tags=None: [
        {"subject": MARK + "sub", "relation": "calls", "object": MARK + "obj"}]
    # Procedural: canned pattern → a procedural row appears.
    procedural_extractor.bg_client = _FakeBG(
        f"When {MARK}, the user asks for code then the class.")


def stub_summary_llm(prompt, max_tokens=400):
    return f"{MARK} summary of the conversation so far."


# ── snapshot-diff cleanup ───────────────────────────────────────────────────
# (table, pk-column) for the derived/global stores we clean by diff.
DERIVED = [
    ("codex_edges", "id"), ("codex_events", "id"), ("codex_snapshots", "id"),
    ("codex_entities", "id"), ("procedural_memory", "id"),
    ("context_clusters", "id"), ("batch_summaries", "id"),
    ("idempotency_keys", "key"), ("import_runs", "id"),
]


def snap(table, pk):
    return {r[0] for r in db.execute(text(f"SELECT {pk} FROM {table}")).fetchall()}


created_conv_ids: list = []
created_hashes: list = []
created_import_ids: list = []
decay_conv_id = None
before = {}


try:
    install_pipeline_stubs()
    for t, pk in DERIVED:
        before[t] = snap(t, pk)

    # ══ Check 1: format adapters → golden NormalizedConversations ══
    print("\n[1] format adapters")
    _f, cg = formats.normalize_file(os.path.join(FIX, "chatgpt_export.json"))
    check("chatgpt: current-node path, branch skipped",
          [(t.role, t.text) for t in cg[0].turns] == [
              ("user", "betamsg question one"),
              ("assistant", "gammareply answer one"),
              ("user", "epsilonmsg question two")]
          and cg[0].branch_messages_skipped == 1)

    _f, cl = formats.normalize_file(os.path.join(FIX, "claude_export.json"))
    check("claude: latest-leaf path, empty-text→content block, branch skipped",
          [(t.role, t.text) for t in cl[0].turns] == [
              ("user", "etamsg hello there"),
              ("assistant", "thetareply world response"),
              ("user", "kappamsg followup again")]
          and cl[0].branch_messages_skipped == 1)

    _f, ds = formats.normalize_file(os.path.join(FIX, "deepseek_export.json"))
    check("deepseek: latest-edit branch, THINK dropped, REQUEST/RESPONSE roles",
          [(t.role, t.text) for t in ds[0].turns] == [
              ("user", "mumsg query here"),
              ("assistant", "nureply the answer"),
              ("user", "ximsg the followup")]
          and ds[0].branch_messages_skipped == 1)

    _f, jl = formats.normalize_file(os.path.join(FIX, "generic.jsonl"))
    check("jsonl: two conversations split by key, titles carried",
          len(jl) == 2 and jl[0].title == "rhoconv" and jl[1].title == "phiconv"
          and len(jl[0].turns) == 3 and len(jl[1].turns) == 2)

    check("detect_format identifies each shape",
          formats.detect_format(os.path.join(FIX, "chatgpt_export.json")) == "chatgpt"
          and formats.detect_format(os.path.join(FIX, "claude_export.json")) == "claude"
          and formats.detect_format(os.path.join(FIX, "deepseek_export.json")) == "deepseek"
          and formats.detect_format(os.path.join(FIX, "generic.jsonl")) == "jsonl"
          and formats.detect_format(os.path.join(FIX, "raw_dump.txt")) == "raw")

    # ══ Check 2: mini import end-to-end (stub LLM) ══
    print("\n[2] replay engine end-to-end")
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(days=3)
    # conv 1 — small, code in responses (forces lossless → codex), a session gap
    conv1 = NormalizedConversation(
        provider="test", source_id=MARK + "c1", title=MARK + " codeconv",
        turns=[
            NormalizedTurn("user", f"{MARK} explain the function", t0),
            NormalizedTurn("assistant", "sure ```def f(): return 1```", t0),
            NormalizedTurn("user", f"{MARK} and the class now", t0 + timedelta(minutes=5)),
            NormalizedTurn("assistant", "ok ```class C: pass```", t0 + timedelta(minutes=5)),
            NormalizedTurn("user", f"{MARK} much later question", t0 + timedelta(days=2)),
            NormalizedTurn("assistant", "later ```x = 1```", t0 + timedelta(days=2)),
        ])
    # conv 2 — large enough to cross the C4 recent-window (>25k chars)
    big = (MARK + " paragraph of filler content that is long enough to matter ") * 90
    big_turns = []
    for i in range(8):
        bt = t0 + timedelta(minutes=10 * i)
        big_turns.append(NormalizedTurn("user", f"{MARK} q{i} " + big, bt))
        big_turns.append(NormalizedTurn("assistant", f"{MARK} a{i} " + big, bt))
    conv2 = NormalizedConversation(
        provider="test", source_id=MARK + "c2", title=MARK + " bigconv",
        turns=big_turns)

    cb_calls = []
    report = importer.import_conversations(
        db, None, [conv1, conv2], "hybrid",
        classifier=None, embedder=StubEmbedder(), llm=stub_summary_llm,
        run_batch_summary=False, run_conversation_summary=True,
        progress_cb=lambda r: cb_calls.append(dict(r)))

    c1_id = uuid.uuid5(importer.NS_ICE_IMPORT, f"test:{MARK}c1")
    c2_id = uuid.uuid5(importer.NS_ICE_IMPORT, f"test:{MARK}c2")
    created_conv_ids += [c1_id, c2_id]
    created_hashes += [formats.content_hash(conv1), formats.content_hash(conv2)]

    rows1 = db.query(EpisodicMemory).filter_by(conversation_id=c1_id).order_by(
        EpisodicMemory.timestamp).all()
    check("conv1: 3 turn-pairs stored", len(rows1) == 3)
    check("conv1: ORIGINAL timestamps preserved",
          rows1[0].timestamp == t0 and rows1[2].timestamp == t0 + timedelta(days=2))
    check("conv1: sessions gap-derived (2 close = same, 2-day gap = new)",
          rows1[0].session_id == rows1[1].session_id
          and rows1[0].session_id != rows1[2].session_id)
    check("conv1: raw_text has User/Assistant shape, embedding set",
          rows1[0].raw_text.startswith(f"User: {MARK} explain")
          and "\n\nAssistant:" in rows1[0].raw_text
          and rows1[0].embedding is not None)
    check("conv1: ts_provenance = original", rows1[0].ts_provenance == "original")

    codex_new = db.query(CodexEntity).filter(
        CodexEntity.canonical_name.like(f"{MARK}%")).count()
    check("codex rows appear (canned triplet extracted)", codex_new >= 1)
    proc_new = db.query(ProceduralMemory).filter(
        ProceduralMemory.pattern_name.like(f"When {MARK}%")).count()
    check("procedural rows appear (canned pattern)", proc_new >= 1)
    link_rows = db.execute(text(
        "SELECT count(*) FROM episodic_cluster_links l JOIN episodic_memory e "
        "ON e.id = l.episodic_id WHERE e.conversation_id = :c"),
        {"c": c1_id}).scalar()
    check("cluster rows appear for the imported conversation", link_rows >= 1)

    summ = db.query(ConversationSummary).filter_by(conversation_id=c2_id).first()
    check("C4 conversation summary written for the large conversation",
          summ is not None and MARK in (summ.summary_text or ""))
    check("import_progress fired per conversation (progress_cb)",
          len(cb_calls) == 2)
    check("report totals", report["conversations"] == 2 and report["turns"] == 11
          and report["complete"] is True)

    # ══ Check 3: re-run → 100% skipped by hash, zero duplicates ══
    print("\n[3] idempotent re-run")
    before_turns = db.query(EpisodicMemory).filter(
        EpisodicMemory.conversation_id.in_([c1_id, c2_id])).count()
    report2 = importer.import_conversations(
        db, None, [conv1, conv2], "hybrid",
        classifier=None, embedder=StubEmbedder(), llm=stub_summary_llm,
        run_batch_summary=False, run_conversation_summary=False)
    after_turns = db.query(EpisodicMemory).filter(
        EpisodicMemory.conversation_id.in_([c1_id, c2_id])).count()
    check("re-run skips both conversations by hash",
          report2["skipped_conversations"] == 2 and report2["conversations"] == 0)
    check("re-run creates zero duplicate turns", before_turns == after_turns)

    # ══ Check 4: decay-policy math + immunity window ══
    print("\n[4] decay policy")
    cd = importer.compute_decay
    s, imm, arch = cd("fresh", now - timedelta(days=100), now, False)
    check("fresh: 1.0, no immunity, not archived", s == 1.0 and imm is None and not arch)
    s, imm, arch = cd("preserve", now - timedelta(days=100), now, False)
    check("preserve: 1.0 + 14-day immunity window",
          s == 1.0 and imm is not None
          and abs((imm - now).days - importer.IMMUNE_WINDOW_DAYS) <= 0)
    s, imm, arch = cd("fast_forward", now - timedelta(days=20), now, False)
    check("fast_forward 20d: score == 0.95**20 exactly",
          abs(s - 0.95 ** 20) < 1e-9 and not arch)
    s, imm, arch = cd("fast_forward", now - timedelta(days=300), now, False)
    check("fast_forward very-old: floored at COLD_THRESHOLD, archived",
          abs(s - 0.05) < 1e-9 and arch)
    s, imm, arch = cd("hybrid", now - timedelta(days=15), now, False)
    check("hybrid ≤30d: preserve semantics (1.0 + immunity)",
          s == 1.0 and imm is not None)
    s, imm, arch = cd("hybrid", now - timedelta(days=45), now, False)
    check("hybrid 45d: fast-forward from day 30 → 0.95**15",
          abs(s - 0.95 ** 15) < 1e-9 and imm is None)
    s, imm, arch = cd("fast_forward", now - timedelta(days=400), now, True)
    check("creative: floored at 0.3, never archived", abs(s - 0.3) < 1e-9 and not arch)

    # live decay.py immunity-window regression
    decay_conv_id = uuid.uuid4()
    db.execute(text("INSERT INTO conversations (id, memory_scope_type) "
                    "VALUES (:i, 'auto')"), {"i": decay_conv_id})
    old_ts = now - timedelta(days=30)
    immune_id, expired_id = uuid.uuid4(), uuid.uuid4()
    for rid, until in ((immune_id, now + timedelta(days=5)),
                       (expired_id, now - timedelta(days=1))):
        db.execute(text(
            "INSERT INTO episodic_memory (id, conversation_id, batch_id, "
            "context_reliance, raw_text, timestamp, decay_score, "
            "decay_immune, is_bookmarked, access_count, is_archived, "
            "topic_tags, decay_immune_until, idempotency_key) VALUES "
            "(:id, :c, :b, 'Zero_Shot', :rt, :ts, 1.0, FALSE, FALSE, 0, "
            "FALSE, '{}', :until, :k)"),
            {"id": rid, "c": decay_conv_id, "b": uuid.uuid4(),
             "rt": f"User: {MARK}\n\nAssistant: x", "ts": old_ts,
             "until": until, "k": f"decayimm-{rid}"})
    db.commit()
    apply_decay(cycles=1)
    immune_row = db.get(EpisodicMemory, immune_id)
    expired_row = db.get(EpisodicMemory, expired_id)
    db.refresh(immune_row)
    db.refresh(expired_row)
    check("decay: open immunity window survives apply_decay",
          immune_row.decay_score == 1.0)
    check("decay: expired immunity window decays normally",
          expired_row.decay_score < 1.0)

    # ══ Check 5: F14 raw slicer ══
    print("\n[5] raw slicer (F14)")
    # test-tune the window so the tiny fixture yields 2 slices with an overlap
    _mt, _ow, _st = (raw_slicer.RAW_SLICE_TOKENS, raw_slicer.RAW_OVERLAP_WORDS,
                     raw_slicer.SEAM_TURNS)
    raw_slicer.RAW_SLICE_TOKENS, raw_slicer.RAW_OVERLAP_WORDS = 16, 5
    raw_text = open(os.path.join(FIX, "raw_dump.txt")).read()
    from src.memory.chunking import chunk_text
    n_slices = len(chunk_text(raw_text, max_tokens=16, overlap_words=5))
    check("raw dump slices at word boundaries (≥2 slices, overlap)", n_slices >= 2)

    ROLE = {f"{MARK}alphaone": "user", f"{MARK}betatwo": "assistant",
            f"{MARK}gammathree": "user", f"{MARK}deltafour": "assistant",
            f"{MARK}epsilonfive": "user"}
    MARKERS = ["alphaone", "betatwo", "gammathree", "deltafour", "epsilonfive"]
    seam_calls = []
    import json as _json

    def stub_raw_llm(prompt, system=""):
        if prompt.startswith("Two adjacent slices"):     # seam call
            seam_calls.append(prompt)
            m = __import__("re").search(
                r"Slice B head:\n(\[.*?\])\n\nRaw overlap", prompt, __import__("re").DOTALL)
            return m.group(1) if m else "[]"
        turns = []                                        # extract call
        for mk in MARKERS:
            if mk in prompt:
                key = f"{MARK}{mk}"
                turns.append({"role": ROLE[key], "text": key})
        return _json.dumps(turns)

    mtime = now - timedelta(hours=1)
    raw_conv = raw_slicer.slice_raw_text(
        raw_text, "rawmark dump", llm=stub_raw_llm, mtime=mtime)
    texts = [t.text for t in raw_conv.turns]
    check("raw: overlap seam call was invoked", len(seam_calls) >= 1)
    check("raw: dedup removed the planted boundary duplicate",
          texts.count(f"{MARK}gammathree") == 1 and len(texts) == 5)
    check("raw: synthetic timestamps end at mtime, 1/min back",
          raw_conv.ts_synthetic is True and raw_conv.turns[-1].ts == mtime
          and raw_conv.turns[0].ts == mtime - timedelta(minutes=4))

    # import the raw conversation → turns carry synthetic provenance
    importer.import_conversations(
        db, None, [raw_conv], "hybrid", classifier=None,
        embedder=StubEmbedder(), run_batch_summary=False,
        run_conversation_summary=False)
    raw_cid = uuid.uuid5(importer.NS_ICE_IMPORT, "raw:rawmark dump")
    created_conv_ids.append(raw_cid)
    created_hashes.append(formats.content_hash(raw_conv))
    raw_rows = db.query(EpisodicMemory).filter_by(conversation_id=raw_cid).all()
    check("raw import: turns flagged synthetic_raw_import provenance",
          len(raw_rows) >= 1
          and all(r.ts_provenance == "synthetic_raw_import" for r in raw_rows))
    raw_slicer.RAW_SLICE_TOKENS, raw_slicer.RAW_OVERLAP_WORDS, raw_slicer.SEAM_TURNS = \
        _mt, _ow, _st

    # ══ Check 6: service layer + ImportRun lifecycle ══
    print("\n[6] service layer")
    from src.services import ingestion as ingestion_svc
    from src.services.errors import ValidationError
    prev = ingestion_svc.start_import(
        db, os.path.join(FIX, "generic.jsonl"), "preserve", dry_run=True)
    check("dry_run: preview with estimate, no run row created",
          prev.get("dry_run") is True and prev["total_conversations"] == 2
          and prev["estimate_seconds"] == round(prev["total_turns"] * 6))
    try:
        ingestion_svc.start_import(db, os.path.join(FIX, "generic.jsonl"), "bogus")
        bad_policy = False
    except ValidationError:
        bad_policy = True
    check("bad decay policy rejected", bad_policy)

    run = ingestion_svc.start_import(
        db, os.path.join(FIX, "generic.jsonl"), "preserve",
        dry_run=False, runtime=None, classifier=None)
    created_import_ids.append(uuid.UUID(run["id"]))
    jl2 = formats.normalize_file(os.path.join(FIX, "generic.jsonl"))[1]
    for c in jl2:
        created_conv_ids.append(uuid.uuid5(importer.NS_ICE_IMPORT, f"jsonl:{c.source_id}"))
        created_hashes.append(formats.content_hash(c))
    check("inline import completes + counters bumped",
          run["status"] == "completed" and run["done_conversations"] == 2
          and run["done_turns"] == 3)   # 3 turn-pairs (convA 2 + convB 1)
    status = ingestion_svc.import_status(db, run["id"])
    check("import_status returns the run by id",
          status["id"] == run["id"] and status["status"] == "completed")

finally:
    # ── cleanup: explicit for what we created, snapshot-diff for derived ──
    try:
        ids = list(created_conv_ids) + ([decay_conv_id] if decay_conv_id else [])
        # 1) rows I created that reference episodic / conversations / runs
        if ids:
            db.execute(text(
                "DELETE FROM episodic_cluster_links WHERE episodic_id IN "
                "(SELECT id FROM episodic_memory WHERE conversation_id = ANY(:c))"),
                {"c": ids})
            db.execute(text("DELETE FROM conversation_summaries "
                            "WHERE conversation_id = ANY(:c)"), {"c": ids})
        if created_hashes:
            db.execute(text("DELETE FROM import_conversations "
                            "WHERE content_hash = ANY(:h)"), {"h": created_hashes})
        if created_import_ids:
            db.execute(text("DELETE FROM import_conversations WHERE import_id = ANY(:i)"),
                       {"i": created_import_ids})
        if ids:
            db.execute(text("DELETE FROM episodic_memory "
                            "WHERE conversation_id = ANY(:c)"), {"c": ids})
            # my clusters, explicit (FK → conversations; episodic already gone)
            db.execute(text("DELETE FROM context_clusters "
                            "WHERE conversation_id = ANY(:c)"), {"c": ids})
        db.commit()
        # 2) snapshot-diff the derived/global stores — removes any remaining new
        #    context_clusters (FK → conversations) BEFORE conversations go
        for t, pk in DERIVED:
            keep = before.get(t, set())
            if keep:
                db.execute(text(f"DELETE FROM {t} WHERE {pk} <> ALL(:k)"),
                           {"k": list(keep)})
            else:
                db.execute(text(f"DELETE FROM {t}"))
        db.commit()
        # 3) finally the conversations (now unreferenced)
        if ids:
            db.execute(text("DELETE FROM conversations WHERE id = ANY(:c)"), {"c": ids})
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"  cleanup warning: {exc}")
    db.close()

print(f"\n{'='*50}\n  {_passed} passed, {_failed} failed\n{'='*50}")
sys.exit(1 if _failed else 0)
