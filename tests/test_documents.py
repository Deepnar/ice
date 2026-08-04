"""C12 behavioral test: documents become memory, opt-in, and stay opt-in.

Runs against the live Postgres (docker up). Inserts its own uniquely-marked
rows and deletes them afterwards — never truncates (the dev DB holds real data).

Every scope check is TWO-SIDED: the in-scope row IS returned as well as the
out-of-scope row is NOT. A one-sided "the document text is absent" passes just
as happily on a broken filter as on a working one, because an empty result
satisfies it.

Run: uv run python tests/test_documents.py
"""
import hashlib
import os
import sys
import tempfile
import uuid

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text as sql_text

# ── Stub the ONE embedder BEFORE any worker module captures it (G13). Unlike
# most suites this one is deterministic AND text-varying, so a document
# section can actually WIN a vector search instead of tying with everything
# (G30's complaint about constant-embedder suites). ─────────────────────────
import src.memory.embedder as _emb_mod


class StubEmbedder:
    def _vec(self, s):
        h = hashlib.sha256((s or "").encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(1024).astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def encode(self, texts, convert_to_tensor=False, **kwargs):
        if isinstance(texts, (list, tuple)):
            arr = (np.stack([self._vec(t) for t in texts]) if texts
                   else np.zeros((0, 1024), dtype="float32"))
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

from src.api.db import SessionLocal  # noqa: E402
from src.classifier.schemas import ClassificationResult  # noqa: E402
from src.ingestion.documents import ingest, parsers  # noqa: E402
from src.ingestion.documents import kind as kind_mod
from src.memory.models import (  # noqa: E402
    CodexEdge,
    CodexEntity,
    CodexEvent,
    ContextCluster,
    Conversation,
    ConversationSummary,
    Document,
    DocumentLink,
    EpisodicChunk,
    EpisodicClusterLink,
    EpisodicMemory,
    ProceduralMemory,
)
from src.services import documents as documents_svc  # noqa: E402
from src.services import scoping as scoping_svc  # noqa: E402
from src.services.errors import ValidationError  # noqa: E402
from src.workers import codex_extractor, post_flight, procedural_extractor  # noqa: E402

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


HEX = uuid.uuid4().hex[:6].translate(str.maketrans("0123456789", "abcdefghij"))
MARK = "docmark" + HEX                 # appears in the document only
SHARED = "sharedmark" + HEX            # appears in the document AND a chat turn
EMB = [0.05] * 1024

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


_codex_calls = []


def install_pipeline_stubs():
    post_flight.extract_key_terms = lambda t, e, max_chars=2500: {
        "entities": [MARK], "figures": [], "identifiers": []}
    post_flight.generate_summary = lambda *a, **k: (
        f"{MARK} summary.", 1.0, f"{MARK} abstract.")

    def _triplets(t, model_override="", topic_tags=None, gaps=None):
        _codex_calls.append(t)
        return [{"subject": MARK + "sub", "relation": "describes",
                 "object": MARK + "obj"}]
    codex_extractor.extract_triplets = _triplets
    procedural_extractor.bg_client = _FakeBG(
        f"When {MARK}, the user asks about the document.")


def stub_summary_llm(prompt, max_tokens=400):
    return f"{MARK} document-level summary."


def clf(prompt):
    return ClassificationResult(
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", raw_probs=[0.0] * 27,
        max_confidence=0.9, prompt=prompt)


def mk_chat_turn(conv, body):
    t = EpisodicMemory(
        conversation_id=conv.id, batch_id=uuid.uuid4(),
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory",
        raw_text=f"User: {body}\n\nAssistant: noted", embedding=EMB,
        decay_score=1.0, idempotency_key=f"test-c12-{uuid.uuid4()}")
    db.add(t)
    db.commit()
    return t


def doc_text_visible(conv, marker=MARK):
    """Does a retrieval run for `conv` surface the document's text?

    `conv_id` is derived the way `retrieve()` derives it — from
    `scope["conversation_id"]`, which only the self-scoping modes set. Passing
    the current conversation's id unconditionally would pin every leg to that
    one conversation and make cross-conversation memory (the entire point)
    invisible, so a test that did that would report failure for every
    document, enabled or not."""
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    scope = scoping_svc.resolve_retrieval_scope(db, conv)
    conv_id = scope.get("conversation_id")
    orch = HybridRetrievalOrchestrator(db, _emb_mod._embedder)
    frags = orch._bm25_episodic(clf(marker), scope=scope, conv_id=conv_id)
    frags += orch._vector_episodic(_emb_mod._embedder.encode(marker).tolist(),
                                   clf(marker), scope=scope, conv_id=conv_id)
    return any(marker in f.text for f in frags)


def denied_entities(conv):
    """The codex deny set this conversation's scope produces."""
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    scope = scoping_svc.resolve_retrieval_scope(db, conv)
    orch = HybridRetrievalOrchestrator(db, _emb_mod._embedder)
    orch._resolve_exclusion_sets(scope)
    return orch._denied_entity_ids


# Trap 6, disarmed: the cleanup used to select this suite's documents from a
# hardcoded FILENAME allow-list, so every new fixture leaked until someone
# noticed. Record which document conversations existed BEFORE the run instead;
# anything new is ours, whatever it is called and however it was created.
_pre_doc_convs = {r[0] for r in db.execute(sql_text(
    "SELECT id FROM conversations WHERE kind <> 'chat'")).fetchall()}

tmpdir = tempfile.mkdtemp(prefix="ice-c12-")
conv_a = conv_b = conv_incog = None
doc = doc_csv = None

try:
    install_pipeline_stubs()
    conv_a = Conversation(memory_scope_type="auto")
    conv_b = Conversation(memory_scope_type="auto")
    conv_incog = Conversation(memory_scope_type="none")
    db.add_all([conv_a, conv_b, conv_incog])
    db.commit()

    # ══ Check 1: a markdown upload becomes a document conversation ════════
    print("── check 1: ingestion end-to-end ──")
    md_path = os.path.join(tmpdir, "spec.md")
    # Deliberately a REAL-sized document (>25k chars): the conversation-summary
    # worker only writes a row once a conversation outgrows the recent window
    # (0.3 x (23k - 1.8k) = 6,360 tokens), which is correct behavior — a
    # two-paragraph file needs no summary — and a toy fixture would test the
    # gate instead of the pipeline.
    filler = (" The pipeline stage records provenance, applies the density "
              "decision, and hands the result to the next stage without "
              "mutating what the author wrote.")
    with open(md_path, "w") as fh:
        fh.write(f"# Overview\n\nThe {MARK} subsystem coordinates ingestion "
                 f"and also mentions {SHARED} exactly once.{filler * 40}\n\n")
        for n in range(2, 9):
            fh.write(f"# Section {n}\n\nSection {n} of the specification "
                     f"describes stage {n} of the pipeline.{filler * 40}\n\n")
    result = documents_svc.add_document(
        db, conversation_id=str(conv_a.id), path=md_path,
        classifier=None, embedder=_emb_mod._embedder, llm=stub_summary_llm,
        kind_llm=lambda p, s: "DOCUMENT")
    doc = db.get(Document, uuid.UUID(result["id"]))
    doc_conv = db.get(Conversation, doc.conversation_id)
    sections = db.query(EpisodicMemory).filter_by(
        conversation_id=doc.conversation_id).order_by(
            EpisodicMemory.timestamp).all()
    check("document row is ready", doc.status == "ready")
    check("document has its own conversation, kind='document'",
          doc_conv is not None and doc_conv.kind == "document")
    check("sections were stored", len(sections) >= 2
          and doc.n_sections == len(sections))
    check("each section carries a provenance header",
          all(s.raw_text.startswith("[spec.md") for s in sections))
    check("section text survives verbatim",
          any(MARK in s.raw_text for s in sections))

    # ══ Check 2: temporal + session shape ════════════════════════════════
    print("── check 2: timestamps, provenance, session ──")
    check("ts_provenance is document_ingest",
          all(s.ts_provenance == "document_ingest" for s in sections))
    check("one session for the whole document",
          len({s.session_id for s in sections}) == 1
          and sections[0].session_id is not None)
    check("section timestamps ascend",
          all(sections[i].timestamp < sections[i + 1].timestamp
              for i in range(len(sections) - 1)))

    # ══ Check 3: codex ran, and the lossless forcing is load-bearing ═════
    print("── check 3: the document reached the knowledge graph ──")
    ents = db.query(CodexEntity).filter(
        CodexEntity.canonical_name.like(f"{MARK}%")).all()
    check("codex entities extracted from the document", len(ents) >= 1)
    check("every section is flagged lossless (forced for documents)",
          all(s.lossless_flag for s in sections))
    # The forcing is what makes check 3 pass: under the chat gate, `has_code`
    # reads the (empty) assistant half and entropy alone would decide.
    low_entropy = [s for s in sections if (s.entropy_score or 0) < 0.35]
    check("...and at least one section would have FAILED the chat gate",
          bool(low_entropy))

    # ══ Check 4: clustering ══════════════════════════════════════════════
    print("── check 4: sections entered the clustering pipeline ──")
    # ⚠ Divergence from the spec's check 4, recorded rather than papered over
    # (rule 12): it said "the sections are linked to a cluster". They are not,
    # and should not be — C5's wait-for-a-friend rule creates a cluster only
    # from >=2 MUTUALLY SIMILAR turns, and this suite's stub embedder is a
    # per-text hash, so no two sections are similar by construction. Asserting
    # a link would assert the stub's luck. What C12 owns is the WIRING — that
    # a document's sections are ordinary clustering input — and that is what
    # is asserted. Whether real embeddings cluster a document's sections is
    # C5's threshold question, measured at Z1.
    from src.workers.clustering import run_cluster_assignment
    cstats = run_cluster_assignment(db, conversation_ids=[str(doc.conversation_id)])
    check("clustering processes the document's sections like any turn",
          cstats["processed"] == len(sections))
    check("...and they are eligible (not private, not cluster-blocked)",
          not any(s.is_private for s in sections))

    # ══ Check 5: the document-level summary ══════════════════════════════
    print("── check 5: document summary ──")
    summary = db.get(ConversationSummary, doc.conversation_id)
    check("conversation_summaries holds the document summary",
          summary is not None and summary.summary_text)

    # ══ Check 6: isolation under `auto`, two-sided, and the live toggle ══
    print("── check 6: opt-in visibility (auto) ──")
    check("uploader A reads the document", doc_text_visible(conv_a) is True)
    check("B does NOT read it (and A did, so this is not vacuous)",
          doc_text_visible(conv_b) is False)
    documents_svc.set_document_enabled(db, str(doc.id), str(conv_b.id), True)
    db.expire_all()
    check("B reads it once enabled", doc_text_visible(conv_b) is True)
    documents_svc.set_document_enabled(db, str(doc.id), str(conv_b.id), False)
    db.expire_all()
    check("B stops reading it once disabled", doc_text_visible(conv_b) is False)
    check("A is unaffected by B's toggle", doc_text_visible(conv_a) is True)

    # ══ Check 7: the closed-set arm (manual scope) ═══════════════════════
    print("── check 7: opt-in visibility (manual/closed set) ──")
    conv_b.memory_scope_type = "manual"
    db.commit()
    check("manual B does not read the document",
          doc_text_visible(conv_b) is False)
    documents_svc.set_document_enabled(db, str(doc.id), str(conv_b.id), True)
    db.expire_all()
    scope_b = scoping_svc.resolve_retrieval_scope(db, conv_b)
    check("...the doc conversation joined the closed set",
          str(doc.conversation_id) in (scope_b.get("conversation_ids") or []))
    check("...and manual B now reads it", doc_text_visible(conv_b) is True)

    # ══ Check 8: the knowledge latch ═════════════════════════════════════
    print("── check 8: knowledge promotion latch ──")
    db.refresh(doc)
    check("latch tripped when a SECOND conversation enabled it",
          doc.knowledge_shared is True and doc.shared_at is not None)
    documents_svc.set_document_enabled(db, str(doc.id), str(conv_b.id), False)
    conv_b.memory_scope_type = "auto"
    db.commit()
    db.expire_all()
    db.refresh(doc)
    check("latch does NOT reset when both switch off",
          doc.knowledge_shared is True)
    check("text is opt-in again after disabling",
          doc_text_visible(conv_b) is False)
    doc_entity_ids = {e.id for e in db.query(CodexEntity).filter(
        CodexEntity.canonical_name.like(f"{MARK}%")).all()}
    check("...but the promoted knowledge is NOT denied to B",
          not (doc_entity_ids & denied_entities(conv_b)))

    # A second, unpromoted document proves the negative side is real: with the
    # latch untripped its entities ARE denied.
    md2 = os.path.join(tmpdir, "private_notes.md")
    with open(md2, "w") as fh:
        fh.write(f"# Notes\n\nThe {MARK}solo widget is only described here, "
                 "and nowhere else in the store at all.\n")
    result2 = documents_svc.add_document(
        db, conversation_id=str(conv_a.id), path=md2, classifier=None,
        embedder=_emb_mod._embedder, llm=stub_summary_llm,
        kind_llm=lambda p, s: "DOCUMENT")
    doc2 = db.get(Document, uuid.UUID(result2["id"]))
    db.refresh(doc2)
    scope_b2 = scoping_svc.resolve_retrieval_scope(db, conv_b)
    check("an UNPROMOTED document is in B's knowledge deny set",
          doc2.knowledge_shared is False
          and str(doc2.conversation_id) in
          (scope_b2.get("exclude_knowledge_conversation_ids") or []))
    check("...while the promoted one is NOT",
          str(doc.conversation_id) not in
          (scope_b2.get("exclude_knowledge_conversation_ids") or []))

    # ══ Check 9: an entity the document SHARES with a chat survives ══════
    print("── check 9: shared knowledge is never denied ──")
    shared_entity = CodexEntity(canonical_name=SHARED, entity_type="concept",
                                source="conversation")
    db.add(shared_entity)
    db.flush()
    chat_turn = mk_chat_turn(conv_b, f"a note about {SHARED} in a normal chat")
    doc2_turn = db.query(EpisodicMemory).filter_by(
        conversation_id=doc2.conversation_id).first()
    db.add(CodexEvent(entity_id=shared_entity.id, event_type="created",
                      batch_source=doc2_turn.batch_id))
    db.add(CodexEvent(entity_id=shared_entity.id, event_type="created",
                      batch_source=chat_turn.batch_id))
    db.commit()
    check("entity evidenced by BOTH the hidden doc and a chat is not denied",
          shared_entity.id not in denied_entities(conv_b))

    # ══ Check 10: incognito ══════════════════════════════════════════════
    print("── check 10: incognito refuses documents ──")
    refused = False
    try:
        documents_svc.set_document_enabled(db, str(doc.id),
                                           str(conv_incog.id), True)
    except ValidationError:
        refused = True
    check("enabling a document in an incognito conversation is refused",
          refused)

    # ══ Check 11: de-duplication ═════════════════════════════════════════
    print("── check 11: same bytes = same document ──")
    again = documents_svc.add_document(
        db, conversation_id=str(conv_b.id), path=md_path, classifier=None,
        embedder=_emb_mod._embedder, llm=stub_summary_llm,
        kind_llm=lambda p, s: "DOCUMENT")
    n_docs = db.query(Document).filter_by(sha256=doc.sha256).count()
    check("re-upload did not create a second document",
          again.get("deduplicated") is True and n_docs == 1)
    check("...it enabled the existing one in the asking conversation",
          again.get("enabled_here") is True)
    documents_svc.set_document_enabled(db, str(doc.id), str(conv_b.id), False)

    # ══ Check 12: a PDF with no text layer is refused ════════════════════
    print("── check 12: scanned PDF refused, not silently empty ──")
    blank_pdf = os.path.join(tmpdir, "scan.pdf")
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(blank_pdf, "wb") as fh:
            writer.write(fh)
        refused_msg = ""
        try:
            parsers.parse_file(blank_pdf)
        except parsers.UnsupportedDocument as exc:
            refused_msg = str(exc)
        check("no-text PDF raises, naming the scan and OCR",
              "no extractable text" in refused_msg and "OCR" in refused_msg)
    except Exception as exc:
        check(f"no-text PDF refused (skipped: {exc})", False)

    # ══ Check 13: a CSV is DESCRIBED, not dumped ═════════════════════════
    print("── check 13: tabular ingestion describes the table ──")
    csv_path = os.path.join(tmpdir, "metrics.csv")
    with open(csv_path, "w") as fh:
        fh.write("region,churn_rate,accounts\n")
        for i in range(400):
            fh.write(f"r{i % 5},{0.01 * (i % 30):.3f},{100 + i}\n")
    parsed_csv = parsers.parse_file(csv_path)
    schema_text = parsed_csv.blocks[0].text
    check("the description names every column",
          all(c in schema_text for c in ("region", "churn_rate", "accounts")))
    check("the description carries statistics", "min" in schema_text
          and "400" in schema_text.replace(",", ""))
    sample_text = parsed_csv.blocks[1].text
    check("rows are sampled, not dumped",
          sample_text.count("\n") < 60 and "|" in sample_text)

    # ══ Check 14: doc-vs-transcript is decided by CONTENT (G28) ══════════
    print("── check 14: kind detection is style-invariant ──")
    convo = ("{u}: how do I rotate the API key?\n"
             "{a}: open settings, then click rotate.\n"
             "{u}: does that invalidate the old one immediately?\n"
             "{a}: yes, immediately.\n")
    variants = {
        "labelled User/Assistant": convo.format(u="User", a="Assistant"),
        "renamed Me/Bot": convo.format(u="Me", a="Bot"),
        "no labels at all": convo.format(u="", a="").replace(": ", ""),
    }
    seen = {}
    for name, blob in variants.items():
        seen[name] = kind_mod.detect_blob_kind(
            blob, llm=lambda p, s: "TRANSCRIPT")
    check("all three phrasings of one transcript agree",
          len(set(seen.values())) == 1 and set(seen.values()) == {"transcript"})
    prose = ("The assistant pattern is common in service design. "
             "Assistant: is a label some systems print. Assistant: appears "
             "again here. Assistant: and once more, in prose about labels.")
    check("prose that merely CONTAINS 'Assistant:' is a document",
          kind_mod.detect_blob_kind(prose, llm=lambda p, s: "DOCUMENT")
          == "document")
    # The old rule (post_flight's count) disagrees on that same prose — which
    # is what makes replacing it load-bearing rather than cosmetic.
    check("...and the OLD count-the-label rule got it wrong",
          prose.count("Assistant:") >= 3)
    check("an explicit choice overrides detection entirely",
          kind_mod.detect_blob_kind(
              prose, explicit="transcript",
              llm=lambda p, s: "DOCUMENT") == "transcript")

    # ══ Check 15: the RAG leg is gone ════════════════════════════════════
    print("── check 15: the v1 RAG leg no longer exists ──")
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator as _H
    check("_rag_lookup is gone", not hasattr(_H, "_rag_lookup"))
    src_root = os.path.join(os.path.dirname(__file__), "..", "src")
    hits = []
    for dirpath, _dirs, files in os.walk(src_root):
        if "__pycache__" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            body = open(os.path.join(dirpath, fn), encoding="utf-8").read()
            if '"document", "pdf", "reference", "manual", "guide"' in body:
                hits.append(fn)
    check("the five-noun gate appears nowhere in src/", not hits)
    rag_rows = db.execute(sql_text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name IN ('rag_chunks','rag_documents')")).fetchall()
    tables = {r[0] for r in rag_rows}
    check("rag_chunks / rag_documents are dropped", not tables)

    # ══ Check 16: sections are chunk-grade, not is_document ══════════════
    print("── check 16: the invisible-section trap ──")
    check("sections are NOT flagged is_document",
          not any(s.is_document for s in sections))
    # If they were, the vector leg would exclude them and the chunker would
    # produce nothing for them — the row would be represented nowhere.
    chunk_rows = db.query(EpisodicChunk).filter(
        EpisodicChunk.turn_id.in_([s.id for s in sections])).count()
    check("...so their own embeddings are what retrieval searches",
          chunk_rows == 0 and all(s.embedding is not None for s in sections))
    check("build_sections respects block boundaries",
          len(ingest.build_sections(parsers.parse_file(md_path))) >= 2)

    # ══ Check 17: a document that ingested NOTHING says so ═══════════════
    print("── check 17: no silent failure ──")
    # Found by driving the REST adapter by hand: every section failing left the
    # document status='ready' with n_sections=0 — indistinguishable from a
    # document that genuinely held nothing.
    broken = os.path.join(tmpdir, "broken.md")
    with open(broken, "w") as fh:
        fh.write("# Broken\n\nthis document's sections will all fail to store.\n")
    real_store = ingest._store_section
    ingest._store_section = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("planted section failure"))
    try:
        res_broken = documents_svc.add_document(
            db, conversation_id=str(conv_a.id), path=broken, classifier=None,
            embedder=_emb_mod._embedder, llm=stub_summary_llm,
            kind_llm=lambda p, s: "DOCUMENT")
    finally:
        ingest._store_section = real_store
    check("a document whose every section failed is 'failed', not 'ready'",
          res_broken["status"] == "failed")
    check("...and the error names what happened",
          "every section failed" in (res_broken.get("error") or ""))

    # ══ Check 18: a BLOB ingests through the ENQUEUED path ════════════════
    print("── check 18: blob ingest with a runtime ──")
    # Every check above passes `runtime=None`, so all of them ingest inline —
    # and both live adapters (user_control.py, mcp/server.py) pass the real
    # runtime, which enqueues `ingest_document`, which re-read `source_path`.
    # A blob has never had one, so every pasted document through REST or MCP
    # died as "source file is gone". Same lesson as check 17: the suite proved
    # the call connects, not that the surface works.
    class _EnqueueRuntime:
        """Records the enqueue instead of running it, so the check can then
        run the job itself — which is what the real runtime does later."""
        def __init__(self):
            self.jobs = []

        def enqueue(self, name, **kw):
            self.jobs.append((name, kw))

    rt = _EnqueueRuntime()
    blob_body = ("# Pasted spec\n\n" +
                 f"The {MARK} protocol negotiates a session key.\n\n"
                 "## Second section\n\n" +
                 "Rekeying happens every hour under the same protocol.\n")
    res_blob = documents_svc.add_document(
        db, conversation_id=str(conv_a.id), blob=blob_body,
        filename="pasted.md", runtime=rt, classifier=None,
        embedder=_emb_mod._embedder, llm=stub_summary_llm,
        kind_llm=lambda p, s: "DOCUMENT")
    blob_doc = db.get(Document, uuid.UUID(res_blob["id"]))
    check("the blob's text is stored, not just its sha",
          blob_doc.source_text == blob_body and blob_doc.source_path is None)
    check("...and the ingest was ENQUEUED, not run inline",
          rt.jobs and rt.jobs[0][0] == "ingest_document"
          and res_blob["status"] == "ingesting")

    documents_svc._get_runtime = lambda: None       # run this slice to the end
    documents_svc.run_document_ingest(db, res_blob["id"])
    db.refresh(blob_doc)
    check("the enqueued job ingests the blob instead of failing",
          blob_doc.status == "ready", )
    check("...and does NOT report the source as gone (the pre-fix failure)",
          "source file is gone" not in (blob_doc.error or ""))
    check("its sections really landed",
          blob_doc.n_sections >= 2 and db.query(EpisodicMemory).filter_by(
              conversation_id=blob_doc.conversation_id).count() >= 2)

finally:
    db.rollback()
    # FK order (trap 6): links → chunks → turns → procedural/clusters →
    # documents → conversations. A cleanup that deletes a parent first raises
    # on commit and leaves rows behind in a store that is supposed to be empty.
    doc_convs = [r[0] for r in db.execute(sql_text(
        "SELECT id FROM conversations WHERE kind <> 'chat'")).fetchall()
        if r[0] not in _pre_doc_convs]
    convs = [c for c in (conv_a, conv_b, conv_incog) if c is not None]
    conv_ids = [c.id for c in convs] + doc_convs
    turn_ids = [t.id for t in db.query(EpisodicMemory).filter(
        EpisodicMemory.conversation_id.in_(conv_ids)).all()] if conv_ids else []
    if turn_ids:
        db.query(EpisodicClusterLink).filter(
            EpisodicClusterLink.episodic_id.in_(turn_ids)).delete(
                synchronize_session=False)
        db.query(EpisodicChunk).filter(
            EpisodicChunk.turn_id.in_(turn_ids)).delete(
                synchronize_session=False)
        batch_ids = [t.batch_id for t in db.query(EpisodicMemory).filter(
            EpisodicMemory.id.in_(turn_ids)).all()]
        if batch_ids:
            db.query(CodexEvent).filter(
                CodexEvent.batch_source.in_(batch_ids)).delete(
                    synchronize_session=False)
        db.commit()
    # Edges reference entities, and events reference entities: both go first
    # or the entity DELETE raises and the "cleaned up" store keeps the rows
    # (trap 6 — a `finally` that fails silently is worse than none).
    doomed = [e.id for e in db.query(CodexEntity).filter(
        CodexEntity.canonical_name.like(f"{MARK}%")).all()]
    doomed += [e.id for e in db.query(CodexEntity).filter_by(
        canonical_name=SHARED).all()]
    if doomed:
        db.query(CodexEdge).filter(
            (CodexEdge.source_id.in_(doomed))
            | (CodexEdge.target_id.in_(doomed))).delete(
                synchronize_session=False)
        db.query(CodexEvent).filter(
            CodexEvent.entity_id.in_(doomed)).delete(
                synchronize_session=False)
        db.commit()
        db.query(CodexEntity).filter(CodexEntity.id.in_(doomed)).delete(
            synchronize_session=False)
    db.query(ProceduralMemory).filter(
        ProceduralMemory.pattern_description.like(f"%{MARK}%")).delete(
            synchronize_session=False)
    db.commit()
    if conv_ids:
        db.query(EpisodicMemory).filter(
            EpisodicMemory.conversation_id.in_(conv_ids)).delete(
                synchronize_session=False)
        db.query(ConversationSummary).filter(
            ConversationSummary.conversation_id.in_(conv_ids)).delete(
                synchronize_session=False)
        db.query(DocumentLink).filter(
            DocumentLink.conversation_id.in_(conv_ids)).delete(
                synchronize_session=False)
        db.commit()
        db.query(Document).filter(
            Document.conversation_id.in_(doc_convs)).delete(
                synchronize_session=False)
        db.commit()
        db.query(Conversation).filter(
            Conversation.id.in_(conv_ids)).delete(synchronize_session=False)
        db.commit()
    # Clusters this run created (named by the stub) have no members left.
    for clus in db.query(ContextCluster).all():
        if db.query(EpisodicClusterLink).filter_by(cluster_id=clus.id).count() == 0:
            db.delete(clus)
    db.commit()
    # Trap 6: verify, do not assume `finally` won. Scoped to THIS run's rows —
    # a store-wide count would blame this suite for another suite's residue.
    mine = db.execute(sql_text(
        "SELECT (SELECT count(*) FROM episodic_memory "
        "        WHERE conversation_id = ANY(:cids)) + "
        "       (SELECT count(*) FROM documents "
        "        WHERE conversation_id = ANY(:cids)) + "
        "       (SELECT count(*) FROM document_links "
        "        WHERE conversation_id = ANY(:cids)) + "
        "       (SELECT count(*) FROM codex_entities "
        "        WHERE canonical_name LIKE :mark OR canonical_name = :shared)"),
        {"cids": [str(c) for c in conv_ids] or [str(uuid.uuid4())],
         "mark": f"{MARK}%", "shared": SHARED}).scalar()
    total = db.execute(sql_text(
        "SELECT (SELECT count(*) FROM conversations) + "
        "(SELECT count(*) FROM episodic_memory)")).scalar()
    print(f"\nthis run's rows remaining: {mine} (must be 0)")
    print(f"store-wide conversations+turns: {total} "
          "(non-zero = residue from another suite, not this one)")
    db.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
