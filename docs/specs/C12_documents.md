# C12 — Documents: ingestion, isolation, and the death of the RAG leg

> **STATUS: C12a SHIPPED 2026-07-28 (`27f64eb`, migration `5fe5ad26480b`, 46/46).**
> C12b (in-chat paste promotion) remains open. **Two divergences recorded during
> implementation (rule 12), both corrected in place below:**
> 1. **§5 check 4** said "the doc's sections are linked to a cluster". They are
>    not, and should not be: C5's wait-for-a-friend rule creates a cluster only
>    from ≥2 *mutually similar* turns, and the suite's stub embedder is a
>    per-text hash, so asserting a link would assert the stub's luck. The check
>    now asserts the **wiring** (clustering processes a document's sections like
>    any other turn). Whether real embeddings cluster them is C5's threshold
>    question, measured at Z1.
> 2. **§1 D12's split** put the transcript path in C12b. It shipped in C12a
>    instead, because `add_document` must decide the destination *before* it
>    writes anything — the fork cannot live downstream of ingestion. C12b is
>    therefore only the **in-chat paste promotion** half.
>
> One thing §5 check 5 assumed and reality corrected: a document only gets a
> conversation-summary row once it **outgrows the recent window**
> (`0.3 × (23k − 1.8k)` ≈ 6,360 tokens), which is correct behavior — a
> two-paragraph file needs no summary — so the test fixture is a real-sized
> document rather than a toy.

Assumes decided specs: [C6 (scope semantics — SHIPPED 2026-07-28, `a1f6b8d94c22`)](../ROADMAP.md), [F10_F14_import.md](F10_F14_import.md), [E_coding_core.md](E_coding_core.md) (D10/E10), [C4_C9_memory_quality.md](C4_C9_memory_quality.md), [C10_C11_deletion_chat_control.md](C10_C11_deletion_chat_control.md)

> **Why this spec exists at all.** The roadmap's C12 entry carries an S1 note saying
> "no separate spec — decision-complete as written". **That stopped being true on
> 2026-07-28.** The design session added five decisions the entry does not contain:
> the opt-in visibility model, the enable/disable *live toggle*, the knowledge-promotion
> latch, the document-vs-transcript fork, and the pluggable extraction engine. Rule 12
> says fix the record before improvising past it. The roadmap entry now points here.

---

## 1. Decisions

### D1 — A document becomes its own conversation. (user, 2026-07-28)

Sections are turns; the conversation is the document. This is not a metaphor for
storage convenience — it is what makes every downstream feature free:

| What C12 needs | What it costs, because a doc is a conversation |
| --- | --- |
| "add an existing doc to another chat's scope" | `included_conversation_ids` — C6, shipped |
| a document-level summary | `conversation_summaries` — C4, shipped |
| section-level retrieval | the episodic legs — C2/C3, shipped |
| codex entities from the doc | `evaluate_turn`'s chain — shipped |
| sections joining topic clusters | `run_cluster_assignment` — C5, shipped |
| scope + exclusion honored | `resolve_retrieval_scope` — C6, shipped |

**Rejected:** turns inside the uploading chat (the "paste" shape). It makes sharing a
document mean dragging the whole chat along, and C10's delete-conversation cascade
would destroy the document with the chat that happened to introduce it.

### D2 — Documents are OPT-IN everywhere, forever. (user, 2026-07-28)

> *"it's like a global upload where ANY chat can access it by enabling it BUT at the
> start when one uploads it it's just restricted to the current one"*

A document's **text** is never part of an unscoped read. On upload it is enabled in
exactly one conversation (the one that uploaded it) and invisible everywhere else. Any
conversation may enable it later.

**Mechanism — no new enforcement code.** `resolve_retrieval_scope` computes the
document sets and expresses them in C6's *existing* vocabulary:

* closed-set modes (`manual`, `project`, project-attached): enabled doc conversation ids
  are **added** to `scope["conversation_ids"]`;
* open mode (`auto`): every *not-currently-enabled* doc/transcript conversation id is
  **added** to `scope["exclude_conversation_ids"]`, which `_exclusion_filter` already
  applies to every episodic leg.

### D3 — Enable/disable is a LIVE TOGGLE, not an attachment. (user, 2026-07-28)

> *"when I say enable I mean a toggle like for the next few prompts having the info from
> that doc is useful let me enable it, then later work done let me disable it"*

Visibility of the document's **text** follows the current switch state, flipped freely
at any time, taking effect on the next turn. There is no "attached forever" state and
no cost to switching off.

### D4 — Knowledge promotes to global once a SECOND conversation has ever enabled it. (user, 2026-07-28)

Two different things are being scoped and they are scoped differently:

* **Text** (the sections) — opt-in **forever**, per D2/D3. A 300-page manual never
  floods retrieval in a chat that has not switched it on. This never changes.
* **Knowledge** (codex entities/edges, procedural patterns derived from the doc) —
  isolated while the doc lives in one conversation; **globally visible once a second
  distinct conversation has ever enabled it.**

**Why "all chats" and not "the two that enabled it".** Codex entities are not per-chat
objects — an entity extracted from the PDF is the *same node* the user's chats already
write to. Per-chat knowledge visibility means a permanent per-document access list
consulted on every request, and makes "what do I know" depend on who is asking. The
noise problem D2 protects against is **volume of text**, which stays solved by D2.

**The latch is one-way and event-based.** `documents.knowledge_shared` flips true the
first time a *second distinct* conversation enables the doc, and stays true after both
switch off — a doc you used properly does not lose its knowledge when you tidy up.
Rejected: a duration/count threshold ("enabled for N days"), which would add a
background job plus an unmeasured knob, i.e. another Z1 tuning liability, for a
distinction the second-conversation event already draws cleanly.

**Un-sharing** is C10's delete (its cascade already demotes edges only that source
supported). No separate un-share verb.

**Mechanism.** A second scope key, because the text/knowledge split cannot ride one:

* `scope["exclude_conversation_ids"]` — text exclusion (episodic legs), as today.
* `scope["exclude_knowledge_conversation_ids"]` — the deny-set source for
  codex/procedural. **Absent ⇒ falls back to `exclude_conversation_ids`, byte-identical
  to C6's shipped behavior.** `resolve_retrieval_scope` sets it explicitly only when
  documents are in play: user exclusions + unenabled *unpromoted* docs, i.e. **promoted
  docs are excluded from the text but NOT from the deny set**.

### D5 — Incognito conversations cannot enable documents. (derived, C6 invariant)

`none` scope means "reads only itself" and C6 enforces it before anything else. Rather
than carve an exception, the service **refuses** the enable with a clear error naming
the fix ("incognito conversations read only themselves — change this conversation's
scope to use documents here"). Precedent: E1 already refuses project attachment for
incognito conversations. Documents uploaded *from* an incognito conversation are
refused for the same reason, in the same place.

### D6 — The doc-vs-transcript fork is a first-class choice, not a sniff. (user, 2026-07-28)

> *"the user can paste an actual big doc, OR it's just a massive info/context … or a
> conversation in txt format … there should be some sort of way to know if it's an
> actual doc or a conversation"*

A blob is one of two things and they have different destinations:

* **document** → parse → section → document conversation (`kind='document'`);
* **transcript** → F14's amnesia slicer (`slice_raw_text`) → ghost conversation
  (`kind='transcript'`), turns with real roles, `ts_provenance='synthetic_raw_import'`.

**An explicit `kind` argument always wins** (the UI choice, F-owed; the CLI flag; the
REST/MCP field). When absent, `detect_blob_kind()` asks the **background model** over a
head+middle+tail sample: *"Is this a transcript of a dialogue between a person and an
AI assistant, or is it a document?"* — a judgment about **content**, not typography.

⚠ **This replaces a G28 permanently-wrong-record bug.** Today the same question is
answered by `post_flight.py:171` — `raw_text.count("Assistant:") >= 3` — which is wrong
in both directions (paste a transcript that says "Bot:" and a 5,000-word dialogue is
filed as a document; write a 2,001-word prose answer and it is filed as a document
too). G28 keeps the general sweep; C12 owns this call site because C12 rewrites it.

⚠ **The same bet is already live in the import path and nobody had noticed:**
`formats.detect_format` routes **every** `.txt` to the raw slicer — i.e. it assumes any
unformatted text file is a conversation. `.txt` now routes through `detect_blob_kind`
too, with the explicit format argument still winning.

⚠ **G28 acceptance applies to the replacement.** `detect_blob_kind` is an inference,
so it owes the paraphrase-invariance evidence G28 demands: the same content, reformatted
(role labels stripped/renamed, casing changed, whitespace mangled), must yield the same
kind. Check 14 in §5 is that test in miniature; G28's probe set inherits it.

### D7 — A promoted paste never rewrites what the user wrote. (derived)

When a huge in-chat paste is promoted to a document/transcript conversation, **the
original chat turn stays verbatim**. It is marked (`promoted_document_id`), skipped by
the chunker, and excluded from retrieval as a duplicate — the promoted conversation is
the retrievable copy. The alternative (replace the turn's `raw_text` with a marker) was
rejected: mutating stored user content to save storage is the wrong trade in a system
whose entire proposition is that it remembers accurately.

### D8 — The RAG leg dies; `rag_documents` becomes the registry. (user, 2026-07-28)

`_rag_lookup` and `rag_chunks` are deleted. `rag_documents` is rebuilt as the
**document registry** (`documents`) — needed regardless for de-duplication, citation
provenance, the enable/disable toggles, and the F-owed library UI.

What is being deleted, precisely, so the record is honest about what it was:

* **the leg had no live writer** — `drop_zone.py` is a standalone `watchdog` script that
  nothing starts (`./ice` launches uvicorn only; C7 deleted Celery and drop_zone was
  never moved into `runtime.JOBS`), and it is the one module excluded from the smoke
  import sweep;
* **it was gated on five English nouns** — `orchestrator.py:1933`, an inline list
  (`document/pdf/reference/manual/guide`), a G28-catalogued style bet: *"what does the
  spec say"* retrieved nothing;
* **it ignored scope entirely** — a global `SELECT … FROM rag_chunks` with no
  conversation, project, or exclusion filter: the same leak class G29 found in three
  legs and C6 fixed in a fourth;
* **it was never measured** — the "vector_rag" rows in the Exp-2 reports are the
  *baseline system*, not this leg.

**There is no replacement gate.** Document content is memory; it is retrieved by the
same vector/BM25/codex/cluster legs as everything else, so no lexical trigger exists to
be style-dependent. This is the C12-side answer to G28's third bullet.

`drop_zone.py` is rewritten as a small `ingest_folder` runtime job (scan `ingest_inbox/`,
call the service, move to `processed/`) — no watchdog thread, no second classifier
instance, which **closes G13**.

### D9 — Formats: everything that is a pure-python wheel; nothing that needs a system binary. (user: "as much as possible … if it's obscure don't do it")

**In:** PDF (`pypdf`), DOCX (`python-docx`), PPTX (`python-pptx`), XLSX (`openpyxl`),
CSV/TSV (pandas — already a dependency), HTML (`beautifulsoup4`), and everything
text-native read directly (`.txt .md .rst .org`, and source files —
`.py .js .ts .java .c .cpp .h .go .rs .rb .php .sh .sql .yaml .toml .ini .json .xml .css`).

**Out, with a loud error naming what IS supported:** legacy `.doc`, `.rtf`, `.odt`,
`.epub`, images. Obscure relative to the work.

**Scanned PDFs (no text layer): detected and refused, not silently ingested empty** —
`"no extractable text (N pages, 0 characters) — this looks like a scan; OCR is not
enabled"`.

### D10 — OCR is a pluggable engine, deferred to Track F. (user, 2026-07-28)

Open WebUI does not OCR in-process: it exposes a **content-extraction engine** setting
whose non-default options (Apache Tika, Docling) run as separate Docker containers, and
those do the OCR — Tika's image bundles Tesseract. ICE copies the shape:
`extract_text(path)` dispatches on `settings.document_extraction_engine`
(`"builtin"` default; `"tika"` / `"docling"` reserved), so adding OCR later is a new
branch plus a compose service — **no system binary in `setup.sh`, no fight with the
packaged-app end-state.** The engine item lands in Track F (which already carries real
backend work: F5's SSE events, F7's search, F13's `session_replays`).

### D11 — Tabular files are ingested as a DESCRIBED table, not as rows. (derived)

A CSV/XLSX becomes: one **description** section (filename, sheet, columns with inferred
dtypes, row count, per-column stats — min/max/mean for numerics, top values for
categoricals, null counts) plus bounded **sample** sections (head rows + a deterministic
sample, rendered as markdown tables, capped by `DOC_TABLE_SAMPLE_ROWS = 50`).

Rejected: chunking every row group. Answering *"what was the churn rate in March"* from
100k rows is a **data query** — SQL/pandas territory — not memory retrieval, and dumping
the rows in buys noise rather than capability. Recorded as a seam: a future tool-use
item can query the file; the registry keeps `source_path`.

### D12 — Split: C12a (documents) then C12b (paste promotion + transcripts).

C12a is self-contained and unblocks E10. C12b depends on C12a's pipeline existing. Both
are in C12's original scope; neither is a throwaway. Precedent: A9 → A9a/A9b/A9c.

---

## 2. Algorithm & data model

### 2.1 Schema (one migration, head after `a1f6b8d94c22`)

```sql
-- conversations gains a kind: 'chat' (default) | 'document' | 'transcript'
ALTER TABLE conversations
  ADD COLUMN kind TEXT NOT NULL DEFAULT 'chat';
CREATE INDEX ix_conversations_kind ON conversations (kind) WHERE kind <> 'chat';

-- rag_documents -> documents (rebuilt; rag_chunks dropped)
DROP TABLE rag_chunks;
DROP TABLE rag_documents;

CREATE TABLE documents (
    id               UUID PRIMARY KEY,
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    filename         TEXT NOT NULL,
    file_type        TEXT NOT NULL,          -- 'pdf' | 'docx' | ... | 'txt'
    kind             TEXT NOT NULL,          -- 'document' | 'transcript'
    origin           TEXT NOT NULL,          -- 'upload'|'paste'|'watch_folder'|'project'
    sha256           TEXT NOT NULL UNIQUE,   -- de-dup: same bytes = same document
    byte_size        BIGINT NOT NULL,
    n_sections       INTEGER NOT NULL DEFAULT 0,
    page_count       INTEGER,                -- NULL where meaningless
    source_path      TEXT,                   -- D11's seam; NULL for pastes
    project_id       UUID REFERENCES projects(id),
    knowledge_shared BOOLEAN NOT NULL DEFAULT FALSE,   -- D4 latch
    shared_at        TIMESTAMPTZ,
    status           TEXT NOT NULL DEFAULT 'pending',  -- pending|ingesting|ready|failed
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE document_links (
    document_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    enabled          BOOLEAN NOT NULL DEFAULT TRUE,     -- D3 live toggle
    first_enabled_at TIMESTAMPTZ NOT NULL DEFAULT now(),-- D4 latch evidence
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, conversation_id)
);
CREATE INDEX ix_document_links_conversation ON document_links (conversation_id)
    WHERE enabled;

-- D7: a promoted paste points at what it became; the turn itself is untouched.
ALTER TABLE episodic_memory ADD COLUMN promoted_document_id UUID REFERENCES documents(id);
```

`document_links` rows are **never deleted on disable** — `first_enabled_at` is the
latch's evidence, so "has any second conversation ever enabled this" is
`SELECT count(*) FROM document_links WHERE document_id = :d` ≥ 2.

`ts_provenance` gains a third value: `'document_ingest'`.

### 2.2 Parse → sections

```
parse(path|blob) -> DocumentParse(text_blocks: [(text, meta)], page_count, file_type)
```
one parser per family in `src/ingestion/documents/parsers.py`, each returning blocks
with structural metadata (`page`, `heading`, `slide`, `sheet`). Then:

```
sections = []
for block, meta in blocks:
    for i, piece in enumerate(chunk_text(block, max_tokens=CHUNK_TOKENS,
                                         overlap_words=OVERLAP_WORDS)):
        sections.append(Section(text=piece, meta=meta, index=len(sections)))
```

`chunk_text` is **the shared C2 chunker** (`src/memory/chunking.py`) — markdown headings
are already hard boundaries there (C3 point 3), which is exactly the section semantics
wanted. Do not write a second splitter.

Each section's stored `raw_text` is prefixed with one compact provenance header:

```
[report.pdf · §Methods · p.12]
<section text>
```

Deliberate: the header rides *inside* `raw_text` so BM25 finds `report.pdf` (users do
ask "what did the spec say"), the assembler renders it with no change, and `reembed.py`
reproduces it deterministically. `prompt_assembler.py:113` and `reembed.py:67` both
already fall through cleanly when `raw_text` does not start with `"User: "`.

### 2.3 Replay

```python
ingest_document(db, parse, doc_row, *, classifier, embedder, llm, deadline) -> report
```
lives in `src/ingestion/documents/ingest.py` and mirrors `import_conversations`'s loop —
`_store_turn`-equivalent per section, then `evaluate_turn(..., source_kind="document")`,
then `run_cluster_assignment(db, conversation_ids=[doc_conv])`, then
`run_conversation_summaries(...)` for the document-level summary. Timestamps: file mtime
(else upload time) + `i` seconds, so section order is stable and `resolve_session_id`
puts the whole document in one session (C5 session affinity then clusters its sections
together — the desired behavior, for free).

### 2.4 `evaluate_turn` gains `source_kind`

```python
def evaluate_turn(batch_id, prompt, response, conversation_id,
                  model_used="", source_kind="chat"):
```
Two behavior changes, both gated on `source_kind == "document"`:

1. `lossless = True` — **forced**, because C12's entry requires a document's facts to
   reach the graph ("a PDF's people/systems/facts become first-class graph nodes"), and
   the existing gate is `has_code or is_creative or entropy >= threshold` with
   `has_code = "```" in response` and `response == ""` for a section. Forcing the flag is
   correct rather than a hack: the density gate exists to protect codex from
   conversational chatter, and a document is not chatter.
2. the summary system prompt switches to the document variant ("Summarize the following
   excerpt from the document *<title>*") instead of "user/assistant exchange".

### 2.5 Scope resolution (`resolve_retrieval_scope`, `src/services/scoping.py`)

Appended after C6's existing exclusion block, and **only when documents exist**:

```
enabled  = {doc conv ids enabled for this conversation}
all_docs = {every conversation id where kind <> 'chat'}
unshared = {doc conv ids whose documents.knowledge_shared is FALSE}

if scope has "conversation_ids":            # closed set (manual/project/attached)
    scope["conversation_ids"] += enabled - excluded
else:                                       # auto — open read
    scope["exclude_conversation_ids"] += (all_docs - enabled)

# D4: promoted docs lose text visibility but keep knowledge visibility.
scope["exclude_knowledge_conversation_ids"] = (
    user_exclusions | ((all_docs - enabled) & unshared))
```

`_resolve_denied_batches` (`orchestrator.py:1346`) reads
`exclude_knowledge_conversation_ids` **if present, else `exclude_conversation_ids`** —
so every existing C6 path is unchanged.

---

## 3. Files & integration points

**New**
* `src/ingestion/documents/__init__.py` — empty (no barrel re-exports, standing rule).
* `src/ingestion/documents/parsers.py` — per-format parsers + `detect_file_type`.
* `src/ingestion/documents/extract.py` — `extract_text()` engine dispatch (D10).
* `src/ingestion/documents/ingest.py` — sectioning + the replay loop.
* `src/ingestion/documents/kind.py` — `detect_blob_kind()` (D6).
* `src/services/documents.py` — E0 service: `add_document`, `list_documents`,
  `set_document_enabled`, `document_status`, `delete_document`.
* `tests/test_documents.py` — the live-DB behavioral suite (§5).

**Modified**
* `src/memory/models.py` — `Conversation.kind`, `Document`, `DocumentLink`,
  `EpisodicMemory.promoted_document_id`; `RAGDocument`/`RAGChunk` deleted.
* `src/services/scoping.py` — `resolve_retrieval_scope` per §2.5.
* `src/retrieval/orchestrator.py` — `_rag_lookup` **deleted** (leg dict, weight tables,
  `configurable_orchestrator.py` override, the `source_type` docstring);
  `_resolve_denied_batches` reads the new key with the old fallback.
* `src/workers/post_flight.py` — `source_kind` param; document summary prompt.
* `src/workers/drop_zone.py` → `src/ingestion/documents/watch_folder.py` (G13).
* `src/workers/runtime.py` — `JOBS`: `ingest_document` (gpu lane), `ingest_folder` (cpu).
* `src/api/routers/user_control.py` — `/user-control/documents` CRUD + enable toggle.
* `src/mcp/server.py` — `ice_control` document actions.
* `src/memory/reembed.py`, `src/memory/portability.py` — `rag_chunks` rules removed,
  document sections registered.
* `pyproject.toml` — `pypdf`, `python-docx`, `python-pptx`, `openpyxl`, `beautifulsoup4`.
* `scripts/ice_add_document.py` — CLI.

**C12b (second commit)**
* `src/workers/post_flight.py` — promotion branch (D6/D7).
* `src/ingestion/formats.py` — `detect_format`'s `.txt` assumption routed through
  `detect_blob_kind`.

---

## 4. Edge cases & failure modes

1. **Same file uploaded twice** — `documents.sha256` unique: return the existing
   document and enable it in the requesting conversation. Never re-ingest.
2. **Scanned/empty PDF** — refuse with the D9 message; `status='failed'`, error stored.
3. **Encrypted PDF** — refuse loudly; do not store a half-document.
4. **Enormous document** — the same gpu-lane slice/deadline discipline as
   `run_import_replay`; a dry-run estimate (`SECONDS_PER_TURN × n_sections`) is returned
   before ingestion starts, as `start_import` already does.
5. **Crash mid-ingest** — `status='ingesting'` with a stale heartbeat is reaped exactly
   like `ImportRun` (`_reap_stale`); per-section idempotency keys dedupe the re-run.
6. **Enable on an incognito conversation** — refused (D5).
7. **Document deleted (C10)** — `ON DELETE CASCADE` from `documents` → the doc
   conversation's turns/chunks/links; C10's existing edge-demotion handles the graph.
8. **A conversation deleted while it has documents enabled** — `document_links` cascades;
   the document survives (it is its own conversation).
9. **Enabled doc + `manual` scope + the doc also in `excluded_conversation_ids`** —
   exclusion wins (C6's rule: "an exclusion you set is an exclusion that holds").
10. **Legacy `rag_chunks` rows** — the store is empty (standing answer); the migration
    drops the tables outright. Anything in an old backup restores under G23's importer,
    which walks `Base.metadata.sorted_tables` and will simply not see them.
11. **A section that is itself larger than the chunk cap** — the existing
    `> LONG_TURN_CHUNK_WORDS` rule in `post_flight` chunks it; nothing special.
12. **Zero-section parse** (empty file) — refuse, `status='failed'`.

---

## 5. Validation checklist — `tests/test_documents.py`, live DB, own rows, no TRUNCATE

Every scope check is **two-sided** (the in-scope row IS returned *and* the out-of-scope
row is NOT), and the negative side must not be vacuous — trap 5 from the C6 session.

1. `.md` upload → a `kind='document'` conversation exists with N section turns, header
   prefix present, `documents.status='ready'`, `n_sections=N`.
2. Sections carry `ts_provenance='document_ingest'`, one shared `session_id`, ascending
   timestamps.
3. Codex ran: entities extracted from the document exist with the doc's batch ids
   (stub LLM) — and would NOT have run under the old gate (assert `lossless_flag` is
   true on a low-entropy section, i.e. the forcing is load-bearing).
4. Clustering **processes** the doc's sections like any other turn (`processed ==
   len(sections)`) and they are eligible (not private). *Not* "a cluster formed" —
   see the STATUS divergence at the top.
5. A document-level summary row exists in `conversation_summaries` — the fixture must
   be a REAL-sized document (>6,360 tokens), because the worker correctly declines to
   summarise a conversation that still fits the recent window.
6. **Isolation, two-sided:** conversation A (uploader) retrieves a doc-only fact;
   conversation B under `auto` does **not**; enable in B → B does; disable → B does not.
7. **Closed-set mode:** the same, with B in `manual` scope (proves the
   `conversation_ids` union arm, not just the exclusion arm).
8. **Knowledge latch:** with the doc enabled only in A, a doc-exclusive entity is denied
   to B; enable in B (latch trips, `knowledge_shared` true); disable in both; B still
   gets the *entity* but still NOT the section text.
9. **Shared-entity rule:** an entity evidenced by both the doc and a normal chat turn is
   never denied (C6's every-event rule, re-asserted here).
10. Incognito conversation: enable is refused with the D5 error.
11. Re-upload of identical bytes: no second document, no second ingest, link created.
12. Scanned-PDF stand-in (a PDF with no text layer): refused, `status='failed'`, no
    conversation created.
13. CSV: description section names every column; sample sections capped.
14. **`detect_blob_kind` invariance (G28):** one transcript written three ways
    (`User:/Assistant:`, `Me:/Bot:`, no labels at all) yields `transcript` for all three;
    a document with the literal word "Assistant:" in its prose yields `document`.
    Assert the *old* rule disagrees on at least one, so the fix is proven load-bearing.
15. **The RAG leg is gone:** `_rag_lookup` no longer exists and the five-noun list
    appears nowhere in `src/`.
16. Full regression sweep at baseline, `test_session_scoping` especially (its 40 checks
    cover the resolver this spec extends) — and run the WHOLE file, since new rows
    perturb existing checks (trap 5).

**Cleanup order** (trap 6, learned in C6): links → chunks → turns → procedural/clusters
→ documents → conversations, then `SELECT count(*)` to prove the store is clean. Do not
assume `finally` won.

---

## 6. Look-ahead constraints

* **E10 (docs-for-coding)** — its entire backend is `add_document(..., origin='project',
  project_id=…)`. The doc conversation gets `project_id`, so C6's project arm scopes it
  with zero extra code. E10 becomes claimable the moment C12a lands.
* **E4** — registration bootstrap ingests README/docs through this same call.
* **Track F** — owes: the document library + per-chat enable toggles, the upload
  doc-vs-transcript choice, the import wizard (export-format pick vs raw-txt → amnesia),
  and the OCR engine (D10). All four appended to the F ledger this session.
* **C13 (caching)** — `resolve_retrieval_scope` now runs two small document queries per
  request. That is the first *per-request* query C13 should look at; it is invalidated by
  document create/enable/disable only, which is a clean invalidation key.
* **G29** — the token-estimation cluster includes `drop_zone.py:81`; that site dies here.
  `documents/*` must import `estimate_tokens` from `memory/chunking.py`, never re-derive.
* **G30** — this suite is the first test of a full ingest→retrieve loop with a
  non-constant embedder; it does not fix G30's blind spots but must not add to them.
* **B6** — branch import stays F10's territory; documents have no branches.

---

## 7. Traps

1. **Do NOT set `is_document=True` on a section.** It looks right and it makes the
   section **invisible**: the vector leg excludes `is_document` rows
   (`orchestrator.py:997`) on the assumption that chunks will be searched instead, but a
   section is already chunk-sized, so `run_chunk_turn` returns 0 chunks
   (`len(chunks) <= 1`) and nothing represents the row anywhere. Sections are ordinary
   turns in a conversation whose `kind` marks it; visibility comes from `kind`, never
   from `is_document`.
2. **Do NOT write a second chunker.** `chunk_text` already does word boundaries,
   overlap, code awareness, and markdown-heading hard breaks. C2/C3 paid for it.
3. **Do NOT build a new "document retrieval leg."** That is the mistake being deleted.
   The moment a leg exists, something must decide when to run it, and that decision
   becomes a lexical gate again.
4. **Do NOT derive the deny set from `exclude_conversation_ids` unconditionally** once
   D4 lands — a promoted document is excluded from the text and must NOT be denied its
   knowledge. The fallback exists so C6's paths stay byte-identical; the new key is the
   documents path.
5. **Do NOT delete `document_links` rows on disable.** The row *is* the latch evidence.
6. **Do NOT let the enable toggle be a write to `included_conversation_ids`.** It looks
   like the same thing and is not: that field is user-authored `manual` scope, and
   documents must work identically in `auto`, where that field is not read at all.
7. **Do NOT parse the blob to guess doc-vs-transcript.** Every cheap test is a
   typography bet (D6). The model reads the content or the user says so.
8. **Do NOT skip the dry-run estimate.** A 300-page PDF is ~150 LLM extraction calls;
   users deserve the number before it starts, exactly as F10 gives it.
