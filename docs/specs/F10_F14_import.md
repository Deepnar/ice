# F10 + F14 — Conversation import (LSREP as product) · Raw-log extraction v2

Assumes decided specs: `FINAL_experiments.md` (its lme/synth replay adapters and
this importer are ONE ingestion engine — build it as product code under
`src/ingestion/`, experiments consume it), `C7_scheduling.md` (import runs as a
runtime job with progress events), `T_temporal.md` (original timestamps are
sacred — the exocortex needs real dates), `C4_C9` (imported conversations get
summaries at the end of the run), `E0` (importer is a service; F's upload UI is
an adapter). Grounded: LSREP replay machinery in `experiments/{unmature,mature}`
runners; `resolve_session_id` gap logic; the F10 roadmap entry + rough-notes
decay policies (default: imported turns start `decay_score = 1.0`, decay runs
forward from import).

## 1. Decisions

- **D1: one ingestion engine, three front-ends.** `src/ingestion/importer.py`:
  `import_conversations(db, runtime, source: Iterable[NormalizedConversation],
  policy) -> report`. Front-ends: format adapters (D2) for user exports, FINAL's
  lme/synth adapters, F14's raw-log slicer. The engine *replays*: for each turn
  pair in timestamp order — store (with ORIGINAL timestamp + gap-derived
  session_ids) → post-flight chain (density/summary) → codex/procedural
  extraction → clustering; batch summaries + C4 conversation summaries at the
  end. The system "lives through" the history; the result is mature memory, not
  an archive.
- **D2: format adapters** (`src/ingestion/formats.py`), each →
  `NormalizedConversation {title, turns: [{role, text, ts}]}`: ChatGPT
  `conversations.json` (mapping-tree flattened along `current_node` parents),
  Claude export (`conversations.json` chat_messages), generic JSONL
  (`{role, content, timestamp}` per line — the documented escape hatch).
  Unknown structure → error naming the three supported shapes. Adapters are
  pure and unit-tested against committed fixture files.
- **D3: decay policy trio (roadmap semantics, settled):**
  `preserve` (default) — turns enter at `decay_score=1.0` with a
  `decay_immune` window of 14 days (memory feels fully present, then earns its
  keep); `fast_forward` — after replay, run `apply_decay(cycles = simulated
  cycles since each turn's original date)` in one closed-form pass (C7's cycles
  param — old imported memories immediately feel their age; T3's archive/cold
  tiers apply); `fresh` — decay_score=1.0, no immunity, forward-only.
- **D4: idempotent + resumable per conversation.** Each import gets an
  `import_id`; a conversation's normalized-content hash is its idempotency key —
  re-running an export skips already-imported conversations (report says so).
  Progress: structlog `import_progress` (F5/SSE candidate) per conversation;
  kill-safe resume via the hash skip.
- **D5: imported scope + privacy:** conversations arrive `memory_scope_type=
  'auto'`, `is_private=False`; the report reminds the user they can re-scope.
  Bg-model cost note printed up-front (a 1k-conversation export is hours of
  extraction on the laptop — the C7 gpu lane paces it; USER-REQUIRED below).
- **D6 (F14): raw unformatted dumps (no roles, no timestamps).**
  `src/ingestion/raw_slicer.py`: (1) slice at word boundaries into ~2k-word
  windows with 200-word overlaps (shared chunker's greedy packer, C2); (2)
  **one open bg-model session per adjacent pair** resolves the overlap — the
  model sees slice A's tail + slice B's head + the overlap ONCE and returns the
  boundary cut + speaker/turn guesses (replaces the amnesia-method cold
  chunks); (3) turns get synthetic timestamps (file mtime, spaced 1/min) and a
  clearly-synthetic session; (4) dedup at the end (normalized-text hash across
  the import). Output = a NormalizedConversation into D1's engine. Honest
  labeling: entities/turns from raw dumps carry `source='raw_import'` in
  provenance so T-timelines can caveat their dates.
- **USER-REQUIRED (rule 11):** (a) export their history from the provider
  (ChatGPT: Settings → Data controls → Export; Claude: Settings → Export data;
  arrives by email as zip — done = the json file exists locally); (b) choose
  the decay policy at import (one prompt; default preserve); (c) leave the
  machine on for large imports (estimate printed: ~N_turns × bg-model seconds).
- **Empirical deferral:** F14's overlap-resolution quality — first real dump is
  the test; rule: if boundary errors are visible in >10% of spot-checked seams,
  raise overlap to 400 words before touching the prompt.

## 2. Files & integration points

`src/ingestion/{importer,formats,raw_slicer}.py` (+ FINAL's adapters relocate
here when built) · runtime job registration (import runs in the gpu lane,
yields to the user per C7's gates) · REST adapter `POST /user-control/import`
(file path + policy; the drag-drop UI is F's — ledger) + `ice_control action=
import_status` · fixtures: one mini export per format ·
`tests/test_ingestion.py`.

## 3. Edge cases

Branched ChatGPT trees → current_node path only (other branches counted +
reported, not imported — B6 territory). Missing/garbage timestamps in an export
→ per-conversation monotonic synthesis from whatever anchors exist; flagged in
the report (T-honesty). Mid-import kill → resume skips hashed conversations;
the partially-imported one re-runs (its turns dedupe on idempotency_key).
Enormous single conversations → C2/C3 chunking applies as normal (documents
path for pasted blobs). Non-English/emoji-heavy → no special handling
(embedder is multilingual); noted.

## 4. Validation checklist — `tests/test_ingestion.py`

1) each format fixture → identical NormalizedConversation (golden files);
2) mini import end-to-end (stub LLM): turns stored with ORIGINAL timestamps,
session_ids gap-derived, codex/procedural/cluster rows appear, C4 summary
written; 3) re-run → 100% skipped by hash, zero duplicates; 4) `fast_forward`:
a 90-day-old turn's decay_score equals `rate**cycles(90d)` exactly; `preserve`:
immune for 14 days; 5) F14: fixture dump → slices at word boundaries, overlap
session resolves seams (stubbed), dedup removes the planted duplicate,
synthetic-time provenance marked; 6) import_progress events emitted per
conversation.

## 5. Look-ahead constraints

E4's git-log replay cites this engine (repo history as NormalizedConversation).
FINAL's adapters are consumers #2/#3 — if FINAL is implemented first, the engine
is built THERE and this spec's D1 relocates it under `src/ingestion/` (either
order works; the close-out flow fixes the sequence). B6 owns branch import.

## 6. Traps

- Don't import as an archive (bulk-insert without the pipeline) — the entire
  point is living through it; an unprocessed dump is dead weight retrieval
  can't reason over.
- Don't rewrite timestamps to import-time — T-track dies; original dates are
  the product.
- Don't run both decay policies' bookkeeping "to be safe" — one policy per
  import, recorded in the report.
- Don't let F14 guess speakers without the overlap session — cold-chunk
  splitting is the exact failure v2 replaces.
