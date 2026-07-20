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

## [rev 2026-07-20] Re-grounding notes (pre-implementation, rule 12)

Grounded against HEAD `2cebde4` (post G23/C17). Sixteen refinements; where
they contradict §1–§6 above, THESE win:

1. **`apply_decay(cycles)` cannot be D3's fast-forward mechanism.** The live
   function (src/workers/decay.py) clamps cycles to [1, 96] (≈6 days) and
   applies ONE uniform rate to every eligible row — per-turn ages are
   impossible through it. The importer instead computes each turn's score
   closed-form at insert, **deriving daily rates from decay.py's own
   constants** (`DECAY_RATE_UNACCESSED ** CYCLES_PER_DAY` ⇒ 0.95/day;
   creative 0.99/day with the 0.3 floor), then applies `is_archived` when
   score < ARCHIVE_THRESHOLD. `apply_decay` itself is untouched except for
   refinement 2.
2. **`decay_immune` is a permanent boolean owned by bookmarks** — no window
   semantics exist. New nullable column `episodic_memory.decay_immune_until`
   (TIMESTAMPTZ); decay.py's three decay UPDATEs additionally filter
   `(decay_immune_until IS NULL OR decay_immune_until < :now)` — the window
   self-expires, no sweeper job. Bookmarks keep the boolean.
3. **NEW default policy `hybrid` (USER decision, 2026-07-19):** per-turn age
   threshold RECENT_DAYS=30. Turns ≤30d old at import → preserve semantics
   (score 1.0 + 14-day `decay_immune_until`); older turns → fast-forward with
   the aging counted from the threshold (`age_days − 30`), giving a smooth
   ramp instead of a cliff at day 30. A month-old chat arrives fresh, a
   year-old one arrives aged, a long-running one gets an aged head + fresh
   tail. The D3 trio stays selectable; `preserve` is no longer the default.
4. **Import never cold-moves.** Fast-forwarded scores floor at
   COLD_THRESHOLD (0.05): the importer must not delete rows it just created
   (idempotency + T3 own cold transitions; the next natural decay cycle takes
   truly-dead rows cold through the normal machinery).
5. **DeepSeek is a fourth adapter** (user has a real DeepSeek export).
   Format: ChatGPT-style mapping tree (`{id, inserted_at, mapping, title,
   updated_at}`, node `{id, parent, children, message}`) but **no
   `current_node`** and message content in `fragments[]` with types
   REQUEST/RESPONSE/THINK/SEARCH/TOOL_OPEN/TOOL_SEARCH. Role from fragment
   type (REQUEST=user, RESPONSE=assistant); THINK/SEARCH/TOOL_* dropped.
   Path selection at branches: follow the child whose subtree has the latest
   `inserted_at` (latest-edit-wins); skipped branch messages counted.
6. **Claude exports also branch.** `chat_messages` is ALL messages incl.
   abandoned edit-branches (18/68 real conversations); current path = walk
   `parent_message_uuid` up from the latest-created leaf. 430/1472 real
   messages have empty `text` (tool/attachment-only) — fall back to joined
   `content[]` text blocks (skip thinking/tool blocks), drop if still empty.
   Non-empty `text` is authoritative (it renders fuller than block joins).
7. **Real exports in `data/simulation/raw_chats/` are format reference
   ONLY** (user 2026-07-19): never real-imported this session, never
   committed as fixtures — fixtures are tiny synthetic files mimicking each
   shape. ChatGPT JSON adapter is built from documented shape (mapping +
   `current_node`, unix-float `create_time`, `author.role`,
   `content.parts`); the user's proper ChatGPT export arrives later — their
   existing gpt*.txt dumps are F14 material at best.
8. **Per-turn idempotency_key excludes the run id:**
   `sha256("ice-import:{conversation_id}:{turn_index}:{pair_hash16}")`.
   A per-run component would break D4's cross-run resume (new run ⇒ new keys
   ⇒ duplicates). The conversation_id itself is deterministic:
   `uuid5(NS_ICE_IMPORT, "{provider}:{source_conversation_id}")`.
9. **"Runs in the gpu lane" is realized as a self-re-enqueueing sliced job**
   `import_replay` (JOBS entry, gpu lane, kwargs `{import_id, seq}`): each
   dispatch processes conversations until a ~10-min slice budget expires or
   a live generation appears, then re-enqueues `seq+1` and returns — the
   lane frees between slices so live post_flight never starves behind an
   hours-long import, and `_gpu_ready(for_event=True)` gates every
   resumption (the C7 yield). Mid-slice, the engine additionally pauses
   between turns while `generation_in_flight`. Inner stages are DIRECT
   calls: `evaluate_turn` (which itself chains chunking/codex/procedural per
   C7), `run_cluster_assignment(db, conversation_ids=[cid])` per finished
   conversation, `batch_summarize()` + `run_conversation_summaries(db,
   conversation_ids=…)` at finalize.
10. **Persistent state = two new tables.** `import_runs` (id, source_path,
    source_format, policy, kind='replay', status running|completed|failed|
    aborted, totals/counters, timestamps, error, report JSONB) and
    `import_conversations` (content_hash PK, import_id, conversation_id,
    title, n_turns, imported_at) — the hash-skip ledger written only after a
    conversation fully replays; a mid-conversation kill re-runs it and the
    per-turn keys dedupe. One import at a time (a fresh `running` row
    blocks; stale >10 min heartbeat ⇒ auto-mark aborted and proceed).
    C10 note: deleting an imported conversation leaves its hash tombstone —
    re-imports do NOT resurrect it; that is the honest reading of a
    user-initiated forget.
11. **Provenance column, not codex vocabulary:** new
    `episodic_memory.ts_provenance` TEXT NOT NULL DEFAULT 'original'
    ('original' | 'synthetic_raw_import'). F10 turns keep 'original' (their
    timestamps are real); F14 turns get 'synthetic_raw_import'. Codex
    `source` vocabulary (G17/E1b) untouched — T-timelines caveat via the
    batch→turn join.
12. **F14 mechanics pinned:** slices via the shared `chunk_text` (C2 greedy
    packer) with max_tokens ≈ 2,667 (≈2,000 prose words) and
    overlap_words=200; per-slice turn extraction (bg LLM, JSON) then ONE
    seam call per adjacent pair (model sees A's tail turns + B's head turns
    + the raw overlap once, returns the corrected boundary turns);
    normalized-alnum-hash dedup at the end; synthetic timestamps END at file
    mtime spaced 1/min backwards (a file's mtime marks when its conversation
    ended); title = filename. `.txt`/unparseable inputs route to this path.
13. **Message pairing:** consecutive same-role messages merge (`\n\n`);
    pairs are (user → assistant); a trailing user message stores with an
    empty assistant half; leading orphan assistant messages are skipped and
    counted in the report.
14. **Adapters land in the E0 world:** engine under `src/ingestion/`
    (importer.py, formats.py, raw_slicer.py) + a thin service
    `src/services/ingestion.py` (`start_import`, `import_status`) that REST
    (`POST /user-control/import`, `GET /user-control/import[/{id}]`) and
    `ice_control action="import_status"` both adapt — parity per E0.
    `start_import` supports dry_run (parse + count + estimate only; the D5
    cost note uses ~6 s/turn).
15. **Stored-turn parity with `store_turn_async`:** raw_text exactly
    `"User: {u}\n\nAssistant: {a}"` (reembed RULES' `_episodic_source`
    parses that shape), embedding = user half only via
    `src/memory/embedder.py::get_embedder()`, tags/context_reliance from the
    live classifier when available (engine takes injectable classifier/
    embedder/llm à la `run_conversation_summaries`; stub default reliance =
    `Zero_Shot`), sessions via `resolve_session_id(db, cid, turn_ts,
    settings.session_gap_minutes)` — original timestamps make gap-derived
    historical sessions come out free, in replay order.
16. **Validation adaptation:** check 4's fast_forward assertion becomes
    `score == 0.95**(age_days)` (per-day closed form, refinement 1) with the
    COLD floor case checked separately; a hybrid case (45d ⇒
    `0.95**15`) and a decay.py immunity-window regression (immune row
    survives `apply_decay`, expired window decays) are added; check 2 runs
    the REAL `evaluate_turn` with module-attr LLM stubs (house pattern:
    `codex_extractor.extract_triplets = …`), `batch_summarize` skipped via
    flag (global-sweep machinery, tested elsewhere).

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
