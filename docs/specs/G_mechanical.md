# G-track mechanical fixes — one-liners (S1 rule: no full specs for these)

Grounded at commit `31f9b12`. Items already folded into real specs are marked;
the rest carry exact fix + validation in one breath. Execute opportunistically
alongside the flow phases (top-of-roadmap section).

- **G2** → folded into C7 (delete dead SGLang block; shared default; bg model
  None ⇒ chat model).
- **G3** → parked by C7's shared-first (entry already says so; nothing to do).
- **G4** → folded into C7 (dedicated-only nvidia-smi, threshold 70, 10 s cache;
  shared mode gates on runtime flags).
- **G5 SSE resiliency:** in the stream parser, keep the undecoded tail of each
  chunk and prepend it to the next before splitting on newlines; a final
  unparseable tail at stream end is logged + dropped (never silently truncate
  `raw_text` mid-JSON). Validate: fuzz a fixture stream split at every byte
  offset → identical assembled text.
- **G6 indexes:** one Alembic revision importing the still-missing indexes from
  `scripts/database/create_indexes.sql` (minus those already added by
  T3/G23/E1 migrations — diff first); delete the orphan script after. Validate:
  `EXPLAIN` on batch_id lookup uses the index.
- **G7 idempotency:** the model declares `unique=True` but verify the LIVE
  table (`information_schema`) — early migration drift suspected; if missing:
  dedupe (keep oldest per key), then add the constraint in the G6 revision.
  Validate: duplicate insert raises.
- **G8** → folded into C7 D9 (conversations columns; no redis).
- **G9** → folded into Z1-prep D4 (the settings sweep is its first commit).
- **G10** → compaction scheduled by C7's job table (24 h, lossless per Track-T
  constraint); nothing else remains.
- **G11 batch-summarizer coverage:** extend the selection to
  `timestamp < now()-30d AND NOT summarized` regardless of decay state (old-
  but-undecayed turns in long conversations finally compress). Validate: seeded
  old undecayed turn gets a batch summary.
- **G12** → folded into C7 D7 (bg client timeout = `base × clamp(max_tokens/500,
  1, 6)`, retries with backoff under the gpu lane).
- **G13 drop-zone classifier:** `drop_zone.py` takes the embedder/classifier
  from `create_core()` instead of instantiating its own. Validate: one model
  load in logs during ingestion.
- **G14** → folded into C4_C9 D7 (service-enforced 300-token slot cap).
- **G15 noise routing:** in `memory_decision`, `Null_Noise`/`Casual_Banter` as
  the ONLY intent adds a negative bump (`ltm_bump_noise = −1.5`); B3's class
  selection already sends confident-simple to the small class. Validate: banter
  prompt skips retrieval.
- **G17 audit trail:** writers already stamp sources (T `description_updated`,
  D1 `agent_run_id`, E0 `updated_by`, C11 `chat_command`, G16 scope changes) —
  remaining work: `episodic_memory.write_source TEXT DEFAULT 'chat'`
  (import/resurrection/drop-zone set theirs), plus a `memory_audit` SQL VIEW
  unioning the journaled writers for F5/G23 export. Validate: view returns rows
  from three different sources on a seeded DB.
- **G19** → folded into FINAL D10 (audit-or-fold the configurable orchestrator
  before any run; prefer folding flags into the parent as settings).
- **G20 sweep verdicts:** `_hyde_used` flag → delete with the commented block
  (keep the relabeled `_hyde_rewrite` comment per P0.1); `conversations.
  custom_filter` → **DROP column** (C6's richer scope forms supersede it; never
  read today); `session_replays` → F13 owns (keep, note); redis publish + jsonl
  buffer → already deleted by C7; `entropy_score` → done (C1). Validate: grep
  gate for the deleted names.
- **G22 smoke suite (build FIRST in the flow):** pytest, seconds, no GPU/DB:
  all-worker-modules import (kills G21-class bugs), memory-decision math,
  chunker mechanics, timescope detector, budget arithmetic, dynamics
  invariants (Z1-prep D5's test lives here), config loads, `py_compile` sweep.
  Habit: run before every commit. Validate: suite <10 s, red on an injected
  bogus import.
- **G24** → folded into E0 D9 (to_thread at the adapter layer; audit main.py's
  direct `db.query` sites while extracting them into services).
- **G25 log privacy:** `settings.log_redact_content` (default False until the
  F-era flips it) → a structlog processor truncating `raw_text`/message payload
  fields to 80 chars + sha256 prefix; decisions/ids/telemetry untouched; C10's
  manifest already names logs as out-of-scope for deletion. Validate: redacted
  run's log contains the hash, not the text.
- **G26 (P0, FIRST fix of the whole flow):** move the conversation-resolution +
  scope block (main.py:270–303) ABOVE the `classifier.classify` call (~231) so
  `conversation_id` exists; keep classification before model selection (tags
  feed routing). Validate: one live request round-trips; CL7's context-prefix
  actually receives prior turns (log the prefix length).
- **G27 (new 2026-07-25 — shared-mode bg model resolves to the WRONG model; VRAM
  thrash on a 24 GB box):** `bg_client_factory.get_bg_model_name()` in shared mode
  (no `background_model_name` pin) returns `registry.get_fallback_model()`, which is
  *the first `confirmed: true` entry in `models/model_registry.json`* — today
  `gemma4:26b-a4b-it-q4_K_M`. That is **not** "the chat model currently in use", which
  is what shared mode is documented to mean (C7 D7) and what the user believes it
  does. With MoE routing live, chat may route to `qwen3-coder:30b-a3b` while background
  work fires on `gemma4:26b-a4b` ⇒ Ollama holds/swaps **two ~17 GB models on 24 GB
  VRAM**, thrashing on every background task. (`settings.default_fallback_model` =
  `qwen2.5:7b` applies only when NOTHING is confirmed, so it never rescues this case.)
  **Fix — decide between:** (a) resolve the *session's* routed model (SESSION_STATE
  stickiness), falling back to first-confirmed only when there is no session — true
  "shared" semantics; or (b) keep it deterministic and pin a small always-resident bg
  model (`BACKGROUND_MODEL_NAME=qwen2.5:7b`) so bg never competes with chat for VRAM.
  (b) is the safer default for FINAL (the bg model must be manifest-recorded and must
  not vary with routing); (a) is the better product behavior. They can coexist: (a) as
  the default, (b) as the documented pin. **Resolve BEFORE FINAL** — a bg model that
  silently varies with routing is a reproducibility hole in every run. Touches
  `bg_client_factory.py` (+ a config comment); validate with a routed-chat-vs-bg
  assertion.
