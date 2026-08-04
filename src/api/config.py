"""Configuration for the ICE FastAPI proxy."""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://ice:ice_local_dev@localhost:5432/ice_db"
    ollama_base_url: str = "http://localhost:11434"
    classifier_threshold: float = 0.3
    confidence_fallback_threshold: float = 0.75
    # The LIVE checkpoint. Renamed 2026-07-27: this path held a file called
    # ice_classifier_v3_qwen_ft3.pt that had contained a schema-v2 model since
    # B1's promotion — a name asserting "v3, qwen, fine-tune 3" while holding a
    # from-scratch v2 retrain. Harmless to the code (a checkpoint declares its
    # own schema_version) and actively misleading to a reader. The v1 model kept
    # its own honest name and is the rollback artifact.
    classifier_model_path: str = "models/classifier/ice_classifier_v4_schema2.pt"
    label_schema_path: str = "data/labeled/label_schema.json"
    default_fallback_model: str = "qwen2.5:7b"

    # ── G23/C17: store-level embedding identity (fail-loud) ──
    # The ONE embedder every writer and retrieval path shares
    # (src/memory/embedder.py). store_meta's 'embedding' row must agree with
    # these at boot — create_core() refuses to start on a mismatch, because
    # silently cosine-comparing mixed-width/mixed-model vectors is the
    # existential failure G23 exists to prevent. Changing either value
    # requires a migration + scripts/ice_reembed.py run. The classifier serves
    # at native width since B1; the micro-NER still consumes slice384() of the
    # same encode, as would a rolled-back v1 checkpoint (embedder.fit_width).
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024

    # ── C7 D7: shared-first background model ──
    # "shared" reuses the main Ollama LLM for background work (the default —
    # the maintenance runtime's gpu lane + idle gating make this safe);
    # "dedicated" is the power-user config: a separate OpenAI-compatible
    # server on port 8002 (started manually — ./ice no longer launches it).
    background_model_mode: str = "shared"
    # None ⇒ resolve the chat model (registry get_fallback_model) in shared
    # mode / the vLLM default in dedicated mode. Pin a specific model here to
    # override — e.g. a small always-available Ollama model (BACKGROUND_MODEL_
    # NAME=qwen3:4b-instruct) gives you a "dedicated" bg model with no second
    # server: Ollama spins it up on demand, the runtime's idle gating hides
    # the model swap. That replaces the old vLLM side-server for most setups.
    background_model_name: Optional[str] = None
    # G12: bg-client calls scale their timeout with the requested output size:
    # timeout = bg_timeout_base_seconds × clamp(max_tokens / 500, 1, 6).
    bg_timeout_base_seconds: float = 30.0
    # Ask the background model NOT to think. Measured 2026-08-03: every model in
    # the live registry except qwen3:4b-instruct is a reasoning model, and the
    # hidden reasoning block consumes the whole max_tokens budget before a single
    # token of the answer is written — so `content` comes back EMPTY and the
    # entire background layer (triplets, summaries, decisions, cluster names)
    # produced nothing. Raising the budget does not fix it: at 5x the budget the
    # 26B still returned 0 triplets on 12/12 turns, 6x slower, because the
    # reasoning block scales with input length. Of `options`, `keep_alive`,
    # `think` and `chat_template_kwargs`, only `reasoning_effort` survives
    # Ollama's OpenAI-compatible shim (see ROADMAP G32). Set False only for a
    # server that rejects the parameter outright.
    bg_disable_reasoning: bool = True

    # ── C7: in-process maintenance runtime (replaces Celery beat + Redis) ──
    # is_idle() gate for overdue-job dispatch: user quiet this long + no
    # generation in flight (today's 10s redis check was uselessly tight).
    user_active_threshold_seconds: int = 90
    # G4(a): a RUNNING background job stands down if the user did anything
    # within this window. Deliberately much tighter than the threshold that
    # gates *starting* one (90 s): starting early is merely premature, but
    # continuing while the user waits is latency they can feel — Ollama
    # serialises requests to the same model, so their first token queues
    # behind the background generation.
    job_yield_grace_seconds: int = 10
    # queued gpu-lane *event* jobs (post-flight backlog) start draining after
    # this much quiet. Both knobs re-measured at Z1 (ledger vs chat log).
    idle_burst_seconds: int = 120
    # D6/H5: fine-tune never runs unattended. Enough curated labels + a
    # session end → run only if auto_finetune, else one pending review-queue
    # proposal. Threshold aligned with fine_tune.MIN_ROWS_TO_PROMOTE.
    auto_finetune: bool = False
    finetune_min_curated: int = 20
    # Per-job cadences in seconds (G9-aligned: today's beat schedule, plus
    # compaction which beat never scheduled — G10). Keys = runtime job names.
    maintenance_intervals: dict[str, int] = {
        "cluster_assignment": 1800,
        "cluster_merge": 10800,
        "chunk_pending_documents": 7200,
        "decay_episodic": 5400,
        "decay_codex": 5400,
        "decay_procedural": 5400,
        "reflection": 7200,
        "batch_summarize": 7200,
        # C4: evolving whole-conversation summaries (also a session-end burst
        # member; the cadence pass is a no-op for quiet conversations).
        "conversation_summary": 7200,
        "compaction": 86400,
        # D1/D2: the maintenance agent (worklist + bounded LLM decisions);
        # also runs in the session-end burst.
        "maintenance_agent": 43200,
        # E3: reconcile-on-commit fallback — polls registered projects for
        # HEAD drift + hook marker files. The hook is the design; this is the
        # ≤10-min lag path when the hook is declined or the app was down.
        "project_poll": 600,
        # C12: scan ingest_inbox/ for dropped files. A cadence scan, not a
        # watchdog thread — the v1 drop zone WAS a watchdog and nothing ever
        # started it, so the folder silently did nothing for a year.
        "ingest_folder": 900,
    }

    # ── Track E: coding-ICE core ──
    # D11: a project-attached conversation almost always wants context —
    # pointers are cheap (bump, not force; B2 idiom).
    ltm_bump_coding: float = 0.7
    # E8 (D7): new-decision vs active-decision embedding similarity gates.
    # ≥ dedupe threshold + same type + same files ⇒ silent duplicate (skip);
    # ≥ conflict threshold + overlapping files ⇒ auto-supersede (Tier 1);
    # ≥ conflict threshold, no file overlap ⇒ review-queue proposal (Tier 2).
    decision_conflict_threshold: float = 0.85
    decision_duplicate_threshold: float = 0.95
    # E11: pull-based working-tree freshness on code-graph reads. The kill
    # switch restores commit-fresh behavior; the min interval bounds git-status
    # frequency under a burst of agent reads (at most one check per window).
    reconcile_on_read: bool = True
    reconcile_on_read_min_interval_seconds: float = 2.0

    # ── C9 (D4): procedural leg confidence floor — REPLACES the old 3-intent
    # whitelist as the precision mechanism (embedding rank + trigger match +
    # this floor). Z1 rule: >30 active patterns with <5% injection rate ⇒ 0.5.
    procedural_min_conf: float = 0.3

    # ── C16 (model-aware half): total context budget derived from the routed
    # model's context window instead of a hardcoded 23k. fraction reserves the
    # rest of the window for the model's output + safety margin; min/max are
    # guardrails (max keeps huge-window models from drowning in noisy context
    # until C16's need-based filling lands); fallback applies when the model
    # is unknown to the registry.
    # C6: silence longer than this within a conversation opens a new session
    # (a "sitting") — the boundary clustering/C7 maintenance key off.
    session_gap_minutes: int = 30

    # ── C12 D10: document text extraction. "builtin" = the pure-python parser
    # set (embedded text only; scanned PDFs are refused, never silently
    # ingested empty). "tika"/"docling" are reserved for the Track F OCR item —
    # they run as their own containers, exactly as Open WebUI does it, so
    # adding OCR never puts a system binary in setup.sh.
    document_extraction_engine: str = "builtin"

    context_input_fraction: float = 0.75
    context_budget_min: int = 4_000
    context_budget_max: int = 40_000
    context_budget_fallback: int = 23_000

    # ── C16 (need-based half): the budget stops being fiction ──────────────
    # Counting. The tokenizer is the EMBEDDER's (already loaded in process),
    # not the generation model's; the margin covers the cross-family drift,
    # which measured 0-3.5% on prose and ICE's own format and ~16% worst case
    # on Python source. It is calibrated by `token_usage_reconciliation`
    # below, which logs the prediction against the server's own
    # `prompt_eval_count` on every turn — so this number is measured, not
    # argued about. Set the reconciliation off only if a serving stack chokes
    # on `stream_options`.
    # MEASURED, not guessed: the first live reconciliation (tinyllama, a Llama
    # tokenizer) predicted 283 against the server's 331 — ratio 0.855, so 1.10
    # was NOT enough and the margin would have under-read the prompt. 1.20
    # covers that case; the reconciliation log accumulates the ratio per model
    # so Z1/Z2 can set it from a distribution instead of one point. Under-
    # reading is the dangerous direction — it says a prompt fits when it does
    # not — so this errs high.
    # Where the embedder runs. "auto" = the GPU when one exists, else CPU;
    # "cuda"/"cpu" force it. Measured on this repo: 321 ms per encode on CPU
    # against 21 ms on GPU (22.5 s vs 0.38 s for a batch of 100), for ~1.2 GB
    # of VRAM. Every chat turn encodes the prompt, so CPU was adding ~300 ms
    # to every request. Set "cpu" when the generation model needs the VRAM.
    embedding_device: str = "auto"

    # A9b: the BACKGROUND entity tier. The pre-flight tier deliberately keeps
    # the micro-NER — it shares the already-loaded encoder with the classifier
    # and the embedder, so it costs no extra VRAM on the synchronous path. This
    # model is only ever used post-flight, where the maintenance runtime's gpu
    # lane already serialises and idle-gates the work, so it can be loaded for
    # a drain and released again rather than sitting resident.
    # Measured 2026-08-03 (see PROVENANCE): against the micro-NER on a full
    # turn it is 4.7x faster (101 ms vs 345 ms on GPU), emits 15.6 entities
    # against 52.7, gives entity TYPES for free, and separates conversations
    # 28x better for clustering where the micro-NER tags `and`/`but`/`not` as
    # entities. Empty string disables it and everything falls back to the
    # micro-NER.
    background_ner_model: str = "numind/NuNER_Zero"
    background_ner_device: str = "auto"        # auto|cuda|cpu, like the embedder
    background_ner_threshold: float = 0.5
    # NuNER Zero's max_len counts ITS OWN word split, in which punctuation is a
    # word: 350 whitespace words measured 424 and were silently truncated. 250
    # holds with margin.
    background_ner_chunk_words: int = 250

    token_count_safety_margin: float = 1.20
    token_usage_reconciliation: bool = True

    # Room for the ANSWER. Nothing reserved this before C16, which is how a
    # 40,000-token budget could be derived for a runner that had allocated
    # 32,768 — already negative before a word was generated.
    context_generation_reserve: int = 2_048
    # Never squeeze memory below this, however large the user's own message is.
    context_budget_floor: int = 1_500
    # Budget against the window the SERVER allocated (via /api/ps) rather than
    # the registry's claim, which was measured 2x off on a live model.
    context_use_serving_window: bool = True

    # Coverage-based stopping (the "when is this enough" decision). The
    # question is a vector, every candidate is a vector; selection subtracts
    # what each pick covers and stops when nothing left adds anything. These
    # are STARTING values behind named knobs — Z1 is the tuning gate, and Z2's
    # mini-experiment is where they get moved.
    retrieval_coverage_enabled: bool = False
    coverage_alpha: float = 0.7            # coverage vs retrieval-confidence blend
    coverage_min_gain: float = 0.02        # a pick must cover at least this much
    coverage_min_keep: int = 2             # the knee may never cut below this
    coverage_max_keep: int = 40            # hard rail on the greedy loop
    coverage_knee_enabled: bool = True
    coverage_knee_min_prominence: float = 0.08   # below this: no knee, no cut
    # Set-level quality gate: if the BEST candidate is this far below what a
    # real match looks like, inject nothing rather than the best of a bad set.
    # Coverage measures direction, never quality — this is the only arm that
    # can say "there is nothing here".
    retrieval_set_floor_enabled: bool = False
    retrieval_set_floor: float = 0.25
    # A10's leg-diversity guarantee. Default ON = unchanged behaviour; it is a
    # DESIGNED answer to measured leg under-representation, so it is retired
    # only against a number, and with the user.
    retrieval_leg_guarantee_enabled: bool = True
    # Identity collapse: a chunk and its parent turn are the same memory.
    retrieval_collapse_enabled: bool = True
    retrieval_max_frags_per_turn: int = 2

    # The recent-turns window is CONTINUITY, not evidence — it is never
    # relevance-filtered. It is bounded by the sitting instead: this session's
    # turns, plus the last turn of the previous one so a 31-minute break does
    # not amputate the thread.
    recent_window_scope: str = "session"   # "session" | "count"
    recent_window_bridge_turns: int = 1
    recent_window_max_turns: int = 40      # marathon-sitting rail
    recent_window_max_turn_frac: float = 0.35   # one turn may not eat the window

    # ── G32/a1: constrained decoding for background extraction ──
    # Shape only, by default. Measured on 300 real turns (2026-08-04, see
    # PROVENANCE): the JSON *shape* constraint is a clear win — it removes
    # malformed-by-confusion output entirely. Constraining `relation` to the
    # 197-value vocabulary as well is a different bet, and on the same run it
    # kept 100% of triplets but ~78% of the ones it rescued were WRONG,
    # because a few relations act as attractors for anything the vocabulary
    # cannot express (`is` → `is_employed_by`, `is_sibling_of` →
    # `is_separated_from`).
    # ⚠ The enum is NOT settled and this flag is NOT decoration: whether it
    # ends up off, on over a repaired vocabulary, or on for a subset of call
    # sites is a **Z2 decision** (user, 2026-08-04). It is wired and usable so
    # Z2 can flip it and measure rather than rebuild it.
    # Raised from the historical 500 by G32/a1: at 500, finish_reason came
    # back "length" on 2 of 12 real turns and the JSON was unparseable every
    # time — the whole turn's extraction lost, silently, because the salvage
    # path cannot tell a truncated answer from a thin one. The schema does not
    # help: it guarantees a valid prefix, not a complete document.
    codex_extraction_max_tokens: int = 1200

    codex_constrain_shape: bool = True
    codex_constrain_relation_enum: bool = False

    # num_ctx. Telling the server the window we need costs KV-cache VRAM, so
    # it is opt-in and clamped. "fit" asks for exactly what the assembled
    # prompt needs plus the generation reserve.
    ollama_send_num_ctx: bool = False
    ollama_num_ctx_mode: str = "fit"
    ollama_num_ctx_max: int = 32_768

    # ── B2: principled memory-retrieval decision (log-odds combination) ──
    # These REPLACE the scattered hard LTM overrides. Every weight lives here
    # (not in code) because B2 sits on top of the current classifier, which B1
    # will retrain — so this is re-tuned, not rewritten, after B1.
    ltm_decision_threshold: float = 0.5        # τ: retrieve iff P(need_mem) > τ
    ltm_prior_bias: float = 0.4                # log-odds tilt: "prefer LTM, don't force"
    ltm_length_weight: float = 0.8             # β on the memory-pressure prior
    ltm_pressure_midpoint_tokens: int = 2000   # history-beyond-window at which P_len=0.5
    ltm_pressure_scale_tokens: int = 4000      # logistic steepness of the pressure prior
    ltm_bump_creative: float = 0.7             # Creative_&_Media topic → bump (not slam)
    # (ltm_bump_reference died with DI3 — D8, 2026-07-27. It was the anaphora
    #  rule's demoted bump, and nothing sets `reference_signal` any more. It also
    #  measured net-negative on the rows it fired for; see memory_decision's
    #  module docstring. ltm_bump_referential below is the surviving signal.)
    ltm_bump_referential: float = 0.5          # lighter referential-word presence → bump
    ltm_bump_low_confidence: float = 0.8       # topic/intent uncertainty safety net → bump

    # ── Track T: temporal retrieval & idea evolution ──
    # timescope_enabled is the kill-switch (False ⇒ every query behaves as
    # "current", byte-identical to pre-T behavior). Pads/fractions shape the
    # detector's windows (specs/T_temporal.md §2.1 table); probation is D-U1's
    # cold-resurrection score, just above decay.ARCHIVE_THRESHOLD (0.1).
    # timeline_*/evolution_* are consumed by T4 (timeline builder — next
    # session); declared here so the settings block ships once.
    timescope_enabled: bool = True
    ltm_bump_timescope: float = 3.0
    # B1 D7: the Temporal_Recall sigmoid counts as the same evidence as a fired
    # detector for the bump above (OR, never AND). Only the detector sets windows.
    #
    # Raised 0.6 → 0.85 (2026-07-27, B1 run-2 probe audit). The v2 head does not
    # fire as an independent time signal — it fires as a SHADOW of Needs_Memory,
    # with which it co-occurs in 79% of its training positives. Measured on
    # hand-authored probes: mean p_temporal 0.87 across a cell of memory-needing
    # prompts carrying no temporal content whatsoever, 0.82 on "is the migration
    # plan still consistent with the deadline I'm working to". Because this is
    # OR'd with the detector, a low threshold makes the deterministic parser
    # redundant and drags the retrieval decision toward always-retrieve — the
    # exact failure B2 exists to prevent. 0.85 keeps the genuine no-parseable-date
    # catches (which score 0.93+) while cutting the shadow.
    # Z1-prep owns the final value; sweep it against the INDEPENDENT probe sets,
    # never the held-out split (that split shares the labelers' bias).
    temporal_label_threshold: float = 0.85
    timescope_pad_min_days: int = 3
    timescope_pad_month_days: int = 14
    timescope_pad_span_days: int = 21
    timescope_pad_year_days: int = 30
    timescope_rel_short_frac: float = 0.4    # ≤ week-granularity relatives
    timescope_rel_short_cap_days: int = 45
    timescope_rel_long_frac: float = 0.15    # month/year-granularity relatives
    timescope_rel_long_cap_days: int = 120
    timescope_probation_score: float = 0.12
    timescope_cold_limit: int = 5
    timeline_max_tokens: int = 300
    timeline_max_transitions: int = 8
    timeline_max_fragments: int = 2
    timeline_max_fragments_evolution: int = 4
    evolution_era_buckets: int = 4
    evolution_per_era: int = 3

    # (The seven DI3_* knobs lived here until D8, 2026-07-27, and went with the
    #  module. Note for anyone reading old configs: DI3_LTM_REFERENCE_DENSITY_
    #  THRESHOLD never did anything — it was the "past ten turns" tier of the
    #  reference rule, and no caller ever passed conversation_length.)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()