"""Configuration for the ICE FastAPI proxy."""
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.paths import REPO_ROOT


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
    # G28: conservative current-source skip and source-backed negative rescue.
    # The model's enum is never substituted for a calibrated classifier prior.
    memory_source_gate_enabled: bool = True  # paired question/source proof; unknown retains B2
    memory_source_rescue_enabled: bool = True  # admit only a proven, budget-fitting original
    memory_source_gate_model: str = "gemma4:e4b"
    memory_source_gate_input_tokens: int = Field(default=8192, gt=0)
    memory_source_gate_context_tokens: int = Field(default=32768, gt=0)
    memory_source_gate_output_tokens: int = Field(default=256, gt=0)
    memory_source_proof_output_tokens: int = Field(default=2048, gt=0)

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
    # G27/A12: the general background model (summaries and their claim pass —
    # which also writes the codex facts —, folds, reflection, procedures,
    # maintenance agent, naming). gemma4:e4b was chosen by the
    # 2026-08-27 bake-off (11 candidates × 9 jobs) and the background prompts
    # are tuned for it — swapping it is a re-measure, not a config edit.
    # Explicitly None ⇒ shared mode falls through to the first confirmed
    # registry entry (NOT the chat model, and possibly a large one), warned on
    # every call as `bg_model_unpinned`; dedicated mode uses the vLLM default.
    background_model_name: Optional[str] = "gemma4:e4b"
    # ⚑ ICE MANAGES ITS OWN MODEL RESIDENCY (maintainer, standing requirement:
    # "dont let default ollama do it"). After the maintenance runtime drains its
    # queue, the background model is explicitly unloaded rather than left to the
    # host's `OLLAMA_KEEP_ALIVE`. Measured need, 2026-08-27: three background
    # models sat resident at ~14 GB of a 24 GB card, `UNTIL: Forever`, for jobs
    # that had finished hours before — because `/v1/chat/completions` drops
    # `keep_alive` and ICE therefore had no way to say otherwise.
    # ⚠ This is the SMALL half of G32(a). Per-request `keep_alive` still needs
    # the native chat endpoint; this only releases after a drain.
    # Set False on a machine where the background model is shared with something
    # else that would pay the reload cost.
    bg_release_after_drain: bool = True
    bg_release_timeout_seconds: float = 10.0
    # G12: bg-client calls scale their timeout with the requested output size:
    # timeout = bg_timeout_base_seconds × clamp(max_tokens / 500, 1, 6).
    bg_timeout_base_seconds: float = 30.0
    # G12, input half (2026-10-15): native background calls add this much per
    # 1,000 input tokens on top of the output-scaled timeout. Measured on
    # gemma4:e4b: a 69K-token prompt reads in 14.6 s (+~4 s reload when the
    # context size changes), so 1 s per 1K is ~4x headroom.
    bg_timeout_seconds_per_1k_input: float = 1.0
    # G32/G4 (2026-10-15): which door background calls use in shared mode.
    # "auto" ⇒ Ollama's own /api/chat, which honours num_ctx and keep_alive;
    # "openai" ⇒ the OpenAI-compatible /v1 door (for a server that only speaks
    # that). Dedicated mode always uses the OpenAI-compatible door.
    # Measured on Ollama 0.32.13: /v1 loads at 32,768 tokens and a ~50K-token
    # input came back HTTP 200 / "stop" with its head silently dropped — and
    # the model invented the dropped fact.
    background_transport: Literal["auto", "ollama", "openai"] = "auto"
    # Native background calls size num_ctx to the input: the smallest of
    # floor, 2×floor, 4×floor… that fits, capped by the model's own maximum.
    # The floor matches what Ollama loads by default on a 24 GB card, so the
    # common case never changes the runner (each change reloads, ~4 s).
    bg_num_ctx_floor: int = Field(default=32_768, ge=2_048)
    # ICE keeps its background model resident while it works and releases it
    # itself after the queue drains (`bg_release_after_drain`), so the host's
    # OLLAMA_KEEP_ALIVE never decides. Ollama duration or seconds; -1 = until
    # released.
    bg_keep_alive: int | str = -1
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
    # proposal. G9 collapsed fine_tune.MIN_ROWS_TO_PROMOTE into this — the
    # two were the same gate declared twice, with a comment asserting they
    # agreed rather than anything enforcing it.
    auto_finetune: bool = False
    finetune_min_curated: int = 20
    # Per-job cadences in seconds (G9-aligned: the old beat schedule).
    # Keys = runtime job names.
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
    # ⚑ Activation on EVIDENCE, not on re-emission (G49, 2026-08-16).
    # `is_active` required reinforcement_count >= 3, and reinforcement needs a
    # later session to produce a semantically equivalent description. Historically,
    # cosine approved repeats; two
    # write-ups of the SAME habit measure 0.708 against a 0.85 bar — so nothing
    # activated. Measured on the seeded arm: 61 patterns citing 514 turns
    # between them, ONE active, with patterns citing 20 turns each sitting at
    # reinforcement 1. The evidence was already there and was being ignored.
    # G42 enforces >=2 distinct in-range cited turns at creation, so cited turns are the signal
    # this system can actually produce; reinforcement stays as a second path.
    # ⚠ 10 is a defensible default, NOT a calibration. The C9 note above states
    # the rule for settling it: >30 active patterns with <5% injection rate
    # => raise. Measure the injection rate after the re-seed and tune here.
    procedural_min_cited_turns: int = 10

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
    # v3: local cross-encoder relevance, after fusion and before selection.
    retrieval_rerank_enabled: bool = True
    retrieval_rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"
    retrieval_rerank_revision: str = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    retrieval_rerank_device: str = "auto"
    retrieval_rerank_candidates: int = Field(default=64, ge=1, le=256)
    retrieval_rerank_batch_size: int = Field(default=4, ge=1, le=16)
    retrieval_rerank_max_tokens: int = Field(default=4096, ge=128, le=32768)
    retrieval_rerank_min_score: Optional[float] = Field(default=None, allow_inf_nan=False)
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
    # A10's fusion-fallback family-diversity strategy (guarantee + fair filling).
    # OFF fills by fused score; successful reranking always owns its order.
    # Default ON = unchanged behaviour; it is a
    # DESIGNED answer to measured leg under-representation, so it is retired
    # only against a number, and with the user.
    retrieval_leg_guarantee_enabled: bool = True
    # Identity collapse: a chunk and its parent turn are the same memory.
    retrieval_collapse_enabled: bool = True
    retrieval_max_frags_per_turn: int = 2

    # ── G9: retrieval scoring knobs (were module constants + __init__ attrs) ──
    # Everything below was a literal in orchestrator.py until 2026-08-08, so
    # tuning any of it meant editing code. Z1's staged sweep needs them
    # config-driven; the defaults here ARE the pre-G9 values, verified against
    # the base commit by tests/test_settings_freeze.py.
    #
    # ⚠ Read these at the point of USE, never bind them to a module global or
    # an instance attribute at construction. Z1's sweeper mutates the settings
    # singleton at runtime, and a second copy of a value is how the ablation
    # harness's `recency_boost` flag came to do nothing for a year (see the
    # G19 note in ROADMAP).
    #
    # C8: recency folded INTO the vector score, because the candidate set is
    # cut by score before post-fusion bonuses can rescue anything recent.
    retrieval_episodic_recency_boost: float = 0.25
    retrieval_episodic_recency_tau_days: float = 30.0

    # C15: the wide net's budget derives from the model-aware retrieval budget.
    retrieval_wide_net_budget_fraction: float = 0.3
    retrieval_wide_net_budget_floor: int = 1500

    # Post-fusion additive bonuses. Each is a FRACTION of the base score:
    # 0.5 means +50%, and the sum is clamped to [min, max] before the score is
    # multiplied by (1 + bonus).
    retrieval_bonus_bookmarked: float = 0.5
    retrieval_bonus_recent_top_10pct: float = 1.0
    retrieval_bonus_recent_top_30pct: float = 0.5
    retrieval_bonus_long_narrative: float = 1.5
    retrieval_bonus_substantial: float = 0.5
    retrieval_penalty_short: float = -0.7
    retrieval_bonus_keyword_match: float = 1.0
    retrieval_max_total_bonus_multiplier: float = 4.0
    retrieval_min_total_bonus: float = -0.9
    # Soft meta-discussion downweight: multiply by this when the source turn
    # leans meta and the query wants narrative fact.
    retrieval_meta_downweight_factor: float = 0.55

    # Word-count brackets the length bonuses key on.
    retrieval_long_narrative_words: int = 800
    retrieval_substantial_words: int = 400
    retrieval_short_words: int = 80

    # Post-fusion recency bands, as a fraction of the conversation. Below the
    # minimum turn count the whole bonus is skipped — in a short conversation
    # "the most recent 10%" is one turn and means nothing.
    retrieval_recent_top_pct: float = 0.10
    retrieval_recent_mid_pct: float = 0.30
    retrieval_recency_min_turns: int = 20

    # Reciprocal-rank fusion. Larger k flattens the rank curve.
    retrieval_rrf_k: int = 60

    # ── G9: the intent-dependent leg weights (Z1 stage 2 sweeps these first) ──
    # A table, kept shaped like a table — a row is one intent's opinion about
    # all four legs, and flattening it into thirty scalar fields would destroy
    # the thing that makes it reviewable. Resolved and VALIDATED on every read
    # by src/retrieval/leg_weights.py; see that module for why validation is at
    # read time and why the two halves fail differently.
    retrieval_leg_base_weights: dict[str, float] = {
        "bm25": 0.8,
        "vector": 1.0,
        "codex": 0.5,
        "procedural": 0.2,
        # T4: pinned, not blended — its firing condition is the gate.
        # ⚠ INERT since CODEX_V4 (2026-10-16): no leg emits timeline fragments
        # (a current fact carries its replaced predecessor inline). Kept as an
        # inert seam for a future history leg (decided 2026-10-17).
        "timeline": 0.6,
    }
    # intent label → per-leg override. An intent absent from this table
    # contributes the base weights instead.
    #
    # `Code_Change` is B1 D9's v2 coding intent. E12 measured that the row is
    # genuinely reached: the head fires it on 5.75% of held-out rows carrying a
    # mean 1.79 intents, so it lands ~56% of its intended weight and still
    # shifts the biggest leg by 0.291 against base weights of 0.2–1.2.
    #
    # D9's second row — Codebase_Query → {codex 1.3, bm25 0.9, vector 0.8,
    # procedural 0.6} — is deliberately absent: the label was dropped in B1's
    # run 2 (test F1 0.10), so it cannot appear in intent_tags and a row for it
    # would be unreachable. Those are the tuned starting values if E7's MCP
    # traffic ever earns the label back.
    retrieval_leg_profiles: dict[str, dict[str, float]] = {
        "Factual_Retrieval":       {"vector": 1.2, "bm25": 0.8, "codex": 0.1, "procedural": 0.1},
        "Utility_Formatting":      {"vector": 1.2, "bm25": 0.8, "codex": 0.1, "procedural": 0.1},
        "Troubleshooting":         {"vector": 1.0, "bm25": 0.8, "codex": 0.3, "procedural": 1.2},
        "Strategic_Planning":      {"vector": 1.0, "bm25": 0.8, "codex": 0.3, "procedural": 1.2},
        "Generation":              {"vector": 0.6, "bm25": 0.6, "codex": 1.2, "procedural": 0.1},
        "Ideation":                {"vector": 0.6, "bm25": 0.6, "codex": 1.2, "procedural": 0.1},
        "Open_Exploration":        {"vector": 0.6, "bm25": 0.6, "codex": 1.2, "procedural": 0.1},
        "Emotional_Processing":    {"vector": 1.1, "bm25": 0.6, "codex": 0.9, "procedural": 0.0},
        "Analysis_&_Summarization": {"vector": 1.1, "bm25": 0.6, "codex": 0.9, "procedural": 0.0},
        "Decision_Making":         {"vector": 1.1, "bm25": 0.6, "codex": 0.9, "procedural": 0.0},
        "Casual_Banter":           {"vector": 0.5, "bm25": 0.2, "codex": 0.0, "procedural": 0.0},
        "Null_Noise":              {"vector": 0.5, "bm25": 0.2, "codex": 0.0, "procedural": 0.0},
        "Code_Change":             {"procedural": 1.2, "codex": 1.0, "vector": 0.6, "bm25": 0.6},
    }
    # topic label → cumulative deltas applied after the intent blend.
    retrieval_leg_topic_overrides: dict[str, dict[str, float]] = {
        "Creative_&_Media": {"codex": 0.3},
        "Software_&_Tech": {"procedural": 0.4},
    }

    # Per-leg candidate limits. These bound what each leg hands to fusion, so
    # they set the ceiling everything downstream can pick from.
    retrieval_bm25_candidate_limit: int = 100
    retrieval_vector_candidate_limit: int = 100
    # T4 evolution queries need a wider net to span eras before stratifying.
    retrieval_vector_candidate_limit_evolution: int = 300
    retrieval_chunk_candidate_limit: int = 30
    retrieval_wide_net_candidate_limit: int = 100
    retrieval_procedural_limit: int = 5
    retrieval_batch_summary_limit: int = 3
    retrieval_conversation_summary_limit: int = 2

    # Session diversification: how many fragments one FOREIGN conversation may
    # contribute (the active conversation is uncapped).
    retrieval_max_per_conversation: int = 3

    # Cluster restriction: how many clusters scope a query, and how many
    # candidates the SQL over-fetches before tag-overlap re-ranking.
    retrieval_cluster_top_k: int = 10
    # Audit A12 (2026-10-15): automatic scoping to the top-k clusters is OFF.
    # Re-clustering the dev history so clusters became meaningful (raw floor
    # 0.55: 38 clusters, largest 44%) scoped out 19/108 gold turns of the 67
    # reviewed questions (2/108 even with today's single mega-cluster). An
    # explicit user-chosen cluster scope still applies.
    retrieval_auto_cluster_scope: bool = False
    retrieval_cluster_candidate_multiplier: int = 3

    # ── G9 (Tier 2): operational knobs ─────────────────────────────────────
    # These do not decide what ICE remembers — they bound cost, batch sizes and
    # timeouts. Z1 does not sweep them; they are here so the packaged app can
    # expose them and so nothing is left needing a code edit.
    gpu_util_threshold: int = 70
    gpu_check_cache_seconds: float = 10.0
    runtime_probe_timeout: float = 2.0
    runtime_probe_cache_ttl_seconds: int = 300
    # Missed cadences compress into one pass; this caps the catch-up so a long
    # downtime cannot apply an unbounded exponent in a single statement.
    runtime_cycles_cap: int = 96
    runtime_lease_ttl_seconds: float = 180.0
    chat_confirm_ttl_seconds: int = 600

    # Document ingestion cost estimates and batch sizes.
    document_section_seconds: int = 1
    document_seconds_per_section: float = 6.0
    document_blob_kind_sample_chars: int = 1500
    document_watch_files_per_run: int = 1
    document_catchup_docs_per_run: int = 5
    # Tabular files are described, not dumped: a head plus a deterministic
    # sample, every column described but at most this many rendered.
    document_table_sample_rows: int = 50
    document_table_max_render_cols: int = 30
    document_table_top_values: int = 5

    # Import decay policy windows (F10).
    import_immune_window_days: int = 14
    import_recent_days: int = 30
    import_seconds_per_turn: float = 6.0
    import_stale_run_seconds: float = 600.0
    import_slice_budget_seconds: float = 600.0
    # F14 raw-transcript slicing. ~2,000 prose words per slice.
    raw_slice_tokens: int = 2667
    raw_slice_overlap_words: int = 200
    raw_slice_seam_turns: int = 3

    # Coding side (E-track).
    reconcile_max_commits_scanned: int = 20
    # E11 D5: warn past this, never cap — a Z1 signal, not a limit.
    reconcile_large_dirty_set: int = 50
    code_graph_resolved_confidence: float = 1.0
    code_graph_heuristic_confidence: float = 0.6
    code_graph_max_files: int = 20_000

    # Fine-tune promotion: the held-out split. The row-count gate is
    # finetune_min_curated above — one gate, one setting.
    finetune_val_fraction: float = 0.2

    slot_token_cap: int = 300

    # ── G9: the write path (what gets stored, and in what form) ────────────
    # "Memory is earned" is decided by these numbers. A turn at or below the
    # raw-keep length is stored raw with no summary; above the entropy
    # threshold it earns lossless_flag and gates codex extraction; a summary
    # is trusted only if it retains this fraction of the turn's must-terms.
    turn_raw_keep_max_words: int = 350
    turn_long_turn_chunk_words: int = 600
    turn_density_lossless_threshold: float = 0.35
    turn_summary_coverage_threshold: float = 0.7
    turn_max_must_terms: int = 25
    # ⚑ The summariser's generation ceiling, and it BOUND — hard. It was a
    # literal 300 in post_flight, and on the arm-B seed 84 of 212 summaries
    # (40%) came back truncated: `bg_model_output_truncated` fired at
    # max_tokens=300 that many times, and 47 summaries have no `abstract_text`
    # because the Abstract line the prompt asks for comes AFTER the Key terms
    # block and a 300-token cut removes it. A ceiling that severs 40% of its
    # outputs is not a budget, it is a defect. 900 is ~3x the observed
    # requirement and still an order of magnitude under the background model's
    # window; the truncation rate is the number to watch after changing it.
    turn_summary_max_tokens: int = 900
    # SUMMARY_CLAIMS_V3 (2026-10-15): v2 = legacy cited assertions accepted
    # all-or-nothing; v3 = {sources, quote, speaker, text} claims with
    # mechanical quote/speaker checks, enforced JSON schema and per-claim
    # acceptance. v3 since 2026-10-15 after the spec's §3 qualification
    # (31 pieces: usable pieces 9 → 28, verified claims 163 → 290,
    # contradicted 1 → 1, broken JSON 1 → 0). v4 = v3 + modality and the
    # source's subject/relation/object (CODEX_V4 D4): every supported claim is
    # also a codex fact — the only path that writes the codex (CODEX_V4, measured
    # 2026-10-16: §5.1 qualification, §5.2 reconciliation, the side retrieval
    # comparison). Under v2/v3 no facts are written (facts_stage_skipped).
    summary_assertion_method: Literal["v2", "v3", "v4"] = "v4"
    # Windows are summarised in pieces of at most this many prompt tokens, one
    # call each (turn summary, fold group). 2026-10-15: C's largest
    # turn is ~70K tokens; one call at that size was cut silently by the server.
    summary_piece_tokens: int = Field(default=4096, ge=512)
    # Output cap = clamp(ratio × piece input tokens, job floor, ceiling). The
    # job floor is its legacy cap (turn/conversation summary 900). Set from the SUMMARY_CLAIMS_V3 §3 run (31 pieces, 2026-10-15): v3
    # output/input median 0.53, max 1.05 on ≥2K-token pieces (list-dense turns
    # emit ~one claim + quote per line); 1.3 = max × 1.25. Raised to 2.0 by
    # the CODEX_V4 §5.1 run (41 pieces, 2026-10-15): at 1.3, 5/26 v3 and
    # 15/26 v4 2K-token C pieces were cut mid-JSON; the v4 fact shape needs up
    # to 1.58 on ≥2K-token pieces (cap 8,192 run) → 2.0 = 1.58 × 1.25.
    summary_output_ratio: float = Field(default=2.0, gt=0)
    summary_output_ceiling: int = Field(default=8192, ge=256)

    # Chunking (A1), used by document ingest.
    chunk_tokens: int = 550
    chunk_overlap_words: int = 50

    # C4: generation word target and soft grouping target for whole evidence
    # representations. Hard capacity is checked on the complete model request.
    # v3 cited JSON notes need schema/citation overhead beyond the content
    # word target. Provider-length fallback retains complete source, not prefix.
    conversation_summary_max_tokens: int = 900
    conversation_summary_max_words: int = 250
    conversation_summary_chunk_words: int = 3500
    conversation_note_prompt_tokens: int = 650

    # G11: a turn is batch-summarised once its decay falls below the retrieval
    # floor OR it passes this age, whichever comes first. Handed to G9 by G11.
    batch_summary_age_days: int = 30

    # What the classifier sees of the conversation before the current turn.
    # ⚠ Changing these changes what the classifier was TRAINED against, so
    # they are a retrain-coupled knob, not a free one (see B1).
    classifier_context_turns: int = 3
    classifier_context_max_words: int = 500
    classifier_turn_word_cap: int = 150

    # ── G9: memory dynamics (decay, promotion, archival) ───────────────────
    # Z1's D5 rule: these are NOT swept against a probe loop, because they are
    # temporal behaviours a single-turn probe cannot see. They are solved from
    # target half-lives and pinned by tests/test_dynamics_invariants.py.
    #
    # So the DAILY target is the setting and the per-cycle rate is derived:
    # rate = daily ** (1 / cycles_per_day). Storing the per-cycle rate instead
    # would mean anyone retuning it has to do the algebra by hand, on a number
    # whose meaning (0.9968) is unreadable.
    #
    # There is no `cycles_per_day` setting on purpose: it is DERIVED from
    # maintenance_intervals by decay.cycles_per_day(). It has to equal
    # 86400 / cadence or the daily target below is not the decay that actually
    # happens, so a knob that could disagree with the cadence has no correct
    # value — it only has a wrong one that looks configured.
    decay_daily_unaccessed: float = 0.95
    decay_daily_accessed: float = 0.98
    decay_daily_creative: float = 0.99
    # G42: procedural extraction reads a SESSION of the user's own prompts, not
    # one turn. A single turn cannot evidence recurrence, so asking a model to
    # name a "recurring" habit from one turn guarantees invention — measured
    # 2026-08-12, all 247 stored patterns were evidenced by exactly one turn and
    # only 2 ever reached the reinforcement threshold. Sessions shorter than
    # `min_session_turns` are skipped rather than guessed at; `session_step`
    # re-extracts as a session grows instead of once per turn (a 20-turn session
    # costs 5 calls, not 20).
    # CODEX_V4 D5 — fact entity links. An object longer than this is a value
    # (a sentence, a tensor, a URL with a query), kept as text on the fact, not
    # an entity.
    codex_fact_entity_max_words: int = 6
    codex_fact_entity_max_chars: int = 80
    # CODEX_V4 D8 — candidates per channel (keyword, meaning, entity) of the
    # facts leg; same scale as the claim leg it replaces.
    codex_fact_candidate_limit: int = 64
    # CODEX_V4 D7 — the post-rerank strength bonus bound B (support 2/3,
    # decayed usage 1/3). 0 until §5.3 sets B from the measured spread of
    # rerank logits on the development questions; not a guessed value.
    codex_fact_strength_bonus: float = 0.0
    # CODEX_V4 D6 — same / replaces / both_true between a new fact and earlier
    # facts of the SAME speaker. Candidates: shares a named entity, or the 8
    # nearest by fact embedding at cosine >= the floor (dev history: 262 of
    # 11,878 nearest pairs clear 0.80, ~1 per turn). The decision model is the
    # background model unless set: a 27B judged 2/94 false replaces vs gemma's 2
    # after the label rule, but cannot sit beside the background stack on a
    # 24 GB card (§5.2, 2026-10-16).
    codex_fact_candidate_cosine: float = 0.80
    codex_fact_candidate_k: int = 8
    codex_fact_reconcile_model: str = ""
    # ── Z1 probe generation (tooling, not a runtime path) ───────────────────
    # An external model writes the typed probe set (G48). Declared here because
    # Settings forbids extra keys: leaving these as bare .env lines made EVERY
    # import of this module raise ValidationError, which takes the whole
    # application down — the proxy, the workers and every test at once. Found
    # 2026-08-13, seconds after they were added.
    # ⚠ probe_api_key is a CREDENTIAL. Never log it, never echo it, never write
    # it into a tracked file. `.env` is gitignored (.gitignore:5) and untracked.
    probe_api_key: str = ""
    probe_api_base_url: str = ""
    probe_model: str = "deepseek-v4-flash"

    # CoE AI Gateway (TCET campus, added 2026-08-17) — OpenAI-compatible,
    # Qwen3.6-35B-A3B on a DGX Spark, vLLM-served. See docs/MODELS.md §3.
    # Reached for where a model's JUDGEMENT is wanted but spinning up a local
    # model is the only alternative: no VRAM, no model swap, no cold start.
    # ⚠ NOT the paired judge — that stays pinned to probe_model so verdicts
    # remain comparable across runs (maintainer's call 2026-08-17).
    # ⚠ coe_api_key is a CREDENTIAL, same rules as probe_api_key above.
    # ⚠ AND THESE THREE LINES ARE WHY THE COMMENT ABOVE EXISTS: adding
    # COE_* to `.env` without declaring them here took the whole application
    # down again on 2026-08-17 (`extra_forbidden`), exactly as recorded for
    # 2026-08-13. An env var is not optional config — declare it or omit it.
    coe_api_key: str = ""
    coe_api_base_url: str = ""
    coe_model: str = "qwen3.6"

    # Output ceiling for a batch summary. Was hardcoded at 500, which truncated
    # 2 of the 3 summaries on the arm-1 store mid-sentence (one ended "...**The
    # plan is", another cut a list at item 15) while `bg_model_output_truncated`
    # logged it and nothing acted on it. A batch summary is what retrieval
    # serves INSTEAD of the raw turns once they age out, so a truncated one
    # silently loses the tail of a 50-turn span — the compression is lossy in
    # the one direction it must not be. 2-3 paragraphs preserving names,
    # numbers and decisions over 50 turns needs the room.
    batch_summary_max_tokens: int = 1200

    procedural_min_session_turns: int = 3
    procedural_session_step: int = 5
    procedural_max_prompt_chars: int = 12000
    # v3 retained-session output hit the old 160-token cap. Complete outputs
    # and source citations are required; this cap is not a truth threshold.
    procedural_extraction_max_tokens: int = 512
    # Historical replay pin; inert in the v3 writer since mutual entailment
    # replaced cosine approval. Opposite workflow orders scored 0.986.
    procedural_similarity_threshold: float = 0.85
    # Similarity nominates candidates; complete bidirectional support merges.
    procedural_match_candidates: int = Field(default=5, ge=1, le=32)

    decay_strengthen_amount: float = 0.15
    # Master switch for the strengthening WRITE itself, not its magnitude.
    # Zeroing decay_strengthen_amount stops decay_score moving but NOT the
    # access_count increment beside it, and that increment is enough to make
    # retrieval irreproducible: measured 2026-08-12, two identical runs over
    # the same store agreed on only 26 of 40 result sets with it on, and 40 of
    # 40 with it off. access_count has no reader in the retrieval path, so the
    # drift is a pure side effect — the rows are rewritten and equal-scoring
    # candidates come back in a different order. Left ON in production; the
    # Z1 scorer turns it off so a tuning delta is not competing with it.
    retrieval_strengthen_writes: bool = True
    decay_archive_threshold: float = 0.1
    decay_cold_threshold: float = 0.05
    # Creative turns never decay below this — they are re-read in bursts
    # months apart, so the usual "unused means unwanted" inference is wrong.
    decay_creative_floor: float = 0.3

    # Daily fade of codex facts' prompt usage (CODEX_V4 D7) — ranks, never
    # admits, never touches source support.
    codex_decay_daily: float = 0.99

    # Procedural patterns: unreinforced for this long, with fewer than this
    # many reinforcements, and the pattern is retired.
    procedural_stale_days: int = 180
    procedural_min_reinforcement: int = 3

    # ── G9: clustering (C5 v5) ─────────────────────────────────────────────
    # The assignment bar, the merge bars, and the bonuses that adjust raw
    # centroid similarity. Merging is permanent, which is why it carries both
    # a higher bar and a stricter bonus cap than assignment.
    cluster_similarity_threshold: float = 0.6
    # Audit A12 (2026-10-15): bonuses (tags ≤0.10 + entities ≤0.30 + session
    # 0.10) could lift a raw cosine of ~0.1 over the 0.6 threshold, and union-
    # find chained such pairs: the dev history became ONE cluster and the
    # largest store's biggest cluster held 1,075/1,471 turns, 51% of them
    # below 0.6 raw. Bonuses now apply only at or above this raw similarity.
    # Measured on a dev-store copy: 0.45 → largest cluster 237/251, 0.50 → 222,
    # 0.55 → 110 (38 clusters); logs/audit-a12-clustering-2026-10-15-r1/.
    cluster_min_raw_similarity: float = 0.55
    cluster_merge_similarity_threshold: float = 0.90
    cluster_merge_min_raw_sim: float = 0.82
    cluster_max_turns_per_run: int = 25
    cluster_entity_overlap_bonus_per_shared: float = 0.08
    cluster_entity_overlap_bonus_cap: float = 0.30
    cluster_merge_entity_bonus_cap: float = 0.10
    cluster_tag_overlap_bonus: float = 0.05
    # v5: capped, because within one conversation tags are near-uniform and
    # uncapped this was a constant offset rather than a signal.
    cluster_tag_overlap_bonus_cap: float = 0.10
    cluster_session_affinity_bonus: float = 0.10
    cluster_singleton_age_hours: float = 24.0
    cluster_name_regen_interval: int = 5
    cluster_entity_extraction_max_chars: int | None = None

    # ── G9: maintenance agent per-run caps (D8) ────────────────────────────
    # A flooded review queue is worse than no agent, which is what the tier-2
    # cap is defending.
    agent_max_scanned: int = 50
    agent_max_llm_decisions: int = 25
    agent_max_tier2_proposals: int = 5
    agent_max_applications: int = 10
    # Z1's manual review of the first proposals moves this to 0.94 (if >20% of
    # "same" verdicts are wrong) or 0.85 (if it never fires).
    agent_dup_cosine_threshold: float = 0.90
    agent_dup_pairs_per_run: int = 10
    agent_stale_slot_days: int = 14
    agent_stale_task_days: int = 14

    # ── G9: the dynamic-budget ladders ─────────────────────────────────────
    # These split the context budget between the recent-turns window and
    # retrieval, and they are the knob the flaw ablation scored WORST:
    # dynamic_budget measured −0.11 while adding tokens (14.9k→24.5k) and
    # doubling fragment count. The roadmap's verdict is "retune, mistuned at
    # high turn counts" — which is only possible once the ladders are data.
    #
    # Fallback ceiling when no model-derived budget is passed. C16 derives the
    # real one from the routed model's window; this is the legacy/direct-caller
    # path, and it duplicates context_budget_fallback on purpose for now — the
    # two are consumed by different call paths.
    context_total_budget_fallback: int = 23_000
    context_overhead_reserve: int = 1_800

    # Retrieval's cap grows with conversation length: [turn_count_below, base,
    # per_turn]. Past the last bracket there is no cap. Read in order.
    context_growth_cap_ladder: list[list[int]] = [
        [30, 2_000, 150],
        [100, 5_000, 100],
        [500, 10_000, 30],
    ]
    # Share of the budget given to the recent window, by conversation length:
    # [turn_count_below, fraction]. Past the last bracket, the default applies.
    # (The 50/200 and 500/default rows carry equal values — that is the shipped
    # shape, preserved rather than collapsed, because collapsing it would be a
    # tuning decision wearing a refactor's clothes.)
    context_recent_fraction_ladder: list[list[float]] = [
        [10, 0.3],
        [50, 0.2],
        [200, 0.2],
        [500, 0.15],
    ]
    context_recent_fraction_default: float = 0.15
    # Long turns shift budget toward retrieval: [avg_tokens_per_turn_above,
    # delta]. First match wins, so keep it descending.
    context_recent_density_ladder: list[list[float]] = [
        [3000, -0.15],
        [1500, -0.10],
        [800, -0.05],
    ]
    # Label-driven adjustments. Each GROUP applies its delta at most once, even
    # when several of its labels are active — which is why this is a list of
    # groups and not a label→delta map.
    context_recent_fraction_groups: list[dict] = [
        {"head": "intent", "delta": -0.10,
         "labels": ["Factual_Retrieval", "Troubleshooting", "Analysis_&_Summarization"]},
        {"head": "intent", "delta": 0.10,
         "labels": ["Emotional_Processing", "Casual_Banter"]},
        {"head": "topic", "delta": 0.05, "labels": ["Creative_&_Media"]},
        {"head": "topic", "delta": -0.05, "labels": ["Software_&_Tech"]},
        {"head": "topic", "delta": 0.05,
         "labels": ["Social_&_Relationships", "Lifestyle_&_Health"]},
    ]
    context_recent_fraction_min: float = 0.05
    context_recent_fraction_max: float = 0.85

    # ── G9: codex retrieval ────────────────────────────────────────────────
    codex_expansion_max_terms: int = 8
    codex_enum_entity_limit: int = 8
    codex_entity_match_threshold: float = 0.85

    # The recent-turns window is CONTINUITY, not evidence — it is never
    # relevance-filtered. It is bounded by the sitting instead: this session's
    # turns, plus the last turn of the previous one so a 31-minute break does
    # not amputate the thread.
    recent_window_scope: str = "session"   # "session" | "count"
    recent_window_bridge_turns: int = 1
    recent_window_max_turns: int = 40      # marathon-sitting rail
    recent_window_max_turn_frac: float = 0.35   # one turn may not eat the window

    # Source support (NLI) for summary claims and codex facts.
    source_support_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    source_support_revision: str = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
    source_support_device: str = "auto"
    source_support_threshold: float = Field(default=0.95, gt=0.5, le=1.0)
    source_support_max_tokens: int = Field(default=512, ge=16, le=512)
    # G32(b): the native local foreground path explicitly owns its window.
    # "fit" sizes to the guarded prompt + generation reserve; "max" allocates
    # the known serving ceiling. Opting out restores the host's Ollama default.
    ollama_send_num_ctx: bool = True
    ollama_num_ctx_mode: Literal["fit", "max"] = "fit"
    ollama_num_ctx_max: int = 32_768
    foreground_read_timeout_seconds: float = 120.0

    # ── B2: principled memory-retrieval decision (log-odds combination) ──
    # These REPLACE the scattered hard LTM overrides. Every weight lives here
    # (not in code) because B2 sits on top of the current classifier, which B1
    # will retrain — so this is re-tuned, not rewritten, after B1.
    ltm_decision_threshold: float = 0.5        # τ: retrieve iff P(need_mem) > τ
    ltm_prior_bias: float = 0.4                # log-odds tilt: "prefer LTM, don't force"
    ltm_length_weight: float = 0.8             # β on the memory-pressure prior
    ltm_length_logit_cap: float = 3.0          # |logit(P_len)| bound (audit A11; unbounded ⇒ always retrieve)
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
    timeline_max_transitions: int = 8
    evolution_era_buckets: int = 4
    evolution_per_era: int = 3

    # (The seven DI3_* knobs lived here until D8, 2026-07-27, and went with the
    #  module. Note for anyone reading old configs: DI3_LTM_REFERENCE_DENSITY_
    #  THRESHOLD never did anything — it was the "past ten turns" tier of the
    #  reference rule, and no caller ever passed conversation_length.)

    # G31: anchored to the install, not the CWD. As ".env" this was read only
    # when ICE happened to be launched from the repo root; anywhere else
    # pydantic-settings found no file and EVERY setting silently took its code
    # default. That is not hypothetical — `confidence_fallback_threshold` is
    # 0.5 here and 0.75 below, and it gates the orchestrator's wide-net
    # fallback, so the launch directory decided which retrieval path ran.
    # Real environment variables still win over the file; only the file's
    # location changed.
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env",
                                      env_file_encoding="utf-8")


settings = Settings()
