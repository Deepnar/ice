"""Configuration for the ICE FastAPI proxy."""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://ice:ice_local_dev@localhost:5432/ice_db"
    ollama_base_url: str = "http://localhost:11434"
    classifier_threshold: float = 0.3
    confidence_fallback_threshold: float = 0.75
    classifier_model_path: str = "models/classifier/ice_classifier_v3_qwen_ft3.pt"
    label_schema_path: str = "data/labeled/label_schema.json"
    default_fallback_model: str = "qwen2.5:7b"

    # ── G23/C17: store-level embedding identity (fail-loud) ──
    # The ONE embedder every writer and retrieval path shares
    # (src/memory/embedder.py). store_meta's 'embedding' row must agree with
    # these at boot — create_core() refuses to start on a mismatch, because
    # silently cosine-comparing mixed-width/mixed-model vectors is the
    # existential failure G23 exists to prevent. Changing either value
    # requires a migration + scripts/ice_reembed.py run. The classifier and
    # micro-NER consume slice384() of the same encode until B1/A9 retrain.
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

    # ── C7: in-process maintenance runtime (replaces Celery beat + Redis) ──
    # is_idle() gate for overdue-job dispatch: user quiet this long + no
    # generation in flight (today's 10s redis check was uselessly tight).
    user_active_threshold_seconds: int = 90
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

    context_input_fraction: float = 0.75
    context_budget_min: int = 4_000
    context_budget_max: int = 40_000
    context_budget_fallback: int = 23_000

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
    ltm_bump_reference: float = 1.2            # strong anaphora (DI3 reference rule) → bump
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

        # DI3 Configuration
    DI3_ENABLED: bool = True
    DI3_CODE_DENSITY_THRESHOLD: float = 0.3
    DI3_SENTIMENT_DENSITY_THRESHOLD: float = 0.4
    DI3_META_DENSITY_THRESHOLD: float = 0.2
    DI3_NOISE_DENSITY_THRESHOLD: float = 0.8
    DI3_REFERENCE_DENSITY_THRESHOLD: float = 0.2
    DI3_LTM_REFERENCE_DENSITY_THRESHOLD: float = 0.1

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()