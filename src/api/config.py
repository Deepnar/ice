"""Configuration for the ICE FastAPI proxy."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://ice:ice_local_dev@localhost:5432/ice_db"
    redis_url: str = "redis://localhost:6379/0"
    ollama_base_url: str = "http://localhost:11434"
    classifier_threshold: float = 0.3
    confidence_fallback_threshold: float = 0.75
    classifier_model_path: str = "models/classifier/ice_classifier_v3_qwen_ft3.pt"
    label_schema_path: str = "data/labeled/label_schema.json"
    default_fallback_model: str = "qwen2.5:7b"
    background_model_mode: str = "dedicated"   # "dedicated" or "shared"          # ← new

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