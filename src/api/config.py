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