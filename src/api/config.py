"""Configuration for the ICE FastAPI proxy."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://ice:ice_local_dev@localhost:5432/ice_db"
    redis_url: str = "redis://localhost:6379/0"
    ollama_base_url: str = "http://localhost:11434"
    classifier_threshold: float = 0.3
    confidence_fallback_threshold: float = 0.75
    classifier_model_path: str = "models/classifier/ice_classifier_v2_final.pt"
    label_schema_path: str = "data/labeled/label_schema.json"
    default_fallback_model: str = "qwen2.5:7b"          # ← new

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()