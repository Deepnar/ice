"""Factory that returns the background LLM client based on settings."""

from openai import OpenAI
from src.api.config import settings

def get_bg_client() -> OpenAI:
    """Return an OpenAI-compatible client for the background model."""
    if settings.background_model_mode == "shared":
        base_url = f"{settings.ollama_base_url}/v1"
    else:
        base_url = "http://localhost:8002/v1"
    return OpenAI(base_url=base_url, api_key="dummy")