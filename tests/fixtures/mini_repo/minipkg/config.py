"""Configuration for the mini fixture app."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str = "unset"
    db_url: str = "postgresql://localhost:5433/mini"
    debug: bool = False
