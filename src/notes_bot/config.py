"""Application configuration via pydantic-settings.

Every value comes from the environment (and an optional `.env`), never from a
file baked into the image. Required values have no default: a missing one fails
startup instead of running half-configured — see `validate_startup`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Required — no defaults, absence is a startup failure.
    telegram_bot_token: str = Field(alias="TELEGRAM_BOT_TOKEN")
    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(alias="REDIS_URL")
    embeddings_url: str = Field(alias="EMBEDDINGS_URL")
    embedding_model_name: str = Field(alias="EMBEDDING_MODEL_NAME")

    # Defaulted.
    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")
    proxy_ai_url: str = Field(default="http://ai-proxy:8787", alias="PROXY_AI_URL")
    llm_enabled: bool = Field(default=False, alias="LLM_ENABLED")
    llm_enrich_timeout_s: float = Field(default=120.0, alias="LLM_ENRICH_TIMEOUT_S")
    llm_smart_search_timeout_s: float = Field(default=180.0, alias="LLM_SMART_SEARCH_TIMEOUT_S")
    whisper_model: str = Field(default="small", alias="WHISPER_MODEL")
    max_audio_seconds: int = Field(default=1200, alias="MAX_AUDIO_SECONDS")
    max_download_bytes: int = Field(default=104857600, alias="MAX_DOWNLOAD_BYTES")
    search_candidate_k: int = Field(default=200, alias="SEARCH_CANDIDATE_K")
    search_page_size: int = Field(default=5, alias="SEARCH_PAGE_SIZE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
