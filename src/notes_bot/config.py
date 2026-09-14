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
    # pgvector cosine_distance (0 = identical direction). Unset (None) keeps
    # the old always-return-up-to-limit behavior.
    #
    # 0.5 was the first guess and turned out far too loose: for short note
    # texts, intfloat/multilingual-e5-base packs everything into a
    # surprisingly narrow band regardless of topic — a real production
    # query ("порекомендуй уличную еду в тбилиси") measured 0.145 against
    # the actually-relevant video note but only 0.206-0.221 against
    # unrelated grocery notes ("Купить хлеб", "Купить молоко"). 0.25 still
    # isn't a properly calibrated value, just tightened against that one
    # real data point — revisit once there's enough real query/note traffic
    # to calibrate against properly (e.g. by looking at the distance
    # distribution of hits users actually acted on vs. ignored).
    search_max_distance: float | None = Field(default=0.25, alias="SEARCH_MAX_DISTANCE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # notes-bot's /healthz + /readyz — see 06-deployment.md, "Health-пробы".
    health_port: int = Field(default=8080, alias="HEALTH_PORT")
    # notes-gc — see 08-roadmap.md M8 and 03-ingest.md.
    gc_retention_days: int = Field(default=30, alias="GC_RETENTION_DAYS")
    gc_stuck_processing_minutes: int = Field(default=30, alias="GC_STUCK_PROCESSING_MINUTES")
    gc_batch_size: int = Field(default=500, alias="GC_BATCH_SIZE")
    # Worker processes' Prometheus port — see 06-deployment.md, "Наблюдаемость".
    # notes-bot itself serves /metrics on HEALTH_PORT instead (see health.py).
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
