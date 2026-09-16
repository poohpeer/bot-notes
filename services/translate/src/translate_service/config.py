"""Configuration — see docs/architecture/05-contracts.md, "Translate-сервис".

Mirrors embeddings_service/config.py's shape — same env-file/prefix pattern,
same defaulted-vs-required split.
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
        env_prefix="TRANSLATE_",
    )

    model_name: str = Field(
        default="facebook/nllb-200-distilled-600M", alias="TRANSLATE_MODEL_NAME"
    )
    # Weights are baked into the image at build time, same as embeddings —
    # see docs/architecture/06-deployment.md, "Образы".
    model_cache_dir: str = Field(default="/models", alias="TRANSLATE_MODEL_CACHE_DIR")

    # FLORES-200 code — the canonical language search compares embeddings
    # in, per bot-notes' notes-bot Settings.search_target_lang (same value,
    # kept in sync manually; there is no single shared config file across
    # the two services).
    default_target_lang: str = Field(default="eng_Latn", alias="TRANSLATE_DEFAULT_TARGET_LANG")

    max_batch_size: int = Field(default=64, alias="TRANSLATE_MAX_BATCH_SIZE")
    max_total_chars: int = Field(default=200_000, alias="TRANSLATE_MAX_TOTAL_CHARS")
    max_concurrency: int = Field(default=2, alias="TRANSLATE_MAX_CONCURRENCY")
    queue_timeout_s: float = Field(default=20.0, alias="TRANSLATE_QUEUE_TIMEOUT_S")

    log_level: str = Field(default="INFO", alias="TRANSLATE_LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
