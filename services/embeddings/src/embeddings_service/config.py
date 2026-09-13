"""Configuration — see docs/architecture/05-contracts.md, "Внутреннее устройство".

M1's open question (which of the two candidate models to run) is not
decided here: both are supported behind the same contract, and
EMBEDDING_MODEL_NAME picks one. Switching later needs no code change — only
a redeploy, per ADR-2.
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
        env_prefix="EMBEDDINGS_",
    )

    model_name: str = Field(default="intfloat/multilingual-e5-base", alias="EMBEDDINGS_MODEL_NAME")
    # Weights are baked into the image at build time — see
    # docs/architecture/06-deployment.md, "Образы". This is only where the
    # baked cache lives inside the container.
    model_cache_dir: str = Field(default="/models", alias="EMBEDDINGS_MODEL_CACHE_DIR")

    max_batch_size: int = Field(default=64, alias="EMBEDDINGS_MAX_BATCH_SIZE")
    max_total_chars: int = Field(default=200_000, alias="EMBEDDINGS_MAX_TOTAL_CHARS")
    # Concurrent /embed runs sharing the one loaded model — see "Внутреннее
    # устройство": a semaphore, then a queue with a timeout, then 429.
    max_concurrency: int = Field(default=4, alias="EMBEDDINGS_MAX_CONCURRENCY")
    queue_timeout_s: float = Field(default=10.0, alias="EMBEDDINGS_QUEUE_TIMEOUT_S")

    log_level: str = Field(default="INFO", alias="EMBEDDINGS_LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
