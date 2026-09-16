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
    # notes-translate — see 05-contracts.md, "Translate-сервис": both a
    # note's text (at index time) and a search query are translated to
    # translate_target_lang before embedding, so cross-lingual notes/queries
    # (measured: Hebrew query vs. Hebrew note 0.137, same query vs. a
    # Russian note 0.247) compare on equal footing. TRANSLATE_ENABLED=false
    # (e.g. before this service is deployed, or if it's unhealthy for a
    # while) falls back to embedding the original text/query untranslated —
    # the old, worse-but-working cross-lingual behavior, not a hard failure.
    translate_enabled: bool = Field(default=True, alias="TRANSLATE_ENABLED")
    translate_url: str = Field(default="http://notes-translate:8000", alias="TRANSLATE_URL")
    translate_target_lang: str = Field(default="eng_Latn", alias="TRANSLATE_TARGET_LANG")
    translate_timeout_s: float = Field(default=20.0, alias="TRANSLATE_TIMEOUT_S")
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
    # 0.5, then 0.25, were both too loose against one real example: for
    # short note texts, intfloat/multilingual-e5-base packs everything into
    # a surprisingly narrow band regardless of topic — a query
    # ("порекомендуй уличную еду в тбилиси") measured 0.145 against the
    # actually-relevant video note but only 0.206-0.221 against unrelated
    # grocery notes. 0.18 (this field's next value) cleared that gap but
    # turned out too tight against a second real example: a Hebrew note
    # ("5 bike routes in Tel Aviv") the query "где велодорожки" should have
    # matched sat at distance 0.197 — just past 0.18 — while the next
    # candidate (unrelated: beaches near Haifa) was 0.207. 0.20 is the
    # narrow band that satisfies both real examples at once (>0.197,
    # <0.206). An absolute cutoff is a blunt tool for a model this
    # compressed — a relative-margin threshold (keep hits within some delta
    # of the *best* hit's distance, rather than a fixed ceiling) would adapt
    # to that automatically and is worth switching to once there's enough
    # real query/note traffic to calibrate against properly, instead of
    # hand-tuning this single number every time a new example disagrees
    # with the last one.
    search_max_distance: float | None = Field(default=0.20, alias="SEARCH_MAX_DISTANCE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # notes-bot's /healthz + /readyz — see 06-deployment.md, "Health-пробы".
    health_port: int = Field(default=8080, alias="HEALTH_PORT")
    # notes-gc — see 08-roadmap.md M8 and 03-ingest.md.
    gc_retention_days: int = Field(default=30, alias="GC_RETENTION_DAYS")
    gc_stuck_processing_minutes: int = Field(default=30, alias="GC_STUCK_PROCESSING_MINUTES")
    gc_batch_size: int = Field(default=500, alias="GC_BATCH_SIZE")
    # A note reclaimed this many times without ever reaching 'done' is
    # abandoned instead: marked 'failed' with its own error, not re-enqueued
    # again. Without this, a source that can never succeed (a private
    # Instagram account, a since-deleted video) would be picked up, die, and
    # get reclaimed forever, once per GC_STUCK_PROCESSING_MINUTES window —
    # forever spending a worker slot on something that will never finish.
    gc_max_attempts: int = Field(default=3, alias="GC_MAX_ATTEMPTS")
    # `heavy` (Instagram/voice: yt-dlp download + ffmpeg + faster-whisper)
    # has no explicit RQ job_timeout today, so it inherits RQ's own default,
    # 180s — too tight the moment a worker pod has to fetch faster-whisper's
    # weights from Hugging Face cold (observed live: transcription of a
    # 2-minute reel was still running at 180s). This is a ceiling, not a
    # target — raising it costs nothing on a job that finishes in seconds.
    heavy_job_timeout_s: int = Field(default=900, alias="HEAVY_JOB_TIMEOUT_S")
    # Where a reclaimed or abandoned note is reported — see clients/alerts.py.
    # Both unset turns alerts off rather than failing to start, same as
    # ai-proxy's own AI_PROXY_TELEGRAM_* pair (this can point at the same
    # bot/chat; it is its own pair of settings because the two services are
    # deployed independently).
    alerts_telegram_bot_token: str | None = Field(default=None, alias="ALERTS_TELEGRAM_BOT_TOKEN")
    alerts_telegram_chat_id: str | None = Field(default=None, alias="ALERTS_TELEGRAM_CHAT_ID")
    # A note settles into 'done'/'failed' for good, so a genuine repeat of
    # the *same* alert for the *same* note cannot normally happen — this is
    # a safety net against a GC run racing itself, not a real dedup window.
    alerts_min_interval_minutes: float = Field(default=5.0, alias="ALERTS_MIN_INTERVAL_MINUTES")
    # Worker processes' Prometheus port — see 06-deployment.md, "Наблюдаемость".
    # notes-bot itself serves /metrics on HEALTH_PORT instead (see health.py).
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
