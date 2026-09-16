"""Startup validation — see docs/architecture/05-contracts.md, "Конфигурация".

Failing fast here is deliberate: required env vars are enforced by
pydantic-settings at Settings() construction, and this adds the checks that
need a live connection — Postgres, Redis, and the embedding service's
declared dimension. Better to crash at startup than serve traffic half-broken.
"""

from __future__ import annotations

import logging

import httpx
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from notes_bot.config import Settings

log = logging.getLogger(__name__)


class StartupCheckFailed(RuntimeError):
    pass


async def validate_startup(settings: Settings, engine: AsyncEngine) -> None:
    await _check_database(engine)
    await _check_redis(settings)
    await _check_embedding_dim(settings)
    await _check_translate_reachable(settings)


async def _check_database(engine: AsyncEngine) -> None:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed failure
        raise StartupCheckFailed(f"database unreachable: {exc}") from exc


async def _check_redis(settings: Settings) -> None:
    client = redis.from_url(settings.redis_url)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001
        raise StartupCheckFailed(f"redis unreachable: {exc}") from exc
    finally:
        await client.aclose()


async def _check_embedding_dim(settings: Settings) -> None:
    """EMBEDDING_DIM must match the embedding service, or vectors silently
    written at the wrong width would corrupt the VECTOR(768) column."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.embeddings_url}/model")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise StartupCheckFailed(f"embedding service unreachable: {exc}") from exc

    remote_dim = data.get("dim")
    if remote_dim != settings.embedding_dim:
        raise StartupCheckFailed(
            f"EMBEDDING_DIM={settings.embedding_dim} does not match "
            f"embedding service's dim={remote_dim}"
        )


async def _check_translate_reachable(settings: Settings) -> None:
    """Deliberately does not raise, unlike the checks above — notes-translate
    is a search-quality improvement (05-contracts.md, "Translate-сервис"),
    not a requirement to save or search notes at all (queue/tasks.py and
    bot/logic.py both fall back to embedding untranslated text on any
    failure here). Only logs, so a deploy where the NLLB image is still
    pulling doesn't crash-loop the bot over an optional dependency — but the
    log line still exists for the same "real proof" reason
    _check_embedding_dim's does, see 06-deployment.md's deploy verification
    step."""
    if not settings.translate_enabled:
        log.info("translate service: disabled (TRANSLATE_ENABLED=false)")
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.translate_url}/model")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001 — best-effort, see docstring
        log.warning("translate service unreachable at startup: %s", exc)
        return
    log.info("translate service reachable: model=%s", data.get("name"))
