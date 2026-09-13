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
