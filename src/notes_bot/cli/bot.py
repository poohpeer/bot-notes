"""Entrypoint: the Telegram bot process (long polling, replicas=1).

M2 scope: text notes in private chats, /search + /search_mine + /search_all
— see docs/architecture/08-roadmap.md. Exactly one replica: two `getUpdates`
consumers on the same token would split updates between them (01-context.md).
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as async_redis
from aiogram import Bot, Dispatcher
from redis import Redis
from rq import Queue

from notes_bot.bot.handlers import router
from notes_bot.bot.logic import Deps
from notes_bot.clients.embeddings import HttpEmbeddingClient
from notes_bot.clients.search_cache import SearchSessionCache
from notes_bot.config import get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.logging_setup import configure_logging
from notes_bot.startup import StartupCheckFailed, validate_startup

log = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    engine = create_engine(settings)
    try:
        await validate_startup(settings, engine)
    except StartupCheckFailed as exc:
        log.error("startup check failed: %s", exc)
        raise SystemExit(1) from exc

    deps = Deps(
        session_factory=create_session_factory(engine),
        embedding_client=HttpEmbeddingClient(
            settings.embeddings_url,
            model_name=settings.embedding_model_name,
            dim=settings.embedding_dim,
        ),
        search_cache=SearchSessionCache(async_redis.from_url(settings.redis_url)),
        # RQ's Queue is sync-only (no asyncio client), so it gets its own
        # plain redis-py connection alongside the async one used everywhere
        # else — see notes_bot/queue/queues.py.
        fast_queue=Queue("fast", connection=Redis.from_url(settings.redis_url)),
        settings=settings,
    )

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()
    dp.include_router(router)
    dp["deps"] = deps

    log.info("notes-bot starting up")
    await dp.start_polling(bot)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
