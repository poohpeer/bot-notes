"""Entrypoint: the Telegram bot process (long polling, replicas=1).

See docs/architecture/08-roadmap.md. Exactly one replica: two `getUpdates`
consumers on the same token would split updates between them (01-context.md).
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as async_redis
from aiogram import Bot, Dispatcher
from aiohttp import web
from redis import Redis
from rq import Queue

from notes_bot.bot.commands import set_bot_commands
from notes_bot.bot.handlers import router
from notes_bot.bot.logic import Deps
from notes_bot.clients.embeddings import HttpEmbeddingClient
from notes_bot.clients.search_cache import SearchSessionCache
from notes_bot.clients.translate import HttpTranslateClient
from notes_bot.config import get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.health import create_health_app, mark_ready
from notes_bot.logging_setup import configure_logging
from notes_bot.metrics import register_queue_length_collector
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

    bot = Bot(token=settings.telegram_bot_token)

    # RQ's Queue is sync-only (no asyncio client), so it gets its own plain
    # redis-py connection alongside the async one used everywhere else — see
    # notes_bot/queue/queues.py. One connection, two Queue objects: a Queue
    # is just a named view over it, not its own client.
    rq_redis = Redis.from_url(settings.redis_url)
    register_queue_length_collector(rq_redis)

    health_app = create_health_app(bot)
    runner = web.AppRunner(health_app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", settings.health_port).start()
    log.info("health server listening on :%d", settings.health_port)

    try:
        # bot_username (below) needs the same call readiness does — see
        # Deps.bot_username's docstring — so this does both in one getMe.
        me = await bot.get_me()
        mark_ready(health_app)
    except Exception as exc:
        log.error("initial getMe failed: %s", exc)
        raise SystemExit(1) from exc

    # Not readiness-gating: the menu is cosmetic (commands.py's own
    # docstring) and BotFather already has whatever was set last, so a
    # failure here shouldn't hold up polling — log and move on.
    try:
        await set_bot_commands(bot)
    except Exception:
        log.exception("failed to set bot command menu")

    deps = Deps(
        session_factory=create_session_factory(engine),
        embedding_client=HttpEmbeddingClient(
            settings.embeddings_url,
            model_name=settings.embedding_model_name,
            dim=settings.embedding_dim,
        ),
        search_cache=SearchSessionCache(async_redis.from_url(settings.redis_url)),
        fast_queue=Queue("fast", connection=rq_redis),
        heavy_queue=Queue("heavy", connection=rq_redis),
        llm_queue=Queue("llm", connection=rq_redis),
        settings=settings,
        bot_username=me.username or "",
        translate_client=(
            HttpTranslateClient(
                settings.translate_url,
                target_lang=settings.translate_target_lang,
                timeout=settings.translate_timeout_s,
            )
            if settings.translate_enabled
            else None
        ),
    )

    dp = Dispatcher()
    dp.include_router(router)
    dp["deps"] = deps

    log.info("notes-bot starting up as @%s", me.username)
    await dp.start_polling(bot)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
