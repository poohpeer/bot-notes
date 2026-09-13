"""Entrypoint: the Telegram bot process (long polling, replicas=1).

M0 wires configuration, logging, and startup validation only. Handlers,
keyboards, and the aiogram Dispatcher land in M2 — see
docs/architecture/08-roadmap.md.
"""

from __future__ import annotations

import asyncio
import logging

from notes_bot.config import get_settings
from notes_bot.db.engine import create_engine
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

    log.info("notes-bot starting up (handlers land in M2)")
    # TODO(M2): aiogram Dispatcher, long polling, command routing.


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
