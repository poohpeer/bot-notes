"""Entrypoint: the notes-gc CronJob (daily) — hard-delete, see docs/architecture/03-ingest.md.

Stubbed until M8.
"""

from __future__ import annotations

import logging

from notes_bot.config import get_settings
from notes_bot.logging_setup import configure_logging

log = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("notes-gc: nothing to do yet (lands in M8)")


if __name__ == "__main__":
    run()
