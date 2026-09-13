"""Entrypoint: an RQ worker process.

Which queues it listens to is passed on the command line, since notes-worker-
fast (fast, llm) and notes-worker-heavy (heavy) share this same image and
entrypoint — see docs/architecture/06-deployment.md.

Job handlers (process_note, enrich_note, smart_answer) land in M2/M4/M5/M7.
"""

from __future__ import annotations

import logging
import sys

from redis import Redis
from rq import Worker

from notes_bot.config import get_settings
from notes_bot.logging_setup import configure_logging

log = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    queues = sys.argv[1:] or ["fast"]
    log.info("notes-worker starting, queues=%s", queues)

    redis_conn = Redis.from_url(settings.redis_url)
    worker = Worker(queues, connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    run()
