"""structlog JSON logging — see docs/architecture/06-deployment.md, "Наблюдаемость".

Required fields per log line, where applicable: note_id, user_id, chat_id,
job_id, source_type, duration_ms. Never logged: the bot token, note contents,
search query text — personal data.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )
