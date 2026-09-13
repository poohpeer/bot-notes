"""structlog JSON logging — see docs/architecture/06-deployment.md, "Наблюдаемость".

Required fields per log line, where applicable: note_id, user_id, chat_id,
job_id, source_type, duration_ms. Never logged: the bot token, note contents,
search query text — personal data.

Every module in this codebase logs via plain `logging.getLogger(__name__)`
(stdlib), not `structlog.get_logger()`. `structlog.stdlib.ProcessorFormatter`
bridges those stdlib calls through structlog's JSON renderer instead of
requiring every call site to migrate — the previous version of this function
configured structlog's own processor chain but never routed stdlib logging
(what every module actually uses) through it, so log lines rendered as
plain text, not JSON, despite this module's own docstring.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # A stdlib LogRecord (every `logging.getLogger(...)` call site in
        # this codebase) or an already-structlog-processed event —
        # remove_processors_meta strips the wrapper's internal metadata
        # either way before this renders it.
        processors=[
            # Promotes a stdlib call's `extra={...}` kwargs into real JSON
            # keys (note_id, job_id, ...) — must run before
            # remove_processors_meta strips the `_record` it reads from.
            structlog.stdlib.ExtraAdder(),
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
