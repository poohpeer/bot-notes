#!/usr/bin/env python
"""One-off importer for a Telegram Desktop export (result.json) — see
docs/architecture/03-ingest.md, "Импорт экспорта Telegram", and
08-roadmap.md M8.

Usage:
    uv run python tools/import_telegram_export.py result.json --chat-id -1001234567890
    uv run python tools/import_telegram_export.py result.json --chat-id -100... --dry-run
    uv run python tools/import_telegram_export.py result.json --chat-id -100... \\
        --user-map user_map.json

`--chat-id` is the *live* group's chat_id as the bot will see it going
forward (the export's own top-level "id" is not necessarily the same
number bot-api uses for supergroups) — always pass it explicitly rather
than trusting the export.

`--user-map` is an optional JSON file `{"Display Name": 123456789, ...}`
for messages whose author has no numeric `from_id` in the export (deleted
accounts, some old exports) — every message that resolves to neither a
`from_id` nor an entry in this map is skipped and counted, never guessed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from redis import Redis
from rq import Queue

from notes_bot.config import get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.repositories import NoteRepository
from notes_bot.domain.classify import classify_text_message
from notes_bot.logging_setup import configure_logging
from notes_bot.queue.queues import enqueue_process_note

log = logging.getLogger(__name__)

# Same routing as bot/logic.py's save_note — see 03-ingest.md, "Выбрать
# очередь: instagram и voice → heavy, всё остальное → fast". Duplicated
# rather than imported from bot.logic to keep this standalone script's
# dependency surface to the db/domain/queue layers only.
_HEAVY_SOURCE_TYPES = {"instagram", "voice"}


def _extract_text(raw: str | list) -> str:
    """Telegram Desktop's export stores `text` as either a plain string or
    a list of runs (plain strings interleaved with `{"type": ..., "text":
    ...}` dicts for links/mentions/bold/etc.) — this flattens either shape
    to plain text, same as what the user would have typed."""
    if isinstance(raw, str):
        return raw
    parts = []
    for run in raw:
        if isinstance(run, str):
            parts.append(run)
        elif isinstance(run, dict):
            parts.append(run.get("text", ""))
    return "".join(parts)


_FROM_ID_RE = re.compile(r"^user(\d+)$")


def _resolve_user_id(message: dict, user_map: dict[str, int]) -> int | None:
    from_id = message.get("from_id") or ""
    match = _FROM_ID_RE.match(from_id)
    if match:
        return int(match.group(1))
    from_name = message.get("from")
    if from_name and from_name in user_map:
        return user_map[from_name]
    return None


def _parse_date(message: dict) -> datetime:
    if "date_unixtime" in message:
        return datetime.fromtimestamp(int(message["date_unixtime"]), tz=UTC)
    return datetime.fromisoformat(message["date"]).replace(tzinfo=UTC)


class ImportReport:
    def __init__(self) -> None:
        self.imported = 0
        self.skipped_duplicate = 0
        self.skipped_non_text = 0
        self.skipped_no_user = 0

    def summary(self) -> str:
        return (
            f"imported={self.imported} "
            f"skipped_duplicate={self.skipped_duplicate} "
            f"skipped_non_text={self.skipped_non_text} "
            f"skipped_no_user={self.skipped_no_user}"
        )


async def import_export(
    *,
    messages: list[dict],
    chat_id: int,
    user_map: dict[str, int],
    session_factory,
    fast_queue: Queue,
    heavy_queue: Queue,
    dry_run: bool,
) -> ImportReport:
    report = ImportReport()

    for message in messages:
        if message.get("type") != "message":
            report.skipped_non_text += 1
            continue
        text = _extract_text(message.get("text", ""))
        if not text.strip():
            report.skipped_non_text += 1
            continue

        user_id = _resolve_user_id(message, user_map)
        if user_id is None:
            log.warning(
                "import: message id=%s skipped, no user_id (from=%r from_id=%r)",
                message.get("id"),
                message.get("from"),
                message.get("from_id"),
            )
            report.skipped_no_user += 1
            continue

        import_id = f"tg-export:{message['id']}"
        classification = classify_text_message(text)

        async with session_factory() as session:
            repo = NoteRepository(session)
            existing = await repo.get_by_import_id(chat_id=chat_id, import_id=import_id)
            if existing is not None:
                report.skipped_duplicate += 1
                continue

            if dry_run:
                report.imported += 1
                continue

            note = await repo.create_imported_note(
                user_id=user_id,
                chat_id=chat_id,
                source_type=classification.source_type,
                source_url=classification.source_url,
                raw_text=text,
                created_at=_parse_date(message),
                import_id=import_id,
            )
            await session.commit()

        target_queue = (
            heavy_queue if classification.source_type in _HEAVY_SOURCE_TYPES else fast_queue
        )
        enqueue_process_note(target_queue, note.id)
        report.imported += 1

    return report


async def _main_async(args: argparse.Namespace) -> ImportReport:
    settings = get_settings()
    configure_logging(settings.log_level)

    export = json.loads(Path(args.export_file).read_text(encoding="utf-8"))
    messages = export.get("messages", [])
    log.info("import: %d message(s) in export", len(messages))

    user_map: dict[str, int] = {}
    if args.user_map:
        user_map = json.loads(Path(args.user_map).read_text(encoding="utf-8"))

    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    rq_redis = Redis.from_url(settings.redis_url)
    fast_queue = Queue("fast", connection=rq_redis)
    heavy_queue = Queue("heavy", connection=rq_redis)

    report = await import_export(
        messages=messages,
        chat_id=args.chat_id,
        user_map=user_map,
        session_factory=session_factory,
        fast_queue=fast_queue,
        heavy_queue=heavy_queue,
        dry_run=args.dry_run,
    )
    await engine.dispose()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_file", help="path to Telegram Desktop's result.json")
    parser.add_argument(
        "--chat-id", type=int, required=True, help="target live chat_id (bot-api numbering)"
    )
    parser.add_argument("--user-map", help="optional JSON file {display name: telegram user_id}")
    parser.add_argument(
        "--dry-run", action="store_true", help="count what would be imported, write nothing"
    )
    args = parser.parse_args()

    report = asyncio.run(_main_async(args))
    print(report.summary())
    if report.skipped_no_user:
        print(
            f"{report.skipped_no_user} message(s) skipped for missing user_id — "
            "see warnings above, or pass --user-map",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
