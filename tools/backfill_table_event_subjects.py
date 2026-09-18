#!/usr/bin/env python
"""One-off backfill for `table_event` notes created before /events_table
started extracting per-event `subjects` (docs/architecture/03-ingest.md,
"Бэкфилл subjects") — without it, `bot/logic.py`'s
`_filter_table_event_hits_by_subject` has nothing to match against for
these older rows, so /search's subject filter is a no-op for them.

Re-extracts `subjects` from each note's own `raw_text` via a text-only LLM
call (notes_bot.events_table.extract_subjects_from_text) and merges the
result into tags/structured — never touches raw_text or status, so no
re-embed happens.

Usage:
    uv run python tools/backfill_table_event_subjects.py
    uv run python tools/backfill_table_event_subjects.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy import select

from notes_bot.clients.llm import LLMClient, NullLLMClient, ProxyAILLMClient
from notes_bot.config import Settings, get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.models import Note
from notes_bot.db.repositories import NoteRepository
from notes_bot.events_table import extract_subjects_from_text
from notes_bot.logging_setup import configure_logging

log = logging.getLogger(__name__)


def _get_llm_client(settings: Settings) -> LLMClient:
    """Same choice as queue/tasks.py's own `_get_llm_client` - duplicated
    rather than imported since that one is a private, task-module-local
    helper, and this script otherwise only depends on the db/domain
    layers, same as tools/import_telegram_export.py."""
    if not settings.llm_enabled:
        return NullLLMClient()
    return ProxyAILLMClient(settings.proxy_ai_url)


async def backfill(*, session_factory, llm, dry_run: bool, timeout_s: float) -> int:
    updated = 0
    async with session_factory() as session:
        result = await session.execute(
            select(Note).where(
                Note.source_type == "table_event",
                Note.deleted_at.is_(None),
            )
        )
        notes = result.scalars().all()
        repo = NoteRepository(session)

        for note in notes:
            if note.structured and "subjects" in note.structured:
                continue
            subjects = await extract_subjects_from_text(llm, note.raw_text, timeout_s=timeout_s)
            log.info("note_id=%s raw_text=%r -> subjects=%r", note.id, note.raw_text, subjects)
            if dry_run:
                continue
            await repo.merge_table_event_subjects(note.id, subjects=subjects)
            updated += 1

        if not dry_run:
            await session.commit()
    return updated


async def _main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    llm = _get_llm_client(settings)
    if isinstance(llm, NullLLMClient):
        log.warning("LLM_ENABLED=false — every note will backfill to an empty subjects list")

    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    updated = await backfill(
        session_factory=session_factory,
        llm=llm,
        dry_run=args.dry_run,
        timeout_s=args.timeout_s,
    )
    await engine.dispose()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="log what would change, write nothing"
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    updated = asyncio.run(_main_async(args))
    print(f"updated={updated}")


if __name__ == "__main__":
    main()
