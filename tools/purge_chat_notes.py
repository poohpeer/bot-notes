#!/usr/bin/env python
"""Bulk soft-delete every note in one chat — see
docs/architecture/03-ingest.md, "Массовое удаление".

Same reversibility as any single `/trash`-restored note: this only sets
`deleted_at`, never touches the row itself. Restorable one by one via
`/trash` until GC_RETENTION_DAYS (default 30) hard-deletes it for good —
this script has no "undo everything at once" counterpart, so re-check
--dry-run's count before running for real if that matters.

Deliberately not scoped to one `user_id` (unlike the bot's own
`/trash` delete button) — a group chat's notes are commonly saved by
several different people, and "wipe this whole chat" is an operator
action, not a per-owner one.

Usage:
    uv run python tools/purge_chat_notes.py --chat-id -5118196918 --dry-run
    uv run python tools/purge_chat_notes.py --chat-id -5118196918
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import func, select

from notes_bot.config import get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.models import Note
from notes_bot.db.repositories import NoteRepository


async def purge(*, session_factory, chat_id: int, dry_run: bool) -> int:
    async with session_factory() as session:
        if dry_run:
            result = await session.execute(
                select(func.count())
                .select_from(Note)
                .where(Note.chat_id == chat_id, Note.deleted_at.is_(None))
            )
            return result.scalar_one()

        count = await NoteRepository(session).soft_delete_by_chat(chat_id)
        await session.commit()
        return count


async def _main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    count = await purge(session_factory=session_factory, chat_id=args.chat_id, dry_run=args.dry_run)
    await engine.dispose()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-id", type=int, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="only count what would be deleted, write nothing"
    )
    args = parser.parse_args()

    count = asyncio.run(_main_async(args))
    verb = "would soft-delete" if args.dry_run else "soft-deleted"
    print(f"{verb} {count} note(s) in chat_id={args.chat_id}")


if __name__ == "__main__":
    main()
