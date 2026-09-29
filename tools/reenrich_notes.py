#!/usr/bin/env python
"""Re-run enrich_note for every eligible note — see
docs/architecture/03-ingest.md, "Шаг 6. Обогащение через ai-proxy".

Enqueues `enrich_note` (the `llm` queue) for every non-deleted,
`status='done'` note whose `source_type` isn't `table_event` —
`table_event` notes set their own tags at creation time and deliberately
never go through `enrich_note` (`NoteRepository.set_tags_and_skip_enrich`'s
own docstring explains why: it would overwrite already-good tags with a
worse guess made from one short event line). Running enrich_note() on
those would do exactly that, so they're excluded unconditionally.

Same job_id convention as the normal post-processing enqueue
(`enrich_note-{note_id}`) — a note already mid-enrichment just gets its
existing job re-targeted, not duplicated.

Usage:
    uv run python tools/reenrich_notes.py --dry-run
    uv run python tools/reenrich_notes.py
"""

from __future__ import annotations

import argparse

from redis import Redis
from rq import Queue
from sqlalchemy import select

from notes_bot.config import get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.models import Note


async def find_eligible_note_ids(session_factory) -> list[int]:
    async with session_factory() as session:
        result = await session.execute(
            select(Note.id)
            .where(
                Note.deleted_at.is_(None),
                Note.status == "done",
                Note.source_type != "table_event",
            )
            .order_by(Note.id)
        )
        return list(result.scalars().all())


async def _main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    note_ids = await find_eligible_note_ids(session_factory)
    await engine.dispose()

    if args.dry_run:
        for note_id in note_ids:
            print(f"would enqueue enrich_note for note_id={note_id}")
        return len(note_ids)

    queue = Queue("llm", connection=Redis.from_url(settings.redis_url))
    for note_id in note_ids:
        queue.enqueue("notes_bot.queue.tasks.enrich_note", note_id, job_id=f"enrich_note-{note_id}")
    return len(note_ids)


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="only list what would be enqueued")
    args = parser.parse_args()

    count = asyncio.run(_main_async(args))
    verb = "would enqueue" if args.dry_run else "enqueued"
    print(f"{verb} enrich_note for {count} note(s)")


if __name__ == "__main__":
    main()
