"""Entrypoint: the notes-gc CronJob (daily), see docs/architecture/08-roadmap.md M8
and 03-ingest.md.

Two independent jobs, each idempotent and safe to run repeatedly:

1. Hard-delete notes soft-deleted more than `GC_RETENTION_DAYS` ago
   (`idx_notes_gc`). `note_chunks` cascades via `ON DELETE CASCADE`.
2. Reclaim notes stuck in `pending`/`processing` because the worker that
   owned them died mid-job (`idx_notes_unfinished`) — RQ never retries a
   job whose worker process is simply gone, so without this a killed pod
   leaves notes unprocessed forever. Re-enqueues each one into the queue
   its `source_type` belongs in (07-decisions.md, ADR-6/ADR-8: the
   deterministic `job_id` makes a duplicate enqueue harmless even if the
   original job is still, somehow, alive).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from redis import Redis
from rq import Queue
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.config import Settings, get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.repositories import NoteRepository
from notes_bot.logging_setup import configure_logging
from notes_bot.queue.queues import enqueue_process_note

log = logging.getLogger(__name__)

_HEAVY_SOURCE_TYPES = {"instagram", "voice"}


async def hard_delete_expired(
    session_factory: async_sessionmaker, *, retention_days: int, batch_size: int
) -> int:
    before = datetime.now(UTC) - timedelta(days=retention_days)
    total = 0
    while True:
        async with session_factory() as session:
            repo = NoteRepository(session)
            deleted = await repo.hard_delete_expired(before=before, limit=batch_size)
            await session.commit()
        total += deleted
        log.info("notes-gc: hard-deleted %d expired note(s) this batch", deleted)
        if deleted < batch_size:
            return total


async def reclaim_stuck(
    session_factory: async_sessionmaker,
    fast_queue: Queue,
    heavy_queue: Queue,
    *,
    stuck_minutes: int,
    batch_size: int,
) -> int:
    before = datetime.now(UTC) - timedelta(minutes=stuck_minutes)
    total = 0
    while True:
        async with session_factory() as session:
            repo = NoteRepository(session)
            notes = await repo.reclaim_stuck(before=before, limit=batch_size)
            await session.commit()
        for note in notes:
            queue = heavy_queue if note.source_type in _HEAVY_SOURCE_TYPES else fast_queue
            enqueue_process_note(queue, note.id)
        total += len(notes)
        log.info("notes-gc: reclaimed and re-enqueued %d stuck note(s) this batch", len(notes))
        if len(notes) < batch_size:
            return total


async def run_async(settings: Settings) -> None:
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    rq_redis = Redis.from_url(settings.redis_url)
    fast_queue = Queue("fast", connection=rq_redis)
    heavy_queue = Queue("heavy", connection=rq_redis)

    deleted = await hard_delete_expired(
        session_factory,
        retention_days=settings.gc_retention_days,
        batch_size=settings.gc_batch_size,
    )
    reclaimed = await reclaim_stuck(
        session_factory,
        fast_queue,
        heavy_queue,
        stuck_minutes=settings.gc_stuck_processing_minutes,
        batch_size=settings.gc_batch_size,
    )
    log.info("notes-gc: done — hard_deleted=%d reclaimed=%d", deleted, reclaimed)
    await engine.dispose()


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(run_async(settings))


if __name__ == "__main__":
    run()
