"""Entrypoint: the notes-gc CronJobs, see docs/architecture/08-roadmap.md M8
and 03-ingest.md. `--job` picks which of the two runs — see deploy/k8s/gc.yaml
(daily hard-delete) and gc-reclaim.yaml (every 5 minutes) for why they are
split rather than one daily job doing both.

Two independent jobs, each idempotent and safe to run repeatedly:

1. Hard-delete notes soft-deleted more than `GC_RETENTION_DAYS` ago
   (`idx_notes_gc`). `note_chunks` cascades via `ON DELETE CASCADE`. Not
   time-sensitive — daily is plenty.
2. Reclaim notes stuck in `pending`/`processing` because the worker that
   owned them died mid-job (`idx_notes_unfinished`) — RQ never retries a
   job whose worker process is simply gone, and a SIGALRM-based RQ
   job_timeout cannot be trusted to record a reason either (queue/tasks.py's
   process_note_async docstring on why). Re-enqueues each one into the
   queue its `source_type` belongs in (07-decisions.md, ADR-6/ADR-8: the
   deterministic `job_id` makes a duplicate enqueue harmless even if the
   original job is still, somehow, alive) — unless it has already been
   reclaimed `GC_MAX_ATTEMPTS` times, in which case it is abandoned instead
   (marked 'failed') so a source that can never succeed does not get
   reclaimed forever. Time-sensitive: this is the whole recovery path for a
   note whose worker died, so it runs every few minutes, not daily.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from redis import Redis
from rq import Queue
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.clients.alerts import AlertNotifier, build_notifier
from notes_bot.config import Settings, get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.repositories import NoteRepository
from notes_bot.logging_setup import configure_logging
from notes_bot.queue.queues import enqueue_process_note

log = logging.getLogger(__name__)

_HEAVY_SOURCE_TYPES = {"instagram", "voice"}

Job = Literal["all", "hard-delete", "reclaim"]


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
    alerts: AlertNotifier,
    *,
    stuck_minutes: int,
    max_attempts: int,
    heavy_job_timeout_s: int,
    batch_size: int,
) -> tuple[int, int]:
    """Returns (reclaimed, abandoned) counts."""
    before = datetime.now(UTC) - timedelta(minutes=stuck_minutes)
    total_reclaimed = 0
    total_abandoned = 0
    while True:
        async with session_factory() as session:
            repo = NoteRepository(session)
            result = await repo.reclaim_stuck(
                before=before, limit=batch_size, max_attempts=max_attempts
            )
            await session.commit()

        for note in result.reclaimed:
            is_heavy = note.source_type in _HEAVY_SOURCE_TYPES
            queue = heavy_queue if is_heavy else fast_queue
            enqueue_process_note(
                queue, note.id, job_timeout=heavy_job_timeout_s if is_heavy else None
            )
            await alerts.reclaimed(
                note_id=note.id,
                source_type=note.source_type,
                source_url=note.source_url,
                attempt=note.attempts,
                max_attempts=max_attempts,
                stuck_minutes=stuck_minutes,
            )
        for note in result.abandoned:
            await alerts.abandoned(
                note_id=note.id,
                source_type=note.source_type,
                source_url=note.source_url,
                attempts=note.attempts,
                reason=note.error or "",
            )

        total_reclaimed += len(result.reclaimed)
        total_abandoned += len(result.abandoned)
        log.info(
            "notes-gc: reclaimed=%d abandoned=%d stuck note(s) this batch",
            len(result.reclaimed),
            len(result.abandoned),
        )
        # The repository reads at most one shared `batch_size` worth of
        # candidates and splits them between the two outcomes (see
        # NoteRepository.reclaim_stuck) — a full batch means there may be
        # more still waiting, regardless of how it split.
        if len(result.reclaimed) + len(result.abandoned) < batch_size:
            return total_reclaimed, total_abandoned


async def run_async(settings: Settings, *, job: Job) -> None:
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    if job in ("all", "hard-delete"):
        deleted = await hard_delete_expired(
            session_factory,
            retention_days=settings.gc_retention_days,
            batch_size=settings.gc_batch_size,
        )
        log.info("notes-gc: hard_deleted=%d", deleted)

    if job in ("all", "reclaim"):
        rq_redis = Redis.from_url(settings.redis_url)
        fast_queue = Queue("fast", connection=rq_redis)
        heavy_queue = Queue("heavy", connection=rq_redis)
        alerts = build_notifier(
            settings.alerts_telegram_bot_token,
            settings.alerts_telegram_chat_id,
            min_interval_minutes=settings.alerts_min_interval_minutes,
        )
        reclaimed, abandoned = await reclaim_stuck(
            session_factory,
            fast_queue,
            heavy_queue,
            alerts,
            stuck_minutes=settings.gc_stuck_processing_minutes,
            max_attempts=settings.gc_max_attempts,
            heavy_job_timeout_s=settings.heavy_job_timeout_s,
            batch_size=settings.gc_batch_size,
        )
        log.info("notes-gc: reclaimed=%d abandoned=%d", reclaimed, abandoned)

    await engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        choices=("all", "hard-delete", "reclaim"),
        default="all",
        help="which of the two GC jobs to run — see this module's docstring",
    )
    return parser.parse_args()


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()
    asyncio.run(run_async(settings, job=args.job))


if __name__ == "__main__":
    run()
