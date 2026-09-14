from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
import redis as sync_redis
from rq import Queue
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.cli.gc import hard_delete_expired, reclaim_stuck, run_async
from notes_bot.config import Settings
from notes_bot.db.models import Note

pytestmark = pytest.mark.asyncio


class FakeAlerts:
    """Records calls instead of sending — mirrors ai-proxy's own
    FakeNotifier test double."""

    def __init__(self) -> None:
        self.reclaimed_calls: list[dict] = []
        self.abandoned_calls: list[dict] = []

    async def reclaimed(self, **kw):
        self.reclaimed_calls.append(kw)

    async def abandoned(self, **kw):
        self.abandoned_calls.append(kw)

    async def failed(self, **kw):
        pass


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/3")


def _real_settings(**overrides) -> Settings:
    base = dict(
        TELEGRAM_BOT_TOKEN="x",
        DATABASE_URL=os.environ.get(
            "DATABASE_URL", "postgresql+psycopg://postgres:test@localhost:5432/postgres"
        ),
        REDIS_URL=_redis_url(),
        EMBEDDINGS_URL="http://localhost:9",
        EMBEDDING_MODEL_NAME="test-model",
    )
    base.update(overrides)
    return Settings(**base)


@pytest_asyncio.fixture
async def factory(db_engine):
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    yield sf
    async with sf() as session:
        await session.execute(delete(Note).where(Note.user_id == 999))
        await session.commit()


@pytest.fixture
def rq_queues():
    conn = sync_redis.from_url(_redis_url())
    fast = Queue("fast", connection=conn)
    heavy = Queue("heavy", connection=conn)
    fast.empty()
    heavy.empty()
    yield fast, heavy
    fast.empty()
    heavy.empty()
    conn.close()


async def _make_note(factory, **overrides) -> int:
    fields = {
        "user_id": 999,
        "chat_id": 999,
        "is_group": False,
        "visibility": "private",
        "source_type": "text",
        "raw_text": "x",
        "status": "pending",
        **overrides,
    }
    async with factory() as session:
        note = Note(**fields)
        session.add(note)
        await session.commit()
        return note.id


async def _backdate(factory, note_id: int, *, column: str, when: datetime) -> None:
    async with factory() as session:
        await session.execute(update(Note).where(Note.id == note_id).values(**{column: when}))
        await session.commit()


async def test_hard_delete_expired_only_deletes_old_soft_deleted_notes(factory):
    old_id = await _make_note(factory)
    fresh_id = await _make_note(factory)
    live_id = await _make_note(factory)

    old_cutoff = datetime.now(UTC) - timedelta(days=40)
    fresh_cutoff = datetime.now(UTC) - timedelta(days=5)
    await _backdate(factory, old_id, column="deleted_at", when=old_cutoff)
    await _backdate(factory, fresh_id, column="deleted_at", when=fresh_cutoff)
    # live_id stays deleted_at=None

    deleted = await hard_delete_expired(factory, retention_days=30, batch_size=500)
    assert deleted == 1

    async with factory() as session:
        assert await session.get(Note, old_id) is None
        assert await session.get(Note, fresh_id) is not None
        assert await session.get(Note, live_id) is not None


async def test_hard_delete_expired_paginates_across_batches(factory):
    ids = [await _make_note(factory) for _ in range(3)]
    old_cutoff = datetime.now(UTC) - timedelta(days=40)
    for note_id in ids:
        await _backdate(factory, note_id, column="deleted_at", when=old_cutoff)

    deleted = await hard_delete_expired(factory, retention_days=30, batch_size=1)
    assert deleted == 3
    async with factory() as session:
        for note_id in ids:
            assert await session.get(Note, note_id) is None


async def test_reclaim_stuck_resets_status_and_requeues_by_source_type(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    text_id = await _make_note(factory, status="processing", source_type="text")
    voice_id = await _make_note(factory, status="processing", source_type="voice")
    recent_id = await _make_note(factory, status="processing", source_type="text")

    stuck_cutoff = datetime.now(UTC) - timedelta(minutes=60)
    await _backdate(factory, text_id, column="updated_at", when=stuck_cutoff)
    await _backdate(factory, voice_id, column="updated_at", when=stuck_cutoff)
    # recent_id keeps its just-created updated_at — not stuck yet

    alerts = FakeAlerts()
    reclaimed, abandoned = await reclaim_stuck(
        factory,
        fast_queue,
        heavy_queue,
        alerts,
        stuck_minutes=30,
        max_attempts=3,
        heavy_job_timeout_s=900,
        batch_size=500,
    )
    assert (reclaimed, abandoned) == (2, 0)

    async with factory() as session:
        assert (await session.get(Note, text_id)).status == "pending"
        assert (await session.get(Note, voice_id)).status == "pending"
        assert (await session.get(Note, recent_id)).status == "processing"

    assert fast_queue.job_ids == [f"process_note-{text_id}"]
    assert heavy_queue.job_ids == [f"process_note-{voice_id}"]
    assert {c["note_id"] for c in alerts.reclaimed_calls} == {text_id, voice_id}
    assert alerts.abandoned_calls == []


async def test_a_heavy_reclaim_carries_the_heavy_job_timeout(factory, rq_queues):
    """The re-enqueued job must not fall back to RQ's own 180s default —
    that default is exactly what stranded the note in the first place."""
    fast_queue, heavy_queue = rq_queues
    voice_id = await _make_note(factory, status="processing", source_type="voice")
    await _backdate(
        factory, voice_id, column="updated_at", when=datetime.now(UTC) - timedelta(minutes=60)
    )

    await reclaim_stuck(
        factory,
        fast_queue,
        heavy_queue,
        FakeAlerts(),
        stuck_minutes=30,
        max_attempts=3,
        heavy_job_timeout_s=900,
        batch_size=500,
    )

    [job] = heavy_queue.jobs
    assert job.timeout == 900


async def test_reclaim_stuck_requeue_uses_the_same_job_id_on_repeated_reclaims(factory, rq_queues):
    """RQ doesn't dedupe the queue's pending list by job_id (see
    queue/queues.py's enqueue_process_note docstring) — a second GC pass
    that finds the same note still stuck can push a second list entry, but
    both share one job_id, so it's still the same logical job re-armed, not
    a distinct one. process_note's own "already settled" check is what
    makes an eventual double execution harmless."""
    fast_queue, _ = rq_queues
    note_id = await _make_note(factory, status="processing", source_type="text")
    stuck_cutoff = datetime.now(UTC) - timedelta(minutes=60)
    await _backdate(factory, note_id, column="updated_at", when=stuck_cutoff)

    await reclaim_stuck(
        factory,
        fast_queue,
        fast_queue,
        FakeAlerts(),
        stuck_minutes=30,
        max_attempts=3,
        heavy_job_timeout_s=900,
        batch_size=500,
    )
    # Simulate a second GC pass finding it stuck again (e.g. still pending,
    # never picked up).
    await _backdate(factory, note_id, column="updated_at", when=stuck_cutoff)
    await reclaim_stuck(
        factory,
        fast_queue,
        fast_queue,
        FakeAlerts(),
        stuck_minutes=30,
        max_attempts=3,
        heavy_job_timeout_s=900,
        batch_size=500,
    )

    assert set(fast_queue.job_ids) == {f"process_note-{note_id}"}


async def test_a_note_stuck_past_max_attempts_is_abandoned_not_requeued(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    note_id = await _make_note(
        factory, status="processing", source_type="text", attempts=3, source_url=None
    )
    await _backdate(
        factory, note_id, column="updated_at", when=datetime.now(UTC) - timedelta(minutes=60)
    )

    alerts = FakeAlerts()
    reclaimed, abandoned = await reclaim_stuck(
        factory,
        fast_queue,
        heavy_queue,
        alerts,
        stuck_minutes=30,
        max_attempts=3,
        heavy_job_timeout_s=900,
        batch_size=500,
    )
    assert (reclaimed, abandoned) == (0, 1)

    async with factory() as session:
        note = await session.get(Note, note_id)
        assert note.status == "failed"
        assert note.attempts == 4
        assert note.error

    assert fast_queue.job_ids == [], "an abandoned note is not re-enqueued"
    [call] = alerts.abandoned_calls
    assert call["note_id"] == note_id
    assert call["attempts"] == 4
    assert call["reason"]


async def test_a_note_below_the_cap_is_reclaimed_not_abandoned(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    note_id = await _make_note(factory, status="processing", source_type="text", attempts=2)
    await _backdate(
        factory, note_id, column="updated_at", when=datetime.now(UTC) - timedelta(minutes=60)
    )

    reclaimed, abandoned = await reclaim_stuck(
        factory,
        fast_queue,
        heavy_queue,
        FakeAlerts(),
        stuck_minutes=30,
        max_attempts=3,
        heavy_job_timeout_s=900,
        batch_size=500,
    )
    assert (reclaimed, abandoned) == (1, 0)

    async with factory() as session:
        note = await session.get(Note, note_id)
        assert note.status == "pending"
        assert note.attempts == 3


class BrokenQueue:
    """Stands in for a Queue whose Redis connection is down — observed live:
    a `redis.exceptions.TimeoutError` from `enqueue()` used to abort the
    whole batch, silently, since nothing after it ever got a chance to run."""

    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def enqueue(self, *a, **kw):
        raise ConnectionError("simulated Redis outage")


async def test_an_enqueue_failure_does_not_take_the_rest_of_the_batch_down(factory, rq_queues):
    fast_queue, _ = rq_queues
    broken_id = await _make_note(factory, status="processing", source_type="voice")
    healthy_id = await _make_note(factory, status="processing", source_type="text")
    stuck_cutoff = datetime.now(UTC) - timedelta(minutes=60)
    await _backdate(factory, broken_id, column="updated_at", when=stuck_cutoff)
    await _backdate(factory, healthy_id, column="updated_at", when=stuck_cutoff)

    alerts = FakeAlerts()
    reclaimed, abandoned = await reclaim_stuck(
        factory,
        fast_queue,
        BrokenQueue(),  # the heavy queue, for the voice note
        alerts,
        stuck_minutes=30,
        max_attempts=3,
        heavy_job_timeout_s=900,
        batch_size=500,
    )
    assert (reclaimed, abandoned) == (2, 0)

    async with factory() as session:
        broken = await session.get(Note, broken_id)
        healthy = await session.get(Note, healthy_id)
        # Both were reclaimed in the database — that decision predates the
        # enqueue attempt and cannot be rolled back by its failure — but
        # only the healthy one actually reached the queue.
        assert broken.status == "pending"
        assert healthy.status == "pending"
    assert fast_queue.job_ids == [f"process_note-{healthy_id}"]
    assert {c["note_id"] for c in alerts.reclaimed_calls} == {healthy_id}, (
        "no alert for the one that was never actually re-enqueued"
    )


async def test_run_async_job_reclaim_skips_hard_delete(factory, rq_queues):
    """--job reclaim must not also hard-delete — the two run on independent
    schedules precisely so the frequent one stays cheap."""
    old_id = await _make_note(factory)
    await _backdate(
        factory, old_id, column="deleted_at", when=datetime.now(UTC) - timedelta(days=40)
    )

    await run_async(_real_settings(GC_RETENTION_DAYS="30"), job="reclaim")

    async with factory() as session:
        # Still soft-deleted, not hard-deleted — hard-delete never ran.
        assert await session.get(Note, old_id) is not None


async def test_run_async_job_hard_delete_skips_reclaim(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    note_id = await _make_note(factory, status="processing")
    await _backdate(
        factory, note_id, column="updated_at", when=datetime.now(UTC) - timedelta(minutes=60)
    )

    await run_async(_real_settings(GC_STUCK_PROCESSING_MINUTES="30"), job="hard-delete")

    async with factory() as session:
        # Still 'processing' — reclaim never ran.
        assert (await session.get(Note, note_id)).status == "processing"
    assert fast_queue.job_ids == []
