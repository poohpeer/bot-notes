from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
import redis as sync_redis
from rq import Queue
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.cli.gc import hard_delete_expired, reclaim_stuck
from notes_bot.db.models import Note

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def factory(db_engine):
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    yield sf
    async with sf() as session:
        await session.execute(delete(Note).where(Note.user_id == 999))
        await session.commit()


@pytest.fixture
def rq_queues():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    conn = sync_redis.from_url(url)
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

    reclaimed = await reclaim_stuck(
        factory, fast_queue, heavy_queue, stuck_minutes=30, batch_size=500
    )
    assert reclaimed == 2

    async with factory() as session:
        assert (await session.get(Note, text_id)).status == "pending"
        assert (await session.get(Note, voice_id)).status == "pending"
        assert (await session.get(Note, recent_id)).status == "processing"

    assert fast_queue.job_ids == [f"process_note-{text_id}"]
    assert heavy_queue.job_ids == [f"process_note-{voice_id}"]


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

    await reclaim_stuck(factory, fast_queue, fast_queue, stuck_minutes=30, batch_size=500)
    # Simulate a second GC pass finding it stuck again (e.g. still pending,
    # never picked up).
    await _backdate(factory, note_id, column="updated_at", when=stuck_cutoff)
    await reclaim_stuck(factory, fast_queue, fast_queue, stuck_minutes=30, batch_size=500)

    assert set(fast_queue.job_ids) == {f"process_note-{note_id}"}
