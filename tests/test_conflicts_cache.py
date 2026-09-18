from __future__ import annotations

import os

import pytest
import redis.asyncio as redis

from notes_bot.clients.conflicts_cache import ConflictsCache, PendingConflict

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def cache():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    client = redis.from_url(url)
    yield ConflictsCache(client)
    await client.flushdb()
    await client.aclose()


def _conflict(**overrides) -> PendingConflict:
    defaults = dict(
        note_id=1,
        old_text="было",
        new_text="стало",
        new_tags=["экзамен"],
        new_structured={"date_start": "01/09/2026", "date_end": "01/09/2026", "type": "exam"},
    )
    defaults.update(overrides)
    return PendingConflict(**defaults)


async def test_create_then_get_round_trips(cache):
    conflicts = [_conflict(), _conflict(note_id=2, old_text="a", new_text="b")]
    session_id = await cache.create(user_id=1, conflicts=conflicts)
    pending = await cache.get(session_id)
    assert pending is not None
    assert pending.user_id == 1
    assert pending.items() == conflicts


async def test_get_returns_none_for_an_unknown_session(cache):
    assert await cache.get("does-not-exist") is None
