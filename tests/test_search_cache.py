from __future__ import annotations

import os

import pytest
import redis.asyncio as redis

from notes_bot.clients.search_cache import SearchSessionCache

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def cache():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    client = redis.from_url(url)
    yield SearchSessionCache(client)
    await client.flushdb()
    await client.aclose()


async def test_create_then_get_round_trips(cache):
    session_id = await cache.create(
        user_id=1,
        query="pasta",
        mode="all",
        scope_chat_id=None,
        note_ids=[3, 1, 2],
        fragments=["a", "b", "c"],
    )
    session = await cache.get(1, session_id)
    assert session is not None
    assert session.query == "pasta"
    assert session.mode == "all"
    assert session.note_ids == [3, 1, 2]
    assert session.fragments == ["a", "b", "c"]
    assert session.offset == 0


async def test_note_ids_and_fragments_are_capped_at_50(cache):
    session_id = await cache.create(
        user_id=1,
        query="q",
        mode="all",
        scope_chat_id=None,
        note_ids=list(range(80)),
        fragments=[str(i) for i in range(80)],
    )
    session = await cache.get(1, session_id)
    assert len(session.note_ids) == 50
    assert session.note_ids == list(range(50))
    assert session.fragments == [str(i) for i in range(50)]


async def test_mismatched_lengths_are_rejected(cache):
    with pytest.raises(ValueError):
        await cache.create(
            user_id=1,
            query="q",
            mode="all",
            scope_chat_id=None,
            note_ids=[1, 2],
            fragments=["only one"],
        )


async def test_get_unknown_session_returns_none(cache):
    assert await cache.get(1, "does-not-exist") is None


async def test_sessions_are_scoped_per_user(cache):
    session_id = await cache.create(
        user_id=1, query="q", mode="all", scope_chat_id=None, note_ids=[1], fragments=["x"]
    )
    # Same session_id string, different user — must not resolve.
    assert await cache.get(999, session_id) is None


async def test_set_offset_updates_the_stored_session(cache):
    session_id = await cache.create(
        user_id=1,
        query="q",
        mode="all",
        scope_chat_id=None,
        note_ids=[1, 2, 3, 4, 5],
        fragments=["a", "b", "c", "d", "e"],
    )
    await cache.set_offset(1, session_id, 5)
    session = await cache.get(1, session_id)
    assert session.offset == 5
    # Everything else survives the update untouched.
    assert session.note_ids == [1, 2, 3, 4, 5]
    assert session.query == "q"


async def test_set_offset_on_missing_session_is_a_no_op(cache):
    await cache.set_offset(1, "does-not-exist", 5)  # must not raise
    assert await cache.get(1, "does-not-exist") is None


async def test_scope_chat_id_round_trips_for_group_search(cache):
    session_id = await cache.create(
        user_id=1, query="q", mode="all", scope_chat_id=555, note_ids=[1], fragments=["x"]
    )
    session = await cache.get(1, session_id)
    assert session.scope_chat_id == 555
