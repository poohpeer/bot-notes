from __future__ import annotations

import os

import pytest
import redis.asyncio as redis

from notes_bot.clients.pending_events_cache import PendingEventsCache
from notes_bot.events_table import ExtractedEvent

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def cache():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    client = redis.from_url(url)
    yield PendingEventsCache(client)
    await client.flushdb()
    await client.aclose()


def _event(**overrides) -> ExtractedEvent:
    defaults = dict(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="מבחן עברית")
    defaults.update(overrides)
    return ExtractedEvent(**defaults)


async def test_create_then_get_round_trips(cache):
    events = [_event(), _event(text="מבחן מתמטיקה", type="exam")]
    session_id = await cache.create(
        chat_id=-100,
        user_id=1,
        is_group=True,
        tab_name="שכבה י'",
        topic_tags=["школа"],
        events=events,
    )
    pending = await cache.get(session_id)
    assert pending is not None
    assert pending.chat_id == -100
    assert pending.user_id == 1
    assert pending.is_group is True
    assert pending.tab_name == "שכבה י'"
    assert pending.topic_tags == ["школа"]
    assert pending.extracted_events() == events


async def test_get_returns_none_for_an_unknown_session(cache):
    assert await cache.get("does-not-exist") is None


async def test_delete_removes_the_session(cache):
    session_id = await cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=[], events=[_event()]
    )
    await cache.delete(session_id)
    assert await cache.get(session_id) is None
