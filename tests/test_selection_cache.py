from __future__ import annotations

import os

import pytest
import redis.asyncio as redis

from notes_bot.clients.selection_cache import SelectionCache

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def cache():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    client = redis.from_url(url)
    yield SelectionCache(client)
    await client.flushdb()
    await client.aclose()


async def test_toggle_adds_then_removes(cache):
    assert await cache.toggle(1, 100) is True
    assert await cache.get_ids(1) == [100]
    assert await cache.toggle(1, 100) is False
    assert await cache.get_ids(1) == []


async def test_toggle_is_scoped_per_user(cache):
    await cache.toggle(1, 100)
    await cache.toggle(2, 200)
    assert await cache.get_ids(1) == [100]
    assert await cache.get_ids(2) == [200]


async def test_is_selected(cache):
    assert await cache.is_selected(1, 100) is False
    await cache.toggle(1, 100)
    assert await cache.is_selected(1, 100) is True


async def test_clear_empties_the_cart(cache):
    await cache.toggle(1, 100)
    await cache.toggle(1, 101)
    await cache.clear(1)
    assert await cache.get_ids(1) == []


async def test_get_ids_on_an_empty_cart(cache):
    assert await cache.get_ids(999) == []
