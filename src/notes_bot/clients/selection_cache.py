"""Multi-select "cart" for /list's bulk delete — see
docs/architecture/03-ingest.md, "Массовое удаление". One Redis SET per
user, note_ids toggled in/out by ☐/☑️ on individual /list cards, all
deleted at once by /delete_selected.

Deliberately not the SearchSessionCache's whole-blob-on-a-session-id
shape: there's no "session" here to expire as a unit, just a running set
a single user builds up across possibly several /list pages before
acting on it — a plain SET with one key per user matches that better than
tracking a JSON blob and its own separate session id.
"""

from __future__ import annotations

import redis.asyncio as redis

_TTL_SECONDS = 30 * 60


class SelectionCache:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def _key(self, user_id: int) -> str:
        return f"notes:selection:{user_id}"

    async def toggle(self, user_id: int, note_id: int) -> bool:
        """Returns the new membership state (True = now selected). Each
        toggle refreshes the TTL — an actively-being-built cart shouldn't
        expire out from under someone still picking notes."""
        key = self._key(user_id)
        removed = await self._redis.srem(key, note_id)
        if removed:
            return False
        await self._redis.sadd(key, note_id)
        await self._redis.expire(key, _TTL_SECONDS)
        return True

    async def get_ids(self, user_id: int) -> list[int]:
        raw = await self._redis.smembers(self._key(user_id))
        return [int(note_id) for note_id in raw]

    async def is_selected(self, user_id: int, note_id: int) -> bool:
        return bool(await self._redis.sismember(self._key(user_id), note_id))

    async def clear(self, user_id: int) -> None:
        await self._redis.delete(self._key(user_id))
