"""Pending-confirmation cache for /events_table — mirrors
clients/search_cache.py's own shape (same TTL/session_id conventions):
parsed events sit here between the preview message and the
Сохранить/Отмена tap, since callback_data has Telegram's own ~64-byte
limit and can't hold the parsed table itself.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass

import redis.asyncio as redis

from notes_bot.events_table import ExtractedEvent

_TTL_SECONDS = 30 * 60
_SESSION_ID_BYTES = 6


@dataclass(frozen=True)
class PendingEvents:
    chat_id: int
    user_id: int
    is_group: bool
    tab_name: str
    topic_tags: list[str]
    events: list[dict]  # ExtractedEvent, as dicts (json-serializable)

    def extracted_events(self) -> list[ExtractedEvent]:
        return [ExtractedEvent(**e) for e in self.events]


class PendingEventsCache:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"notes:events_table:{session_id}"

    async def create(
        self,
        *,
        chat_id: int,
        user_id: int,
        is_group: bool,
        tab_name: str,
        topic_tags: list[str],
        events: list[ExtractedEvent],
    ) -> str:
        session_id = secrets.token_urlsafe(_SESSION_ID_BYTES)
        pending = PendingEvents(
            chat_id=chat_id,
            user_id=user_id,
            is_group=is_group,
            tab_name=tab_name,
            topic_tags=topic_tags,
            events=[asdict(e) for e in events],
        )
        await self._redis.set(self._key(session_id), json.dumps(asdict(pending)), ex=_TTL_SECONDS)
        return session_id

    async def get(self, session_id: str) -> PendingEvents | None:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        return PendingEvents(**json.loads(raw))

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
