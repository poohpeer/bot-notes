"""Dedup-conflict cache for /events_table — see docs/architecture/03-ingest.md,
"/events_table". Holds the "same date range + type, different text" cases
`bot/logic.py:confirm_events_table` couldn't resolve on its own, between the
per-conflict "старое"/"новое" message and the tap that answers it.

Separate from PendingEventsCache: that one holds an entire *unconfirmed*
table (before any note exists); this one holds only the handful of
conflicts a *confirmed* batch produced against notes that already exist.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass

import redis.asyncio as redis

_TTL_SECONDS = 30 * 60
_SESSION_ID_BYTES = 6


@dataclass(frozen=True)
class PendingConflict:
    note_id: int
    old_text: str
    new_text: str
    new_tags: list[str]
    new_structured: dict


@dataclass(frozen=True)
class PendingConflicts:
    user_id: int
    conflicts: list[dict]  # PendingConflict, as dicts

    def items(self) -> list[PendingConflict]:
        return [PendingConflict(**c) for c in self.conflicts]


class ConflictsCache:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"notes:events_table_conflicts:{session_id}"

    async def create(self, *, user_id: int, conflicts: list[PendingConflict]) -> str:
        session_id = secrets.token_urlsafe(_SESSION_ID_BYTES)
        pending = PendingConflicts(user_id=user_id, conflicts=[asdict(c) for c in conflicts])
        await self._redis.set(self._key(session_id), json.dumps(asdict(pending)), ex=_TTL_SECONDS)
        return session_id

    async def get(self, session_id: str) -> PendingConflicts | None:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        return PendingConflicts(**json.loads(raw))
