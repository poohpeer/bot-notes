"""Search-session cache — see docs/architecture/04-search.md, "Пагинация".

Caches the ranked list of note_ids, not the query vector: re-running ANN per
page can return a different order, which would show the user duplicates or
gaps between pages (the design doc's mistake, corrected here per that same
section).
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import redis.asyncio as redis

_TTL_SECONDS = 30 * 60
_SESSION_ID_BYTES = 6  # short enough for callback_data, long enough to not collide


@dataclass(frozen=True)
class SearchSession:
    query: str
    mode: str
    scope_chat_id: int | None
    note_ids: list[int]
    # Parallel to note_ids: each note's best-matching chunk text at the time
    # of the original ANN run. A small addition to the schema in
    # 04-search.md — that doc caches only note_ids and re-derives everything
    # else from Postgres per page. Note-level fields (title, source_url,
    # tags) still get re-fetched per page the same way, so a note deleted
    # between pages still disappears correctly; only the snippet is cached,
    # because re-deriving "the chunk closest to the original query" per page
    # would mean either caching the query vector (which 04-search.md
    # explicitly avoids, to keep ordering stable) or paying for another
    # embed_query call on every "show more" tap.
    fragments: list[str]
    offset: int
    created_at: str


class SearchSessionCache:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def _key(self, user_id: int, session_id: str) -> str:
        return f"notes:search:{user_id}:{session_id}"

    async def create(
        self,
        *,
        user_id: int,
        query: str,
        mode: str,
        scope_chat_id: int | None,
        note_ids: list[int],
        fragments: list[str],
    ) -> str:
        if len(note_ids) != len(fragments):
            raise ValueError("note_ids and fragments must be the same length")
        session_id = secrets.token_urlsafe(_SESSION_ID_BYTES)
        session = SearchSession(
            query=query,
            mode=mode,
            scope_chat_id=scope_chat_id,
            # Up to 50, per 04-search.md — a search result page never needs
            # to page past that many candidates.
            note_ids=note_ids[:50],
            fragments=fragments[:50],
            offset=0,
            created_at=datetime.now(UTC).isoformat(),
        )
        await self._redis.set(
            self._key(user_id, session_id), json.dumps(asdict(session)), ex=_TTL_SECONDS
        )
        return session_id

    async def get(self, user_id: int, session_id: str) -> SearchSession | None:
        raw = await self._redis.get(self._key(user_id, session_id))
        if raw is None:
            return None
        return SearchSession(**json.loads(raw))

    async def set_offset(self, user_id: int, session_id: str, offset: int) -> None:
        """Only meaningful on an existing, unexpired session — a caller that
        raced an expiry gets nothing to update, and the next `get` reports
        the session as gone rather than silently reviving it."""
        session = await self.get(user_id, session_id)
        if session is None:
            return
        updated = SearchSession(**{**asdict(session), "offset": offset})
        await self._redis.set(
            self._key(user_id, session_id), json.dumps(asdict(updated)), ex=_TTL_SECONDS
        )
