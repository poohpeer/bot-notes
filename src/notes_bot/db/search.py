"""Two-phase ANN search — see docs/architecture/04-search.md.

The design-doc query sorts by `n.id` under `DISTINCT ON`, which returns the
six lowest-id notes instead of the six most relevant (ADR-4). This builds
the corrected query instead: ANN candidates -> best chunk per note (DISTINCT
ON note_id, still ordered by distance) -> page ordered by distance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import ColumnElement, select
from sqlalchemy.ext.asyncio import AsyncSession

from notes_bot.db.models import Note, NoteChunk
from notes_bot.metrics import SEARCH_DURATION_SECONDS


@dataclass(frozen=True)
class SearchHit:
    note_id: int
    title: str | None
    source_url: str | None
    source_type: str
    tags: list[str]
    chunk_text: str
    distance: float
    structured: dict
    summary: str | None


async def search_notes(
    session: AsyncSession,
    *,
    query_vector: list[float],
    acl_predicate: ColumnElement[bool],
    active_model: str,
    candidate_k: int,
    limit: int,
    offset: int,
    max_distance: float | None = None,
) -> list[SearchHit]:
    """Returns up to `limit` hits, ordered by relevance, starting at `offset`.

    Pass `limit = page_size + 1` to learn whether another page exists (the
    caller drops the extra row before rendering) — see 04-search.md,
    "Пагинация".

    `max_distance` (pgvector cosine_distance, 0 = identical direction) drops
    hits past it instead of always returning up to `limit` regardless of how
    weak the match is — with few notes in the corpus, the corpus itself
    doesn't out-compete an unrelated top match into oblivion.
    """
    started = time.perf_counter()
    try:
        return await _search_notes(
            session,
            query_vector=query_vector,
            acl_predicate=acl_predicate,
            active_model=active_model,
            candidate_k=candidate_k,
            limit=limit,
            offset=offset,
            max_distance=max_distance,
        )
    finally:
        # 06-deployment.md, "Латентность /search p50/p95" — timed around
        # the whole call, success or failure, since a slow failure is still
        # a latency problem worth seeing on the graph.
        SEARCH_DURATION_SECONDS.observe(time.perf_counter() - started)


async def _search_notes(
    session: AsyncSession,
    *,
    query_vector: list[float],
    acl_predicate: ColumnElement[bool],
    active_model: str,
    candidate_k: int,
    limit: int,
    offset: int,
    max_distance: float | None,
) -> list[SearchHit]:
    distance = NoteChunk.embedding.cosine_distance(query_vector)

    ann = (
        select(NoteChunk.note_id, NoteChunk.chunk_text, distance.label("distance"))
        .join(Note, Note.id == NoteChunk.note_id)
        .where(
            Note.deleted_at.is_(None),
            Note.status == "done",
            NoteChunk.embedding_model == active_model,
            acl_predicate,
        )
        .order_by(distance)
        .limit(candidate_k)
        .cte("ann")
    )

    # Best chunk per note: DISTINCT ON requires its own ORDER BY starting
    # with the DISTINCT ON column, but that ordering is scoped to this CTE
    # only — the outer query re-sorts by distance, which is the bug ADR-4
    # fixes relative to the design doc's single combined query.
    best = (
        select(ann.c.note_id, ann.c.chunk_text, ann.c.distance)
        .distinct(ann.c.note_id)
        .order_by(ann.c.note_id, ann.c.distance)
        .cte("best")
    )

    page_stmt = (
        select(
            Note.id,
            Note.title,
            Note.source_url,
            Note.source_type,
            Note.tags,
            Note.structured,
            Note.summary,
            best.c.chunk_text,
            best.c.distance,
        )
        .join(Note, Note.id == best.c.note_id)
        .order_by(best.c.distance)
        .limit(limit)
        .offset(offset)
    )
    if max_distance is not None:
        page_stmt = page_stmt.where(best.c.distance <= max_distance)

    result = await session.execute(page_stmt)
    return [
        SearchHit(
            note_id=row.id,
            title=row.title,
            source_url=row.source_url,
            source_type=row.source_type,
            tags=row.tags,
            structured=row.structured,
            summary=row.summary,
            chunk_text=row.chunk_text,
            distance=row.distance,
        )
        for row in result
    ]
