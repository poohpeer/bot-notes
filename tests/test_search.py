"""Integration tests against real Postgres+pgvector — see tests/conftest.py.

Vectors only vary in the first two of 768 dimensions; the rest stay zero.
pgvector's cosine_distance doesn't require normalized inputs, so the exact
direction is what encodes "close" vs "far" from the query.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import true

from notes_bot.db.models import Note, NoteChunk
from notes_bot.db.search import search_notes

pytestmark = pytest.mark.asyncio

QUERY = [1.0, 0.0] + [0.0] * 766
MODEL = "test-model"
EVERYTHING_VISIBLE = true()


def _vec(x: float, y: float) -> list[float]:
    return [x, y] + [0.0] * 766


async def _make_note(session, *, status="done", deleted=False, **kw):
    kw.setdefault("user_id", 1)
    kw.setdefault("chat_id", 1)
    kw.setdefault("is_group", False)
    kw.setdefault("visibility", "private")
    kw.setdefault("source_type", "text")
    kw.setdefault("raw_text", "x")
    note = Note(status=status, **kw)
    if deleted:
        note.deleted_at = datetime.now(UTC)
    session.add(note)
    await session.flush()
    return note


async def _add_chunk(session, note, *, text, embedding, model=MODEL, index=0):
    session.add(
        NoteChunk(
            note_id=note.id,
            chunk_index=index,
            chunk_text=text,
            embedding=embedding,
            embedding_model=model,
        )
    )
    await session.flush()


async def test_ranks_by_relevance_not_by_note_id(db_session):
    """ADR-4 regression: the design doc's query would return the lowest-id
    notes; this must return the closest ones regardless of insertion order."""
    far = await _make_note(db_session)  # inserted first -> lowest id
    close = await _make_note(db_session)  # inserted second -> higher id
    await _add_chunk(db_session, far, text="far", embedding=_vec(0.0, 1.0))  # orthogonal
    await _add_chunk(db_session, close, text="close", embedding=_vec(0.99, 0.01))

    hits = await search_notes(
        db_session,
        query_vector=QUERY,
        acl_predicate=EVERYTHING_VISIBLE,
        active_model=MODEL,
        candidate_k=200,
        limit=10,
        offset=0,
    )
    assert [h.note_id for h in hits] == [close.id, far.id]


async def test_dedups_to_the_best_chunk_per_note(db_session):
    note = await _make_note(db_session)
    await _add_chunk(db_session, note, text="near", embedding=_vec(0.99, 0.01), index=0)
    await _add_chunk(db_session, note, text="far", embedding=_vec(0.0, 1.0), index=1)

    hits = await search_notes(
        db_session,
        query_vector=QUERY,
        acl_predicate=EVERYTHING_VISIBLE,
        active_model=MODEL,
        candidate_k=200,
        limit=10,
        offset=0,
    )
    assert len(hits) == 1
    assert hits[0].chunk_text == "near"


async def test_pagination_limit_and_offset(db_session):
    notes = []
    for i in range(5):
        n = await _make_note(db_session)
        # Decreasing similarity as i grows: n=0 is closest.
        await _add_chunk(db_session, n, text=f"n{i}", embedding=_vec(1.0 - i * 0.1, i * 0.1))
        notes.append(n)

    page1 = await search_notes(
        db_session,
        query_vector=QUERY,
        acl_predicate=EVERYTHING_VISIBLE,
        active_model=MODEL,
        candidate_k=200,
        limit=2,
        offset=0,
    )
    page2 = await search_notes(
        db_session,
        query_vector=QUERY,
        acl_predicate=EVERYTHING_VISIBLE,
        active_model=MODEL,
        candidate_k=200,
        limit=2,
        offset=2,
    )
    assert [h.note_id for h in page1] == [notes[0].id, notes[1].id]
    assert [h.note_id for h in page2] == [notes[2].id, notes[3].id]


async def test_excludes_soft_deleted_notes(db_session):
    visible = await _make_note(db_session)
    deleted = await _make_note(db_session, deleted=True)
    await _add_chunk(db_session, visible, text="v", embedding=_vec(1.0, 0.0))
    await _add_chunk(db_session, deleted, text="d", embedding=_vec(1.0, 0.0))

    hits = await search_notes(
        db_session,
        query_vector=QUERY,
        acl_predicate=EVERYTHING_VISIBLE,
        active_model=MODEL,
        candidate_k=200,
        limit=10,
        offset=0,
    )
    assert [h.note_id for h in hits] == [visible.id]


async def test_excludes_notes_not_yet_done(db_session):
    done = await _make_note(db_session, status="done")
    pending = await _make_note(db_session, status="pending")
    await _add_chunk(db_session, done, text="d", embedding=_vec(1.0, 0.0))
    await _add_chunk(db_session, pending, text="p", embedding=_vec(1.0, 0.0))

    hits = await search_notes(
        db_session,
        query_vector=QUERY,
        acl_predicate=EVERYTHING_VISIBLE,
        active_model=MODEL,
        candidate_k=200,
        limit=10,
        offset=0,
    )
    assert [h.note_id for h in hits] == [done.id]


async def test_excludes_chunks_from_a_different_embedding_model(db_session):
    """A model migration in progress must not mix distances from two
    models — see 02-data-model.md, "Смена embedding-модели"."""
    note = await _make_note(db_session)
    await _add_chunk(db_session, note, text="old", embedding=_vec(1.0, 0.0), model="old-model")

    hits = await search_notes(
        db_session,
        query_vector=QUERY,
        acl_predicate=EVERYTHING_VISIBLE,
        active_model=MODEL,
        candidate_k=200,
        limit=10,
        offset=0,
    )
    assert hits == []


async def test_acl_predicate_is_applied(db_session):
    from notes_bot.domain.acl import visibility_predicate

    mine = await _make_note(db_session, user_id=1, chat_id=1, visibility="private")
    others_private = await _make_note(db_session, user_id=2, chat_id=2, visibility="private")
    await _add_chunk(db_session, mine, text="mine", embedding=_vec(1.0, 0.0))
    await _add_chunk(db_session, others_private, text="theirs", embedding=_vec(1.0, 0.0))

    predicate = visibility_predicate(
        user_id=1, chat_id=1, is_group_chat=False, search_mode="mine_only"
    )
    hits = await search_notes(
        db_session,
        query_vector=QUERY,
        acl_predicate=predicate,
        active_model=MODEL,
        candidate_k=200,
        limit=10,
        offset=0,
    )
    assert [h.note_id for h in hits] == [mine.id]
