from __future__ import annotations

import pytest

from notes_bot.db.repositories import (
    ChunkRepository,
    NewChunk,
    NoteRepository,
    UserSettingsRepository,
)

pytestmark = pytest.mark.asyncio


def _vec(seed: float) -> list[float]:
    return [seed] * 768


async def test_create_text_note_defaults_to_pending_and_given_visibility(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=42,
        raw_text="hello",
        visibility="private",
    )
    assert note is not None
    assert note.status == "pending"
    assert note.visibility == "private"
    assert note.raw_text == "hello"


async def test_create_text_note_is_idempotent_on_chat_and_message_id(db_session):
    """ADR-8: a retried Telegram delivery must not double-save."""
    repo = NoteRepository(db_session)
    first = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=7, raw_text="a", visibility="private"
    )
    second = await repo.create_text_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=7,
        raw_text="a again",
        visibility="private",
    )
    assert first is not None
    assert second is None


async def test_mark_processing_then_done(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=1, raw_text="x", visibility="private"
    )
    await repo.mark_processing(note.id)
    await db_session.flush()
    assert (await repo.get(note.id)).status == "processing"

    await repo.mark_done(note.id)
    await db_session.flush()
    assert (await repo.get(note.id)).status == "done"


async def test_mark_failed_records_error_and_increments_attempts(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=2, raw_text="x", visibility="private"
    )
    await repo.mark_failed(note.id, "boom")
    await db_session.flush()
    refreshed = await repo.get(note.id)
    assert refreshed.status == "failed"
    assert refreshed.error == "boom"
    assert refreshed.attempts == 1


async def test_toggle_visibility_flips_private_to_public_and_back(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=3, raw_text="x", visibility="private"
    )
    assert await repo.toggle_visibility(note.id, user_id=1) == "public"
    assert await repo.toggle_visibility(note.id, user_id=1) == "private"


async def test_toggle_visibility_refuses_a_different_user(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=4, raw_text="x", visibility="private"
    )
    assert await repo.toggle_visibility(note.id, user_id=999) is None
    # Untouched.
    assert (await repo.get(note.id)).visibility == "private"


async def test_toggle_visibility_refuses_a_group_note(db_session):
    """Group notes have no visibility to toggle — enforced by the CHECK
    constraint; the WHERE clause matches zero rows instead of erroring."""
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=555, is_group=True, tg_message_id=5, raw_text="x", visibility=None
    )
    assert await repo.toggle_visibility(note.id, user_id=1) is None


async def test_replace_chunks_deletes_old_and_inserts_new(db_session):
    repo = NoteRepository(db_session)
    chunks_repo = ChunkRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=6, raw_text="x", visibility="private"
    )

    await chunks_repo.replace_chunks(
        note.id,
        [NewChunk(text="a", token_count=1, embedding=_vec(0.1))],
        embedding_model="test-model",
    )
    await db_session.flush()

    await chunks_repo.replace_chunks(
        note.id,
        [
            NewChunk(text="b", token_count=1, embedding=_vec(0.2)),
            NewChunk(text="c", token_count=1, embedding=_vec(0.3)),
        ],
        embedding_model="test-model",
    )
    await db_session.flush()

    from sqlalchemy import select

    from notes_bot.db.models import NoteChunk

    result = await db_session.execute(
        select(NoteChunk.chunk_text)
        .where(NoteChunk.note_id == note.id)
        .order_by(NoteChunk.chunk_index)
    )
    assert [row[0] for row in result] == ["b", "c"]


async def test_user_settings_get_or_create_returns_defaults(db_session):
    repo = UserSettingsRepository(db_session)
    settings = await repo.get_or_create(user_id=42)
    assert settings.search_mode == "all"
    assert settings.default_visibility == "private"


async def test_user_settings_get_or_create_is_idempotent(db_session):
    repo = UserSettingsRepository(db_session)
    first = await repo.get_or_create(user_id=42)
    second = await repo.get_or_create(user_id=42)
    assert first.user_id == second.user_id


async def test_create_note_with_a_source_url(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=100,
        source_type="page",
        source_url="https://example.com/article",
        raw_text="https://example.com/article",
        visibility="private",
    )
    assert note is not None
    assert note.source_type == "page"
    assert note.source_url == "https://example.com/article"


async def test_record_extraction_sets_text_lang_and_error(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=101,
        source_type="page",
        source_url="https://example.com/article",
        raw_text="https://example.com/article",
        visibility="private",
    )
    await repo.record_extraction(note.id, extracted_text="the page text", lang="ru", error=None)
    await db_session.flush()
    refreshed = await repo.get(note.id)
    assert refreshed.extracted_text == "the page text"
    assert refreshed.lang == "ru"
    assert refreshed.error is None


async def test_record_extraction_can_record_a_degraded_note(db_session):
    """extracted_text stays NULL and error explains why — but status is the
    caller's call, not this method's (03-ingest.md: still 'done', not
    'failed', as long as raw_text can be indexed)."""
    repo = NoteRepository(db_session)
    note = await repo.create_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=102,
        source_type="page",
        source_url="https://example.com/blocked",
        raw_text="https://example.com/blocked",
        visibility="private",
    )
    await repo.record_extraction(note.id, extracted_text=None, lang=None, error="blocked address")
    await db_session.flush()
    refreshed = await repo.get(note.id)
    assert refreshed.extracted_text is None
    assert refreshed.error == "blocked address"
