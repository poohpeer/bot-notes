from __future__ import annotations

import pytest

from notes_bot.db.repositories import (
    ChatSettingsRepository,
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


async def test_mark_processing_bumps_updated_at(db_session):
    """The heartbeat notes-gc's reclaim_stuck reads: without this, a note
    that merely waited a while for a free worker (nothing wrong, the queue
    was just busy) would look exactly as stale as one whose worker died the
    instant it started — updated_at would still equal the note's original
    created_at either way.

    Read back through a plain Core SELECT rather than repo.get(): the
    ORM-mapped `note` object stays in the session's identity map, and
    mark_processing's `updated_at=func.now()` (a SQL expression, not
    something SQLAlchemy can evaluate in Python) marks that attribute
    expired rather than updating it in place — a bare attribute access on
    the same object would then need its own implicit reload, which the
    asyncio extension does not allow outside an explicit `await`.

    Backdated with a literal Python datetime, not another `func.now()`: the
    whole test runs inside one transaction (this fixture's isolation
    mechanism), and Postgres's `now()` is transaction-scoped — a second
    `func.now()` in the same transaction would equal the first exactly,
    proving nothing.
    """
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select, update

    from notes_bot.db.models import Note

    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=99, raw_text="x", visibility="private"
    )
    stale = datetime.now(UTC) - timedelta(hours=1)
    await db_session.execute(update(Note).where(Note.id == note.id).values(updated_at=stale))
    await db_session.flush()

    await repo.mark_processing(note.id)
    await db_session.flush()

    result = await db_session.execute(select(Note.updated_at).where(Note.id == note.id))
    assert result.scalar_one() > stale


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
    assert settings.debug_enabled is False


async def test_toggle_debug_flips_and_returns_the_new_value(db_session):
    repo = UserSettingsRepository(db_session)
    assert await repo.is_debug_enabled(42) is False

    first = await repo.toggle_debug(42)
    assert first is True
    assert await repo.is_debug_enabled(42) is True

    second = await repo.toggle_debug(42)
    assert second is False
    assert await repo.is_debug_enabled(42) is False


async def test_is_debug_enabled_for_a_user_with_no_row_is_false(db_session):
    repo = UserSettingsRepository(db_session)
    assert await repo.is_debug_enabled(999) is False


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


async def test_set_enrichment_writes_title_tags_summary_structured(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=200, raw_text="x", visibility="private"
    )
    await repo.set_enrichment(
        note.id,
        title="A Title",
        summary="A summary",
        tags=["food", "tbilisi"],
        structured={"name": "Хинкальная"},
    )
    await db_session.flush()
    refreshed = await repo.get(note.id)
    assert refreshed.title == "A Title"
    assert refreshed.summary == "A summary"
    assert refreshed.tags == ["food", "tbilisi"]
    assert refreshed.structured == {"name": "Хинкальная"}
    assert refreshed.enrich_status == "done"


async def test_set_enrichment_does_not_touch_status_or_extracted_text(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=201, raw_text="x", visibility="private"
    )
    await repo.mark_done(note.id)
    await repo.set_enrichment(note.id, title="t", summary=None, tags=[], structured={})
    await db_session.flush()
    refreshed = await repo.get(note.id)
    assert refreshed.status == "done"


async def test_mark_enrich_failed_records_error(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=202, raw_text="x", visibility="private"
    )
    await repo.mark_enrich_failed(note.id, "quota exhausted")
    await db_session.flush()
    refreshed = await repo.get(note.id)
    assert refreshed.enrich_status == "failed"
    assert refreshed.error == "quota exhausted"


async def test_mark_enrich_skipped(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=203, raw_text="x", visibility="private"
    )
    await repo.mark_enrich_skipped(note.id)
    await db_session.flush()
    refreshed = await repo.get(note.id)
    assert refreshed.enrich_status == "skipped"


async def test_get_first_chunk_embedding_returns_the_lowest_index_chunk(db_session):
    repo = NoteRepository(db_session)
    chunk_repo = ChunkRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=204, raw_text="x", visibility="private"
    )
    await chunk_repo.replace_chunks(
        note.id,
        [
            NewChunk(text="first", token_count=1, embedding=_vec(0.1)),
            NewChunk(text="second", token_count=1, embedding=_vec(0.2)),
        ],
        embedding_model="test-model",
    )
    await db_session.flush()
    embedding = await chunk_repo.get_first_chunk_embedding(note.id)
    assert embedding == pytest.approx(_vec(0.1))


async def test_get_first_chunk_embedding_returns_none_without_chunks(db_session):
    repo = NoteRepository(db_session)
    chunk_repo = ChunkRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=205, raw_text="x", visibility="private"
    )
    assert await chunk_repo.get_first_chunk_embedding(note.id) is None


async def test_set_capture_mode_then_get_returns_it(db_session):
    repo = ChatSettingsRepository(db_session)
    await repo.set_capture_mode(555, "all")
    assert await repo.get_capture_mode(555) == "all"


async def test_get_capture_mode_defaults_to_mentions_and_replies(db_session):
    repo = ChatSettingsRepository(db_session)
    assert await repo.get_capture_mode(999) == "mentions_and_replies"


async def test_set_capture_mode_is_idempotent(db_session):
    repo = ChatSettingsRepository(db_session)
    await repo.set_capture_mode(555, "all")
    await repo.set_capture_mode(555, "mentions_and_replies")
    assert await repo.get_capture_mode(555) == "mentions_and_replies"


async def test_list_own_returns_notes_newest_first(db_session):
    repo = NoteRepository(db_session)
    first = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=300, raw_text="a", visibility="private"
    )
    second = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=301, raw_text="b", visibility="private"
    )
    notes = await repo.list_own(1, limit=10, offset=0)
    assert [n.id for n in notes] == [second.id, first.id]


async def test_list_own_excludes_deleted_and_other_users(db_session):
    repo = NoteRepository(db_session)
    mine = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=302, raw_text="a", visibility="private"
    )
    deleted = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=303, raw_text="b", visibility="private"
    )
    await repo.soft_delete(deleted.id, 1)
    await repo.create_text_note(
        user_id=2, chat_id=2, is_group=False, tg_message_id=304, raw_text="c", visibility="private"
    )
    notes = await repo.list_own(1, limit=10, offset=0)
    assert [n.id for n in notes] == [mine.id]


async def test_list_own_respects_limit_and_offset(db_session):
    repo = NoteRepository(db_session)
    for i in range(5):
        await repo.create_text_note(
            user_id=1,
            chat_id=1,
            is_group=False,
            tg_message_id=400 + i,
            raw_text=str(i),
            visibility="private",
        )
    page1 = await repo.list_own(1, limit=2, offset=0)
    page2 = await repo.list_own(1, limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert {n.id for n in page1}.isdisjoint({n.id for n in page2})


async def test_list_group_returns_notes_from_any_member_in_that_chat(db_session):
    """ADR-10: group notes have no `visibility` — the whole room, not just
    the caller, so /list there must show notes saved by *any* member."""
    repo = NoteRepository(db_session)
    first = await repo.create_text_note(
        user_id=1, chat_id=-100, is_group=True, tg_message_id=600, raw_text="a", visibility=None
    )
    second = await repo.create_text_note(
        user_id=2, chat_id=-100, is_group=True, tg_message_id=601, raw_text="b", visibility=None
    )
    notes = await repo.list_group(-100, limit=10, offset=0)
    assert [n.id for n in notes] == [second.id, first.id]


async def test_list_group_excludes_other_chats_and_deleted(db_session):
    repo = NoteRepository(db_session)
    mine = await repo.create_text_note(
        user_id=1, chat_id=-100, is_group=True, tg_message_id=602, raw_text="a", visibility=None
    )
    deleted = await repo.create_text_note(
        user_id=1, chat_id=-100, is_group=True, tg_message_id=603, raw_text="b", visibility=None
    )
    await repo.soft_delete(deleted.id, 1)
    await repo.create_text_note(
        user_id=1, chat_id=-200, is_group=True, tg_message_id=604, raw_text="c", visibility=None
    )
    notes = await repo.list_group(-100, limit=10, offset=0)
    assert [n.id for n in notes] == [mine.id]


async def test_list_group_does_not_include_a_private_note_in_the_same_chat_id(db_session):
    """`is_group` gates this, not just `chat_id` — a private note happens
    to share its chat_id with the user's own DM, never a room's listing."""
    repo = NoteRepository(db_session)
    await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=605, raw_text="dm", visibility="private"
    )
    notes = await repo.list_group(1, limit=10, offset=0)
    assert notes == []


async def test_soft_delete_then_list_own_deleted(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=500, raw_text="a", visibility="private"
    )
    assert await repo.soft_delete(note.id, 1) is True
    deleted = await repo.list_own_deleted(1, limit=10, offset=0)
    assert [n.id for n in deleted] == [note.id]


async def test_soft_delete_refuses_a_different_user(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=501, raw_text="a", visibility="private"
    )
    assert await repo.soft_delete(note.id, 999) is False
    assert (await repo.get(note.id)).deleted_at is None


async def test_soft_delete_twice_returns_false_the_second_time(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=502, raw_text="a", visibility="private"
    )
    assert await repo.soft_delete(note.id, 1) is True
    assert await repo.soft_delete(note.id, 1) is False


async def test_restore_brings_a_note_back(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=503, raw_text="a", visibility="private"
    )
    await repo.soft_delete(note.id, 1)
    assert await repo.restore(note.id, 1) is True
    # Two prior bulk UPDATEs push the identity-mapped `note` into
    # SQLAlchemy's "expired" state; a plain attribute read would then try
    # an implicit lazy-load outside of an awaited context. An explicit
    # refresh avoids that — see SQLAlchemy's async session caveats.
    await db_session.refresh(note)
    assert note.deleted_at is None


async def test_restore_refuses_a_different_user(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=504, raw_text="a", visibility="private"
    )
    await repo.soft_delete(note.id, 1)
    assert await repo.restore(note.id, 999) is False


async def test_restore_on_a_non_deleted_note_returns_false(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1, chat_id=1, is_group=False, tg_message_id=505, raw_text="a", visibility="private"
    )
    assert await repo.restore(note.id, 1) is False


async def test_edit_text_updates_raw_text_and_resets_status(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=506,
        raw_text="old",
        visibility="private",
    )
    await repo.mark_done(note.id)
    assert await repo.edit_text(note.id, 1, "new text") is True
    refreshed = await repo.get(note.id)
    assert refreshed.raw_text == "new text"
    assert refreshed.status == "pending"


async def test_edit_text_refuses_a_different_user(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=507,
        raw_text="old",
        visibility="private",
    )
    assert await repo.edit_text(note.id, 999, "new text") is False
    assert (await repo.get(note.id)).raw_text == "old"


async def test_edit_text_refuses_a_non_text_note(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=508,
        source_type="page",
        source_url="https://example.com",
        raw_text="https://example.com",
        visibility="private",
    )
    assert await repo.edit_text(note.id, 1, "new text") is False


async def test_edit_text_by_message_finds_the_note_and_returns_its_id(db_session):
    repo = NoteRepository(db_session)
    note = await repo.create_text_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=600,
        raw_text="old",
        visibility="private",
    )
    await repo.mark_done(note.id)
    note_id = await repo.edit_text_by_message(
        chat_id=1, tg_message_id=600, user_id=1, new_text="edited"
    )
    assert note_id == note.id
    refreshed = await repo.get(note.id)
    assert refreshed.raw_text == "edited"
    assert refreshed.status == "pending"


async def test_edit_text_by_message_refuses_a_different_user(db_session):
    repo = NoteRepository(db_session)
    await repo.create_text_note(
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=601,
        raw_text="old",
        visibility="private",
    )
    note_id = await repo.edit_text_by_message(
        chat_id=1, tg_message_id=601, user_id=999, new_text="edited"
    )
    assert note_id is None


async def test_edit_text_by_message_returns_none_for_an_unknown_message(db_session):
    repo = NoteRepository(db_session)
    note_id = await repo.edit_text_by_message(
        chat_id=1, tg_message_id=99999, user_id=1, new_text="edited"
    )
    assert note_id is None
