from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.bot.logic import (
    Deps,
    delete_note,
    edit_note_by_message,
    edit_note_text,
    list_notes,
    list_trash,
    restore_note,
    run_search,
    save_note,
    save_voice_note,
    set_group_capture_mode,
    show_more,
    smart_search,
    toggle_privacy,
)
from notes_bot.clients.search_cache import SearchSessionCache
from notes_bot.config import Settings
from notes_bot.db.models import Note, NoteChunk
from notes_bot.db.repositories import ChatSettingsRepository

pytestmark = pytest.mark.asyncio

MODEL = "fake-model"


class FakeEmbeddingClient:
    model_name = MODEL
    dim = 768

    async def embed_passages(self, texts):
        raise NotImplementedError

    async def embed_query(self, text):
        # Deterministic: same query text always maps to the same vector,
        # letting tests control ranking via note embeddings.
        return [1.0, 0.0] + [0.0] * 766


class FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def enqueue(self, func, *args, job_id=None, **kw):
        self.calls.append((func, args, job_id))


def _vec(x: float, y: float) -> list[float]:
    return [x, y] + [0.0] * 766


def _settings(**overrides) -> Settings:
    base = dict(
        TELEGRAM_BOT_TOKEN="x",
        DATABASE_URL="unused",
        REDIS_URL="unused",
        EMBEDDINGS_URL="unused",
        EMBEDDING_MODEL_NAME=MODEL,
        SEARCH_PAGE_SIZE=2,
        SEARCH_CANDIDATE_K=200,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
async def redis_client():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    client = redis.from_url(url)
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.fixture(autouse=True)
async def _cleanup_notes(db_engine):
    """Unlike db_session (rolled back per test), Deps commits real rows
    directly against the shared engine — this file owns user_id 1 and 2 and
    sweeps them after every test so one test's notes can't leak into the
    next one's search results."""
    yield
    from sqlalchemy import delete

    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with sf() as session:
        await session.execute(delete(Note).where(Note.user_id.in_([1, 2])))
        await session.commit()


@pytest.fixture
def deps(db_engine, redis_client):
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    return Deps(
        session_factory=session_factory,
        embedding_client=FakeEmbeddingClient(),
        search_cache=SearchSessionCache(redis_client),
        fast_queue=FakeQueue(),
        heavy_queue=FakeQueue(),
        llm_queue=FakeQueue(),
        settings=_settings(),
    )


async def _add_note_with_chunk(
    db_engine,
    *,
    user_id=1,
    chat_id=1,
    is_group=False,
    visibility="private",
    x=1.0,
    y=0.0,
    status="done",
    deleted=False,
):
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with sf() as session:
        note = Note(
            user_id=user_id,
            chat_id=chat_id,
            is_group=is_group,
            visibility=visibility,
            source_type="text",
            raw_text="x",
            status=status,
        )
        if deleted:
            note.deleted_at = datetime.now(UTC)
        session.add(note)
        await session.flush()
        session.add(
            NoteChunk(
                note_id=note.id,
                chunk_index=0,
                chunk_text=f"chunk for {note.id}",
                embedding=_vec(x, y),
                embedding_model=MODEL,
            )
        )
        await session.commit()
        return note.id


async def test_save_note_private_chat_uses_default_visibility_and_enqueues(deps):
    result = await save_note(
        deps, user_id=1, chat_id=1, is_group=False, tg_message_id=10, text="hello"
    )
    assert result.created is True
    assert result.visibility == "private"
    assert len(deps.fast_queue.calls) == 1
    assert deps.fast_queue.calls[0][2] == f"process_note:{result.note_id}"


async def test_save_note_duplicate_is_not_created_and_not_enqueued(deps):
    first = await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=11, text="a")
    second = await save_note(
        deps, user_id=1, chat_id=1, is_group=False, tg_message_id=11, text="a again"
    )
    assert first.created is True
    assert second.created is False
    assert len(deps.fast_queue.calls) == 1


async def test_save_note_group_note_has_no_visibility(deps):
    result = await save_note(
        deps, user_id=1, chat_id=999, is_group=True, tg_message_id=12, text="in a group"
    )
    assert result.visibility is None


async def test_toggle_privacy(deps):
    saved = await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=13, text="x")
    new_vis = await toggle_privacy(deps, note_id=saved.note_id, user_id=1)
    assert new_vis == "public"


async def test_list_notes_returns_own_notes_newest_first(deps):
    first = await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=30, text="a")
    second = await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=31, text="b")
    page = await list_notes(deps, user_id=1, offset=0)
    assert [h.note_id for h in page.hits] == [second.note_id, first.note_id]
    assert page.has_more is False


async def test_list_notes_has_more_when_over_page_size(deps):
    for i in range(3):  # SEARCH_PAGE_SIZE=2
        await save_note(
            deps, user_id=1, chat_id=1, is_group=False, tg_message_id=40 + i, text=str(i)
        )
    page = await list_notes(deps, user_id=1, offset=0)
    assert len(page.hits) == 2
    assert page.has_more is True
    assert page.next_offset == 2


async def test_list_notes_excludes_other_users(deps):
    await save_note(deps, user_id=2, chat_id=2, is_group=False, tg_message_id=50, text="not mine")
    page = await list_notes(deps, user_id=1, offset=0)
    assert page.hits == []


async def test_delete_note_then_it_is_gone_from_list_and_in_trash(deps):
    saved = await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=60, text="x")
    assert await delete_note(deps, note_id=saved.note_id, user_id=1) is True

    listed = await list_notes(deps, user_id=1, offset=0)
    assert saved.note_id not in [h.note_id for h in listed.hits]

    trash = await list_trash(deps, user_id=1, offset=0)
    assert [h.note_id for h in trash.hits] == [saved.note_id]


async def test_delete_note_refuses_a_different_user(deps):
    saved = await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=61, text="x")
    assert await delete_note(deps, note_id=saved.note_id, user_id=999) is False


async def test_restore_note_brings_it_back(deps):
    saved = await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=62, text="x")
    await delete_note(deps, note_id=saved.note_id, user_id=1)
    assert await restore_note(deps, note_id=saved.note_id, user_id=1) is True

    listed = await list_notes(deps, user_id=1, offset=0)
    assert [h.note_id for h in listed.hits] == [saved.note_id]


async def test_edit_note_text_re_enqueues_processing(deps):
    saved = await save_note(
        deps, user_id=1, chat_id=1, is_group=False, tg_message_id=63, text="old"
    )
    deps.fast_queue.calls.clear()

    assert await edit_note_text(deps, note_id=saved.note_id, user_id=1, new_text="new") is True
    assert len(deps.fast_queue.calls) == 1
    assert deps.fast_queue.calls[0][2] == f"process_note:{saved.note_id}"


async def test_edit_note_text_refuses_a_different_user(deps):
    saved = await save_note(
        deps, user_id=1, chat_id=1, is_group=False, tg_message_id=64, text="old"
    )
    assert await edit_note_text(deps, note_id=saved.note_id, user_id=999, new_text="new") is False


async def test_edit_note_by_message_re_enqueues_processing(deps):
    await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=65, text="old")
    deps.fast_queue.calls.clear()

    edited = await edit_note_by_message(
        deps, chat_id=1, tg_message_id=65, user_id=1, new_text="new"
    )
    assert edited is True
    assert len(deps.fast_queue.calls) == 1


async def test_edit_note_by_message_refuses_a_different_user(deps):
    await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=66, text="old")
    edited = await edit_note_by_message(
        deps, chat_id=1, tg_message_id=66, user_id=999, new_text="new"
    )
    assert edited is False


async def test_set_group_capture_mode_persists(deps, db_engine):
    await set_group_capture_mode(deps, chat_id=777, mode="all")
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with sf() as session:
        assert await ChatSettingsRepository(session).get_capture_mode(777) == "all"


async def test_smart_search_returns_the_same_page_as_run_search(deps, db_engine):
    """Default `deps` fixture has LLM_ENABLED=false (Settings' own
    default), so this also covers "no enqueue when disabled"."""
    await _add_note_with_chunk(db_engine, x=0.99, y=0.01)

    page = await smart_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    assert len(page.hits) == 1
    assert deps.llm_queue.calls == []


async def test_smart_search_enqueues_smart_answer_when_llm_enabled(db_engine, redis_client):
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    enabled_deps = Deps(
        session_factory=session_factory,
        embedding_client=FakeEmbeddingClient(),
        search_cache=SearchSessionCache(redis_client),
        fast_queue=FakeQueue(),
        heavy_queue=FakeQueue(),
        llm_queue=FakeQueue(),
        settings=_settings(LLM_ENABLED=True),
    )
    await _add_note_with_chunk(db_engine, x=0.99, y=0.01)

    await smart_search(enabled_deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    assert len(enabled_deps.llm_queue.calls) == 1


async def test_save_note_classifies_a_page_url_and_enqueues_it(deps):
    result = await save_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=14,
        text="check this out https://example.com/article",
    )
    assert result.source_type == "page"
    assert len(deps.fast_queue.calls) == 1


async def test_save_note_classifies_a_youtube_url(deps):
    result = await save_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=15,
        text="https://youtu.be/abc123",
    )
    assert result.source_type == "youtube"


async def test_save_note_stores_the_source_url_for_a_link(deps, db_engine):
    result = await save_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=16,
        text="https://example.com/article",
    )
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with sf() as session:
        note = await session.get(Note, result.note_id)
        assert note.source_url == "https://example.com/article"


async def test_save_note_routes_an_instagram_link_to_the_heavy_queue(deps):
    """03-ingest.md, "Шаг 2": instagram и voice → heavy, всё остальное →
    fast."""
    result = await save_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=17,
        text="https://www.instagram.com/p/xyz/",
    )
    assert result.source_type == "instagram"
    assert result.created is True
    assert deps.fast_queue.calls == []
    assert len(deps.heavy_queue.calls) == 1


async def test_save_voice_note_uses_file_id_as_source_url_and_caption_as_raw_text(deps, db_engine):
    result = await save_voice_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=20,
        file_id="AwACAgIAAx",
        caption="a caption",
    )
    assert result.created is True
    assert result.source_type == "voice"

    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with sf() as session:
        note = await session.get(Note, result.note_id)
        assert note.source_url == "AwACAgIAAx"
        assert note.raw_text == "a caption"


async def test_save_voice_note_routes_to_the_heavy_queue(deps):
    result = await save_voice_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=21,
        file_id="AwACAgIAAx",
        caption=None,
    )
    assert result.created is True
    assert deps.fast_queue.calls == []
    assert len(deps.heavy_queue.calls) == 1


async def test_save_voice_note_group_note_has_no_visibility(deps):
    result = await save_voice_note(
        deps,
        user_id=1,
        chat_id=999,
        is_group=True,
        tg_message_id=22,
        file_id="AwACAgIAAx",
        caption=None,
    )
    assert result.visibility is None


async def test_save_voice_note_duplicate_is_not_created(deps):
    first = await save_voice_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=23,
        file_id="AwACAgIAAx",
        caption=None,
    )
    second = await save_voice_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=23,
        file_id="AwACAgIAAx",
        caption=None,
    )
    assert first.created is True
    assert second.created is False
    assert len(deps.heavy_queue.calls) == 1


async def test_run_search_caches_a_session_and_returns_first_page(deps, db_engine):
    await _add_note_with_chunk(db_engine, x=0.99, y=0.01)  # close
    await _add_note_with_chunk(db_engine, x=0.9, y=0.1)  # a bit less close
    await _add_note_with_chunk(db_engine, x=0.0, y=1.0)  # far — third page

    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    assert len(page.hits) == 2  # SEARCH_PAGE_SIZE=2
    assert page.has_more is True
    assert page.session_id is not None


async def test_run_search_with_no_matching_notes_returns_empty(deps):
    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    assert page.hits == []
    assert page.has_more is False
    assert page.session_id is None


async def test_run_search_respects_acl(deps, db_engine):
    await _add_note_with_chunk(db_engine, user_id=1, chat_id=1, visibility="private", x=1.0, y=0.0)
    await _add_note_with_chunk(db_engine, user_id=2, chat_id=2, visibility="private", x=1.0, y=0.0)

    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    assert len(page.hits) == 1


async def test_show_more_returns_the_next_page(deps, db_engine):
    await _add_note_with_chunk(db_engine, x=0.99, y=0.01)
    await _add_note_with_chunk(db_engine, x=0.9, y=0.1)
    third_id = await _add_note_with_chunk(db_engine, x=0.0, y=1.0)

    first_page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    more = await show_more(deps, user_id=1, session_id=first_page.session_id)

    assert len(more.hits) == 1
    assert more.hits[0].note_id == third_id
    assert more.has_more is False
    assert more.expired is False


async def test_show_more_on_unknown_session_reports_expired(deps):
    result = await show_more(deps, user_id=1, session_id="does-not-exist")
    assert result.expired is True
    assert result.hits == []


async def test_show_more_excludes_a_note_deleted_between_pages(deps, db_engine):
    await _add_note_with_chunk(db_engine, x=0.99, y=0.01)
    await _add_note_with_chunk(db_engine, x=0.9, y=0.1)
    deleted_id = await _add_note_with_chunk(db_engine, x=0.0, y=1.0)

    first_page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")

    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with sf() as session:
        note = await session.get(Note, deleted_id)
        note.deleted_at = datetime.now(UTC)
        await session.commit()

    more = await show_more(deps, user_id=1, session_id=first_page.session_id)
    assert more.hits == []
