from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.bot.logic import (
    Deps,
    _filter_table_event_hits_by_subject,
    cancel_events_table,
    confirm_events_table,
    delete_note,
    edit_note_by_message,
    edit_note_text,
    list_group_notes,
    list_notes,
    list_trash,
    resolve_events_table_conflict,
    restore_note,
    run_search,
    save_note,
    save_voice_note,
    set_group_capture_mode,
    show_detail,
    show_more,
    smart_search,
    toggle_privacy,
)
from notes_bot.clients.conflicts_cache import ConflictsCache, PendingConflict
from notes_bot.clients.pending_events_cache import PendingEventsCache
from notes_bot.clients.search_cache import SearchSessionCache
from notes_bot.clients.translate import TranslateServiceError
from notes_bot.config import Settings
from notes_bot.db.models import Note, NoteChunk
from notes_bot.db.repositories import ChatSettingsRepository, UserSettingsRepository
from notes_bot.db.search import SearchHit
from notes_bot.events_table import ExtractedEvent

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


class RecordingEmbeddingClient(FakeEmbeddingClient):
    """Same fixed vector as FakeEmbeddingClient (ranking isn't the point
    here), but records what text embed_query actually received — that's
    what the translate-before-embed tests below assert on."""

    def __init__(self) -> None:
        self.query_calls: list[str] = []

    async def embed_query(self, text):
        self.query_calls.append(text)
        return await super().embed_query(text)


class FakeTranslateClient:
    def __init__(self, fail_with: Exception | None = None) -> None:
        self._fail_with = fail_with
        self.query_calls: list[str] = []

    async def translate_passages(self, texts):
        if self._fail_with:
            raise self._fail_with
        return [f"[en] {t}" for t in texts]

    async def translate_query(self, text: str) -> str:
        self.query_calls.append(text)
        if self._fail_with:
            raise self._fail_with
        return f"[en] {text}"


class FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.kwargs: list[dict] = []

    def enqueue(self, func, *args, job_id=None, **kw):
        self.calls.append((func, args, job_id))
        self.kwargs.append(kw)


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
        # These tests are about pagination/ACL/caching mechanics, not
        # relevance — several deliberately include a "far" note (orthogonal
        # embedding) to exercise a later page, not to be filtered out.
        # Relevance filtering itself is tests/test_search.py's job.
        SEARCH_MAX_DISTANCE=None,
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
    assert deps.fast_queue.calls[0][2] == f"process_note-{result.note_id}"


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


async def test_list_group_notes_shows_notes_from_any_member_of_the_room(deps):
    mine = await save_note(deps, user_id=1, chat_id=-100, is_group=True, tg_message_id=70, text="a")
    theirs = await save_note(
        deps, user_id=2, chat_id=-100, is_group=True, tg_message_id=71, text="b"
    )
    page = await list_group_notes(deps, chat_id=-100, viewer_user_id=1, offset=0)
    assert {h.note_id for h in page.hits} == {mine.note_id, theirs.note_id}


async def test_list_group_notes_marks_ownership_per_note(deps):
    mine = await save_note(deps, user_id=1, chat_id=-100, is_group=True, tg_message_id=72, text="a")
    theirs = await save_note(
        deps, user_id=2, chat_id=-100, is_group=True, tg_message_id=73, text="b"
    )
    page = await list_group_notes(deps, chat_id=-100, viewer_user_id=1, offset=0)
    by_id = {h.note_id: h for h in page.hits}
    assert by_id[mine.note_id].is_owner is True
    assert by_id[theirs.note_id].is_owner is False


async def test_list_group_notes_excludes_the_callers_own_private_dm_notes(deps):
    """The whole reason list_group_notes exists instead of list_notes with
    an extra filter: a room's /list must never leak a member's own private
    DM notes, which is exactly what running list_notes(user_id=...) in a
    group would do."""
    await save_note(deps, user_id=1, chat_id=1, is_group=False, tg_message_id=74, text="private dm")
    page = await list_group_notes(deps, chat_id=-100, viewer_user_id=1, offset=0)
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
    assert deps.fast_queue.calls[0][2] == f"process_note-{saved.note_id}"


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
    # Settings.heavy_job_timeout_s, not RQ's own 180s default — see its
    # docstring for why that default is too tight for this queue.
    assert deps.heavy_queue.kwargs[0]["job_timeout"] == deps.settings.heavy_job_timeout_s


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
    assert deps.heavy_queue.kwargs[0]["job_timeout"] == deps.settings.heavy_job_timeout_s


async def test_save_note_to_the_fast_queue_uses_no_explicit_timeout(deps):
    """RQ's own default (180s) is plenty for text/pages — only the heavy
    queue needs a longer ceiling."""
    result = await save_note(
        deps,
        user_id=1,
        chat_id=1,
        is_group=False,
        tg_message_id=18,
        text="just some text",
    )
    assert result.source_type == "text"
    assert deps.fast_queue.kwargs[0]["job_timeout"] is None


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


def _table_event_hit(note_id: int, subjects: list[str]) -> SearchHit:
    return SearchHit(
        note_id=note_id,
        title=None,
        source_url=None,
        source_type="table_event",
        tags=[],
        chunk_text="x",
        distance=0.1,
        structured={"subjects": subjects},
        summary=None,
    )


def _text_hit(note_id: int) -> SearchHit:
    return SearchHit(
        note_id=note_id,
        title=None,
        source_url=None,
        source_type="text",
        tags=[],
        chunk_text="x",
        distance=0.1,
        structured={},
        summary=None,
    )


async def test_filter_table_event_hits_by_subject_keeps_only_the_matching_subject():
    """The 30/11 bug from prod: a query naming a subject some table_event
    hits don't have must drop those, not just accept the generic "экзамен"
    vector match for all of them."""
    hits = [
        _table_event_hit(1, ["математика"]),
        _table_event_hit(2, ["физика", "искусство"]),
    ]
    result = _filter_table_event_hits_by_subject(hits, "когда экзамен по математике?")
    assert [h.note_id for h in result] == [1]


async def test_filter_table_event_hits_by_subject_keeps_everything_when_no_subject_matches():
    """Nothing to discriminate on - a broad answer beats an empty one."""
    hits = [
        _table_event_hit(1, ["физика"]),
        _table_event_hit(2, ["биология"]),
    ]
    result = _filter_table_event_hits_by_subject(hits, "когда экзамен по химии?")
    assert [h.note_id for h in result] == [1, 2]


async def test_filter_table_event_hits_by_subject_leaves_non_table_event_hits_alone():
    hits = [
        _table_event_hit(1, ["математика"]),
        _table_event_hit(2, ["физика"]),
        _text_hit(3),
    ]
    result = _filter_table_event_hits_by_subject(hits, "математика")
    assert [h.note_id for h in result] == [1, 3]


async def test_filter_table_event_hits_by_subject_is_a_no_op_with_fewer_than_two_table_event_hits():
    hits = [_table_event_hit(1, ["физика"]), _text_hit(2)]
    result = _filter_table_event_hits_by_subject(hits, "математика")
    assert [h.note_id for h in result] == [1, 2]


async def test_filter_table_event_hits_by_subject_tolerates_different_word_endings():
    """Both sides are already in the same canonical translate language
    (04-search.md) - but the translate service's own wording can still
    differ between a full word and its translated query form ("mathematics"
    vs "math"), so the match is a shared-prefix check, not an exact one."""
    hits = [
        _table_event_hit(1, ["mathematics"]),
        _table_event_hit(2, ["literature"]),
    ]
    result = _filter_table_event_hits_by_subject(hits, "when is the math exam?")
    assert [h.note_id for h in result] == [1]


async def test_run_search_with_no_matching_notes_returns_empty(deps):
    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    assert page.hits == []
    assert page.has_more is False
    assert page.session_id is None
    assert page.debug_info is None


async def test_run_search_embeds_the_translated_query_when_translate_client_set(
    db_engine, redis_client
):
    """04-search.md, "Перевод перед эмбеддингом": the vector must come from
    the translated query text, not the original."""
    await _add_note_with_chunk(db_engine, x=0.99, y=0.01)
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    embedding_client = RecordingEmbeddingClient()
    translate_client = FakeTranslateClient()
    deps = Deps(
        session_factory=session_factory,
        embedding_client=embedding_client,
        search_cache=SearchSessionCache(redis_client),
        fast_queue=FakeQueue(),
        heavy_queue=FakeQueue(),
        llm_queue=FakeQueue(),
        settings=_settings(),
        translate_client=translate_client,
    )

    await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="где велодорожки")

    assert translate_client.query_calls == ["где велодорожки"]
    assert embedding_client.query_calls == ["[en] где велодорожки"]


async def test_run_search_falls_back_to_the_original_query_when_translate_fails(
    db_engine, redis_client
):
    """Best-effort: a down translate service must not fail the search, only
    degrade to the old same-language-only behavior."""
    await _add_note_with_chunk(db_engine, x=0.99, y=0.01)
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    embedding_client = RecordingEmbeddingClient()
    translate_client = FakeTranslateClient(fail_with=TranslateServiceError(503, "down"))
    deps = Deps(
        session_factory=session_factory,
        embedding_client=embedding_client,
        search_cache=SearchSessionCache(redis_client),
        fast_queue=FakeQueue(),
        heavy_queue=FakeQueue(),
        llm_queue=FakeQueue(),
        settings=_settings(),
        translate_client=translate_client,
    )

    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")

    assert embedding_client.query_calls == ["q"]
    assert len(page.hits) == 1


async def test_run_search_debug_info_when_relevance_filtered_and_debug_on(db_engine, redis_client):
    """/debug (03-ingest.md): a note that exists but doesn't clear
    SEARCH_MAX_DISTANCE should surface its actual distance, not just
    "nothing found" — that's the whole point of tuning the threshold."""
    await _add_note_with_chunk(db_engine, x=0.0, y=1.0)  # orthogonal -> distance 1.0
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    deps = Deps(
        session_factory=session_factory,
        embedding_client=FakeEmbeddingClient(),
        search_cache=SearchSessionCache(redis_client),
        fast_queue=FakeQueue(),
        heavy_queue=FakeQueue(),
        llm_queue=FakeQueue(),
        settings=_settings(SEARCH_MAX_DISTANCE=0.1),
    )
    async with session_factory() as session:
        await UserSettingsRepository(session).toggle_debug(1)
        await session.commit()

    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")

    assert page.hits == []
    assert page.debug_info is not None
    assert "distance=1.000" in page.debug_info
    assert "0.1" in page.debug_info  # the threshold itself

    async with session_factory() as session:
        await UserSettingsRepository(session).toggle_debug(1)
        await session.commit()


async def test_run_search_no_debug_info_when_debug_off(db_engine, redis_client):
    await _add_note_with_chunk(db_engine, x=0.0, y=1.0)
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    deps = Deps(
        session_factory=session_factory,
        embedding_client=FakeEmbeddingClient(),
        search_cache=SearchSessionCache(redis_client),
        fast_queue=FakeQueue(),
        heavy_queue=FakeQueue(),
        llm_queue=FakeQueue(),
        settings=_settings(SEARCH_MAX_DISTANCE=0.1),
    )

    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")

    assert page.hits == []
    assert page.debug_info is None


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


async def test_show_detail_returns_the_full_hit_for_a_note_in_the_session(deps, db_engine):
    note_id = await _add_note_with_chunk(db_engine, x=0.99, y=0.01)

    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    hit = await show_detail(deps, user_id=1, session_id=page.session_id, note_id=note_id)

    assert hit is not None
    assert hit.note_id == note_id


async def test_show_detail_on_unknown_session_returns_none(deps, db_engine):
    note_id = await _add_note_with_chunk(db_engine, x=0.99, y=0.01)
    hit = await show_detail(deps, user_id=1, session_id="does-not-exist", note_id=note_id)
    assert hit is None


async def test_show_detail_rejects_a_note_id_not_in_the_session(deps, db_engine):
    """A tampered or stale callback_data must not let a note outside the
    original search's own result set be looked up through this session —
    private to another user, so ACL excludes it from user 1's results."""
    await _add_note_with_chunk(db_engine, x=0.99, y=0.01)
    other_id = await _add_note_with_chunk(
        db_engine, user_id=2, chat_id=2, visibility="private", x=0.0, y=1.0
    )

    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")
    hit = await show_detail(deps, user_id=1, session_id=page.session_id, note_id=other_id)

    assert hit is None
    assert other_id not in [h.note_id for h in page.hits]  # sanity: really excluded


async def test_show_detail_rechecks_acl_and_hides_a_note_made_private_since(deps, db_engine):
    """Re-checked at click time, not trusted from the original search
    snapshot — same reasoning as show_more's own deleted-between-pages test."""
    note_id = await _add_note_with_chunk(
        db_engine, user_id=2, chat_id=2, visibility="public", x=0.99, y=0.01
    )

    page = await run_search(deps, user_id=1, chat_id=1, is_group_chat=False, query_text="q")

    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with sf() as session:
        note = await session.get(Note, note_id)
        note.visibility = "private"
        await session.commit()

    hit = await show_detail(deps, user_id=1, session_id=page.session_id, note_id=note_id)
    assert hit is None


def _deps_with_events_cache(db_engine, redis_client, *, translate_client=None) -> Deps:
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    return Deps(
        session_factory=session_factory,
        embedding_client=FakeEmbeddingClient(),
        search_cache=SearchSessionCache(redis_client),
        fast_queue=FakeQueue(),
        heavy_queue=FakeQueue(),
        llm_queue=FakeQueue(),
        settings=_settings(),
        translate_client=translate_client,
        pending_events_cache=PendingEventsCache(redis_client),
        conflicts_cache=ConflictsCache(redis_client),
    )


async def test_confirm_events_table_creates_one_note_per_event(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    events = [
        ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="a"),
        ExtractedEvent(date_start="11/09/2026", date_end="13/09/2026", type="holiday", text="b"),
    ]
    session_id = await deps.pending_events_cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=["школа"], events=events
    )

    result = await confirm_events_table(deps, session_id=session_id, user_id=1)

    assert result.ok is True
    assert result.created == 2
    assert result.skipped_exact == 0
    assert result.conflicts == []
    assert len(deps.fast_queue.calls) == 2
    assert await deps.pending_events_cache.get(session_id) is None

    async with deps.session_factory() as session:
        notes = (
            (
                await session.execute(
                    select(Note).where(Note.user_id == 1, Note.source_type == "table_event")
                )
            )
            .scalars()
            .all()
        )
        assert len(notes) == 2
        by_text = {n.raw_text: n for n in notes}
        assert set(by_text["01/09/2026: a"].tags) == {"экзамен", "школа"}
        assert set(by_text["11/09/2026–13/09/2026: b"].tags) == {"праздник", "школа"}
        assert all(n.enrich_status == "skipped" for n in notes)
        assert all(n.visibility == "private" for n in notes)  # private chat default


async def test_confirm_events_table_puts_subjects_in_tags_and_structured(db_engine, redis_client):
    """04-search.md's subject-based filter reads structured["subjects"] -
    it has to actually be there for a note created by a real confirm."""
    deps = _deps_with_events_cache(db_engine, redis_client)
    events = [
        ExtractedEvent(
            date_start="30/11/2026",
            date_end="30/11/2026",
            type="exam",
            text="a",
            subjects=["физика", "искусство"],
        ),
    ]
    session_id = await deps.pending_events_cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=[], events=events
    )

    await confirm_events_table(deps, session_id=session_id, user_id=1)

    async with deps.session_factory() as session:
        note = (
            (
                await session.execute(
                    select(Note).where(Note.user_id == 1, Note.source_type == "table_event")
                )
            )
            .scalars()
            .one()
        )
        assert set(note.tags) == {"экзамен", "физика", "искусство"}
        assert note.structured["subjects"] == ["физика", "искусство"]


async def test_confirm_events_table_keeps_display_tags_but_canonicalizes_structured_subjects(
    db_engine, redis_client
):
    """/events_table's own subjects stay in the table's language for the
    hashtags shown to the user - structured["subjects"] gets the
    translate_client's canonical-language copy instead, so
    _filter_table_event_hits_by_subject can match a query typed in a
    different language (04-search.md)."""
    deps = _deps_with_events_cache(db_engine, redis_client, translate_client=FakeTranslateClient())
    events = [
        ExtractedEvent(
            date_start="30/11/2026",
            date_end="30/11/2026",
            type="exam",
            text="a",
            subjects=["מתמטיקה"],
        ),
    ]
    session_id = await deps.pending_events_cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=[], events=events
    )

    await confirm_events_table(deps, session_id=session_id, user_id=1)

    async with deps.session_factory() as session:
        note = (
            (
                await session.execute(
                    select(Note).where(Note.user_id == 1, Note.source_type == "table_event")
                )
            )
            .scalars()
            .one()
        )
        assert "מתמטיקה" in note.tags  # display tag: original (table's) language
        assert note.structured["subjects"] == ["[en] מתמטיקה"]  # canonical, for matching


async def test_confirm_events_table_skips_an_exact_duplicate_silently(db_engine, redis_client):
    """03-ingest.md, "/events_table" - "уже существует": same date range,
    type, and text as an already-saved event — a true duplicate, no
    conflict to resolve."""
    deps = _deps_with_events_cache(db_engine, redis_client)
    event = ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="a")

    first_session = await deps.pending_events_cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=[], events=[event]
    )
    first = await confirm_events_table(deps, session_id=first_session, user_id=1)
    assert first.created == 1

    second_session = await deps.pending_events_cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=[], events=[event]
    )
    second = await confirm_events_table(deps, session_id=second_session, user_id=1)

    assert second.created == 0
    assert second.skipped_exact == 1
    assert second.conflicts == []
    async with deps.session_factory() as session:
        notes = (
            (
                await session.execute(
                    select(Note).where(Note.user_id == 1, Note.source_type == "table_event")
                )
            )
            .scalars()
            .all()
        )
        assert len(notes) == 1  # the resend created nothing new


async def test_confirm_events_table_never_dedups_within_the_same_run(db_engine, redis_client):
    """Two genuinely different events sharing a (date_start, date_end,
    type) key *within one run* (e.g. two unrelated entries on the same
    day, both classified "event") must both be added — dedup only ever
    applies against notes from a *previous* run, never siblings from the
    batch currently being confirmed."""
    deps = _deps_with_events_cache(db_engine, redis_client)
    session_id = await deps.pending_events_cache.create(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="t",
        topic_tags=[],
        events=[
            ExtractedEvent(date_start="30/11/2026", date_end="30/11/2026", type="event", text="a"),
            ExtractedEvent(date_start="30/11/2026", date_end="30/11/2026", type="event", text="b"),
        ],
    )

    result = await confirm_events_table(deps, session_id=session_id, user_id=1)

    assert result.created == 2
    assert result.skipped_exact == 0
    assert result.conflicts == []


async def test_confirm_events_table_flags_a_same_key_different_text_as_a_conflict(
    db_engine, redis_client
):
    """Same date range + type, different text — the source table may have
    genuinely changed ("הלוח עשוי להשתנות"), so this must be deferred to
    resolve_events_table_conflict, not guessed at."""
    deps = _deps_with_events_cache(db_engine, redis_client)
    first_session = await deps.pending_events_cache.create(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="t",
        topic_tags=[],
        events=[
            ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="a")
        ],
    )
    await confirm_events_table(deps, session_id=first_session, user_id=1)

    second_session = await deps.pending_events_cache.create(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="t",
        topic_tags=[],
        events=[
            ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="b")
        ],
    )
    result = await confirm_events_table(deps, session_id=second_session, user_id=1)

    assert result.created == 0
    assert result.skipped_exact == 0
    assert len(result.conflicts) == 1
    assert result.conflicts[0].old_text == "01/09/2026: a"
    assert result.conflicts[0].new_text == "01/09/2026: b"
    assert result.conflicts_session_id is not None


async def test_resolve_events_table_conflict_keep_old_is_a_no_op(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    first_session = await deps.pending_events_cache.create(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="t",
        topic_tags=[],
        events=[
            ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="a")
        ],
    )
    await confirm_events_table(deps, session_id=first_session, user_id=1)
    # Simulate the real pipeline having already finished processing this
    # note (process_note would've moved it past 'pending' long before a
    # second /events_table run could arrive) — so "untouched" below is
    # actually observable, not just coincidentally still 'pending'.
    async with deps.session_factory() as session:
        note_id = (
            await session.execute(
                select(Note.id).where(Note.user_id == 1, Note.source_type == "table_event")
            )
        ).scalar_one()
        note = await session.get(Note, note_id)
        note.status = "done"
        await session.commit()

    second_session = await deps.pending_events_cache.create(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="t",
        topic_tags=[],
        events=[
            ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="b")
        ],
    )
    result = await confirm_events_table(deps, session_id=second_session, user_id=1)

    resolved = await resolve_events_table_conflict(
        deps, session_id=result.conflicts_session_id, index=0, keep_new=False, user_id=1
    )
    assert resolved is True

    async with deps.session_factory() as session:
        note = (
            await session.execute(
                select(Note).where(Note.user_id == 1, Note.source_type == "table_event")
            )
        ).scalar_one()
        assert note.raw_text == "01/09/2026: a"  # untouched
        assert note.status == "done"  # never re-processed


async def test_resolve_events_table_conflict_keep_new_replaces_the_note(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    first_session = await deps.pending_events_cache.create(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="t",
        topic_tags=["школа"],
        events=[
            ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="a")
        ],
    )
    first = await confirm_events_table(deps, session_id=first_session, user_id=1)
    assert first.created == 1

    second_session = await deps.pending_events_cache.create(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="t",
        topic_tags=["школа"],
        events=[
            ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="b")
        ],
    )
    result = await confirm_events_table(deps, session_id=second_session, user_id=1)
    assert result.conflicts_session_id is not None

    resolved = await resolve_events_table_conflict(
        deps, session_id=result.conflicts_session_id, index=0, keep_new=True, user_id=1
    )
    assert resolved is True

    async with deps.session_factory() as session:
        note = (
            await session.execute(
                select(Note).where(Note.user_id == 1, Note.source_type == "table_event")
            )
        ).scalar_one()
        assert note.raw_text == "01/09/2026: b"
        assert note.status == "pending"  # needs re-embedding
        assert len(deps.fast_queue.calls) >= 2  # first create + this replace


async def test_resolve_events_table_conflict_rejects_the_wrong_user(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    session_id = await deps.conflicts_cache.create(
        user_id=1,
        conflicts=[
            PendingConflict(
                note_id=1,
                old_text="a",
                new_text="b",
                new_tags=[],
                new_structured={},
            )
        ],
    )
    resolved = await resolve_events_table_conflict(
        deps, session_id=session_id, index=0, keep_new=True, user_id=999
    )
    assert resolved is False


async def test_confirm_events_table_group_notes_have_no_visibility(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    events = [
        ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="event", text="a")
    ]
    session_id = await deps.pending_events_cache.create(
        chat_id=-100, user_id=1, is_group=True, tab_name="t", topic_tags=[], events=events
    )

    result = await confirm_events_table(deps, session_id=session_id, user_id=1)
    assert result.ok is True

    async with deps.session_factory() as session:
        note = (
            await session.execute(
                select(Note).where(Note.user_id == 1, Note.source_type == "table_event")
            )
        ).scalar_one()
        assert note.is_group is True
        assert note.visibility is None


async def test_confirm_events_table_rejects_the_wrong_user(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    events = [
        ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="event", text="a")
    ]
    session_id = await deps.pending_events_cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=[], events=events
    )

    result = await confirm_events_table(deps, session_id=session_id, user_id=999)

    assert result.ok is False
    # Untouched — still there for the real owner to confirm or cancel.
    assert await deps.pending_events_cache.get(session_id) is not None


async def test_confirm_events_table_expired_session(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    result = await confirm_events_table(deps, session_id="does-not-exist", user_id=1)
    assert result.ok is False


async def test_cancel_events_table_discards_the_session(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    events = [
        ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="event", text="a")
    ]
    session_id = await deps.pending_events_cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=[], events=events
    )

    assert await cancel_events_table(deps, session_id=session_id, user_id=1) is True
    assert await deps.pending_events_cache.get(session_id) is None


async def test_cancel_events_table_rejects_the_wrong_user(db_engine, redis_client):
    deps = _deps_with_events_cache(db_engine, redis_client)
    events = [
        ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="event", text="a")
    ]
    session_id = await deps.pending_events_cache.create(
        chat_id=1, user_id=1, is_group=False, tab_name="t", topic_tags=[], events=events
    )

    assert await cancel_events_table(deps, session_id=session_id, user_id=999) is False
    assert await deps.pending_events_cache.get(session_id) is not None
