from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.clients.llm import LLMResult, LLMServiceError
from notes_bot.db.models import Note, NoteChunk
from notes_bot.db.repositories import NoteRepository
from notes_bot.queue.tasks import enrich_note_async

pytestmark = pytest.mark.asyncio

MODEL = "fake-model"


class FakeEmbeddingClient:
    model_name = MODEL
    dim = 768

    async def embed_passages(self, texts):
        raise NotImplementedError

    async def embed_query(self, text):
        raise NotImplementedError


class ScriptedLLMClient:
    """One fixed parsed payload per prompt-file keyword found in `system`,
    so a single client can answer title/tags/summary/place/dupes
    differently within one test."""

    def __init__(self, **by_keyword):
        self._by_keyword = by_keyword
        self.calls: list[dict] = []
        self.fail_with: Exception | None = None

    async def complete(self, *, system, user, json_schema=None, history=None, timeout_s=60.0):
        self.calls.append({"system": system, "user": user})
        if self.fail_with:
            raise self.fail_with
        for keyword, parsed in self._by_keyword.items():
            if keyword in system:
                return LLMResult(text="", parsed=parsed, model="scripted", usage=None)
        return LLMResult(text="", parsed=None, model="scripted", usage=None)


def _vec(x: float, y: float) -> list[float]:
    return [x, y] + [0.0] * 766


@pytest.fixture
async def factory(db_engine):
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    created_ids: list[int] = []

    async def insert_done_note(
        *, source_type="text", raw_text="hello", extracted_text=None, embedding=None
    ) -> int:
        async with sf() as session:
            note = Note(
                user_id=1,
                chat_id=1,
                is_group=False,
                visibility="private",
                source_type=source_type,
                raw_text=raw_text,
                extracted_text=extracted_text,
                status="done",
            )
            session.add(note)
            await session.flush()
            session.add(
                NoteChunk(
                    note_id=note.id,
                    chunk_index=0,
                    chunk_text=extracted_text or raw_text,
                    embedding=embedding or _vec(1.0, 0.0),
                    embedding_model=MODEL,
                )
            )
            await session.commit()
            created_ids.append(note.id)
            return note.id

    sf.insert_done_note = insert_done_note  # type: ignore[attr-defined]

    yield sf

    async with sf() as session:
        for note_id in created_ids:
            note = await session.get(Note, note_id)
            if note is not None:
                await session.delete(note)
        await session.commit()


async def test_enrich_note_skips_a_note_not_yet_done(factory):
    async with factory() as session:
        note = Note(
            user_id=1,
            chat_id=1,
            is_group=False,
            visibility="private",
            source_type="text",
            raw_text="x",
            status="pending",
        )
        session.add(note)
        await session.commit()
        note_id = note.id

    llm = ScriptedLLMClient()
    await enrich_note_async(
        note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        llm_enabled=True,
        timeout_s=10,
    )
    assert llm.calls == []

    async with factory() as session:
        n = await session.get(Note, note_id)
        await session.delete(n)
        await session.commit()


async def test_enrich_note_marks_skipped_when_llm_disabled(factory):
    note_id = await factory.insert_done_note()
    llm = ScriptedLLMClient(title={"title": "should not be used"})

    await enrich_note_async(
        note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        llm_enabled=False,
        timeout_s=10,
    )
    assert llm.calls == []

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.enrich_status == "skipped"
        assert note.title is None


async def test_enrich_note_writes_title_tags_summary(factory):
    note_id = await factory.insert_done_note(raw_text="a hinkali restaurant review")
    # Keywords match a distinctive word from each prompt file (title.md,
    # tags.md, summary.md) — all in Russian, so "title"/"tags"/"summary"
    # themselves never appear in the actual system prompt text.
    llm = ScriptedLLMClient(
        **{
            "заголовок": {"title": "Хинкальная"},
            "тегов": {"tags": ["food", "tbilisi"]},
            "пересказа": {"summary": "A short review"},
        }
    )

    await enrich_note_async(
        note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        llm_enabled=True,
        timeout_s=10,
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.title == "Хинкальная"
        assert note.tags == ["food", "tbilisi"]
        assert note.summary == "A short review"
        assert note.enrich_status == "done"


async def test_enrich_note_only_calls_place_prompt_for_map_notes(factory):
    text_note_id = await factory.insert_done_note(source_type="text")
    # generate_place() is gated on source_type == "map" in enrich_note_async
    # itself — no scripted reply needed here, since it should never be
    # called at all for a plain text note.
    llm = ScriptedLLMClient()

    await enrich_note_async(
        text_note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        llm_enabled=True,
        timeout_s=10,
    )
    async with factory() as session:
        note = await NoteRepository(session).get(text_note_id)
        assert note.structured == {}


async def test_enrich_note_calls_place_prompt_for_map_notes(factory):
    map_note_id = await factory.insert_done_note(
        source_type="map", raw_text="Хинкальная на Руставели"
    )
    llm = ScriptedLLMClient(
        **{
            "заведения": {"name": "Хинкальная", "district": None, "cuisine": "georgian"},
        }
    )

    await enrich_note_async(
        map_note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        llm_enabled=True,
        timeout_s=10,
    )

    async with factory() as session:
        note = await NoteRepository(session).get(map_note_id)
        assert note.structured == {"name": "Хинкальная", "cuisine": "georgian"}


async def test_enrich_note_finds_and_records_a_duplicate(factory):
    original_id = await factory.insert_done_note(
        raw_text="closes at midnight", embedding=_vec(1.0, 0.0)
    )
    new_id = await factory.insert_done_note(
        raw_text="open until midnight", embedding=_vec(0.99, 0.01)
    )
    llm = ScriptedLLMClient(
        **{"Определи": {"is_duplicate": True, "duplicate_note_id": original_id}}
    )

    await enrich_note_async(
        new_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        llm_enabled=True,
        timeout_s=10,
    )

    async with factory() as session:
        note = await NoteRepository(session).get(new_id)
        assert note.structured.get("possible_duplicate_of") == original_id


async def test_enrich_note_marks_failed_on_llm_service_error(factory):
    note_id = await factory.insert_done_note()
    llm = ScriptedLLMClient()
    llm.fail_with = LLMServiceError(503, "quota_exhausted", "no accounts left")

    await enrich_note_async(
        note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        llm_enabled=True,
        timeout_s=10,
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.enrich_status == "failed"
        assert "quota_exhausted" in note.error


async def test_enrich_note_does_not_touch_status_or_extracted_text(factory):
    note_id = await factory.insert_done_note(extracted_text="the extracted text")
    llm = ScriptedLLMClient(title={"title": "T"})

    await enrich_note_async(
        note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        llm_enabled=True,
        timeout_s=10,
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.status == "done"
        assert note.extracted_text == "the extracted text"
