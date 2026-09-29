from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.clients.llm import LLMResult, LLMServiceError
from notes_bot.clients.prompts import load_prompt
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


class ReembeddingClient(FakeEmbeddingClient):
    """Only the correction-and-reembed path (enrich_note_async) actually
    calls embed_passages — every other test in this file must never reach
    it, so FakeEmbeddingClient's own NotImplementedError stays the
    default; this subclass opts specific tests in."""

    async def embed_passages(self, texts):
        return [[0.0, 0.0] + [0.0] * 766 for _ in texts]


class ScriptedLLMClient:
    """One scripted reply for enrich()'s combined call, another for
    find_duplicate's - distinguished by system prompt identity (each is
    its own prompt file, loaded once via load_prompt)."""

    def __init__(self, *, enrich=None, dupes=None):
        self._enrich_prompt = load_prompt("enrich")
        self._dupes_prompt = load_prompt("dupes")
        self._enrich = enrich
        self._dupes = dupes
        self.calls: list[dict] = []
        self.fail_with: Exception | None = None

    async def complete(self, *, system, user, json_schema=None, history=None, timeout_s=60.0):
        self.calls.append({"system": system, "user": user})
        if self.fail_with:
            raise self.fail_with
        if system == self._enrich_prompt:
            return LLMResult(text="", parsed=self._enrich, model="scripted", usage=None)
        if system == self._dupes_prompt:
            return LLMResult(text="", parsed=self._dupes, model="scripted", usage=None)
        return LLMResult(text="", parsed=None, model="scripted", usage=None)


def _enrich_payload(**overrides) -> dict:
    payload = {"title": None, "tags": [], "summary": None}
    payload.update(overrides)
    return payload


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
    llm = ScriptedLLMClient(enrich=_enrich_payload(title="should not be used"))

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
    llm = ScriptedLLMClient(
        enrich=_enrich_payload(
            title="Хинкальная", tags=["food", "tbilisi"], summary="A short review"
        )
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


async def test_enrich_note_structured_empty_for_text_notes(factory):
    """`place`/`places` are gated on source_type inside enrich() itself -
    a plain text note gets neither, even if the model filled one anyway."""
    text_note_id = await factory.insert_done_note(source_type="text")
    llm = ScriptedLLMClient(
        enrich=_enrich_payload(place={"name": "should be dropped for source_type=text"})
    )

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


async def test_enrich_note_writes_place_for_map_notes(factory):
    map_note_id = await factory.insert_done_note(
        source_type="map", raw_text="Хинкальная на Руставели"
    )
    llm = ScriptedLLMClient(
        enrich=_enrich_payload(
            place={"name": "Хинкальная", "district": None, "cuisine": "georgian"}
        )
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
    llm = ScriptedLLMClient(dupes={"is_duplicate": True, "duplicate_note_id": original_id})

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
    llm = ScriptedLLMClient(enrich=_enrich_payload(title="T"))

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


async def test_enrich_note_applies_corrections_and_reembeds(factory):
    """The whole reason `enrich()` returns `corrections` at all (03-ingest.md,
    "Обогащение") - an ASR mishearing ("Белиси" for "Тбилиси") that made it
    into extracted_text must be fixed and re-embedded, not just reflected
    in the title."""
    note_id = await factory.insert_done_note(
        source_type="instagram",
        raw_text="https://instagram.com/reel/x",
        extracted_text="Мы поехали в Белиси на выходные",
    )
    llm = ScriptedLLMClient(
        enrich=_enrich_payload(corrections=[{"wrong": "Белиси", "correct": "Тбилиси"}])
    )

    await enrich_note_async(
        note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=ReembeddingClient(),
        llm_enabled=True,
        timeout_s=10,
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.extracted_text == "Мы поехали в Тбилиси на выходные"
        assert note.status == "done"  # never left 'done' at any persisted point
        chunks = (
            (await session.execute(select(NoteChunk).where(NoteChunk.note_id == note_id)))
            .scalars()
            .all()
        )
        assert any("Тбилиси" in c.chunk_text for c in chunks)


async def test_enrich_note_with_no_corrections_never_reembeds(factory):
    """Guards the no-op path: an untouched extracted_text/chunk_text means
    replace_chunks was never called, not just that the text matches by
    coincidence."""
    note_id = await factory.insert_done_note(
        source_type="instagram",
        raw_text="https://instagram.com/reel/x",
        extracted_text="Мы поехали в Тбилиси",
    )
    llm = ScriptedLLMClient(enrich=_enrich_payload())

    await enrich_note_async(
        note_id,
        session_factory=factory,
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),  # embed_passages would raise if called
        llm_enabled=True,
        timeout_s=10,
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.extracted_text == "Мы поехали в Тбилиси"
