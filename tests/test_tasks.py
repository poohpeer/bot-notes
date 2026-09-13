from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.clients.embeddings import EmbeddingServiceError
from notes_bot.db.models import Note, NoteChunk
from notes_bot.db.repositories import NoteRepository
from notes_bot.queue.tasks import process_note_async

pytestmark = pytest.mark.asyncio


class FakeEmbeddingClient:
    model_name = "fake-model"
    dim = 768

    def __init__(self, vector_value: float = 0.5, fail: Exception | None = None) -> None:
        self._vector_value = vector_value
        self._fail = fail
        self.calls: list[list[str]] = []

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self._fail:
            raise self._fail
        return [[self._vector_value] * 768 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


@pytest.fixture
async def factory(db_engine):
    """Bound directly to the shared test engine, not wrapped in db_session's
    rolled-back transaction: process_note_async opens and commits its own
    separate transactions, exactly as it does against a real database. So
    this fixture tracks and deletes whatever it inserts itself, even when
    the test body raises, rather than leaving a row for the next test."""
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    created_ids: list[int] = []

    async def insert_pending_text_note(*, text: str = "hello world") -> int:
        async with sf() as session:
            note = Note(
                user_id=1,
                chat_id=1,
                is_group=False,
                visibility="private",
                source_type="text",
                raw_text=text,
                status="pending",
            )
            session.add(note)
            await session.commit()
            created_ids.append(note.id)
            return note.id

    sf.insert_pending_text_note = insert_pending_text_note  # type: ignore[attr-defined]

    yield sf

    async with sf() as session:
        for note_id in created_ids:
            note = await session.get(Note, note_id)
            if note is not None:
                await session.delete(note)
        await session.commit()


async def test_process_note_embeds_and_marks_done(factory):
    note_id = await factory.insert_pending_text_note(text="one two three four five")
    client = FakeEmbeddingClient()

    await process_note_async(note_id, session_factory=factory, embedding_client=client)

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.status == "done"
        chunks = (
            (await session.execute(select(NoteChunk).where(NoteChunk.note_id == note_id)))
            .scalars()
            .all()
        )
        assert len(chunks) == 1
        assert chunks[0].embedding_model == "fake-model"


async def test_process_note_marks_failed_on_embedding_error(factory):
    note_id = await factory.insert_pending_text_note()
    client = FakeEmbeddingClient(fail=EmbeddingServiceError(503, "model not loaded"))

    await process_note_async(note_id, session_factory=factory, embedding_client=client)

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.status == "failed"
        assert "503" in note.error
        assert note.attempts == 1


async def test_process_note_marks_failed_on_empty_text(factory):
    note_id = await factory.insert_pending_text_note(text="   ")
    client = FakeEmbeddingClient()

    await process_note_async(note_id, session_factory=factory, embedding_client=client)

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.status == "failed"
        assert client.calls == []  # never reached the embedding call


async def test_process_note_skips_an_already_done_note(factory):
    """A duplicate job (see ADR-8) for a note some other worker already
    finished must not re-embed it."""
    note_id = await factory.insert_pending_text_note()
    async with factory() as session:
        await NoteRepository(session).mark_done(note_id)
        await session.commit()

    client = FakeEmbeddingClient()
    await process_note_async(note_id, session_factory=factory, embedding_client=client)
    assert client.calls == []


async def test_process_note_replaces_chunks_on_reprocessing(factory):
    """Editing a note re-chunks and re-embeds it — invariant 5 in
    02-data-model.md: no leftover chunks from the previous text."""
    note_id = await factory.insert_pending_text_note(text="first version of the note")
    client = FakeEmbeddingClient()
    await process_note_async(note_id, session_factory=factory, embedding_client=client)

    async with factory() as session:
        note = await session.get(Note, note_id)
        note.raw_text = "a completely different second version now"
        note.status = "pending"
        await session.commit()

    await process_note_async(note_id, session_factory=factory, embedding_client=client)

    async with factory() as session:
        chunks = (
            (await session.execute(select(NoteChunk).where(NoteChunk.note_id == note_id)))
            .scalars()
            .all()
        )
        assert len(chunks) == 1
        assert "different second version" in chunks[0].chunk_text
