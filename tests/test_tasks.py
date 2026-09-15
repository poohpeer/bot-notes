from __future__ import annotations

from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.clients.embeddings import EmbeddingServiceError
from notes_bot.db.models import Note, NoteChunk
from notes_bot.db.repositories import NoteRepository, UserSettingsRepository
from notes_bot.queue.tasks import process_note_async

pytestmark = pytest.mark.asyncio


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send(self, chat_id, text):
        self.sent.append((chat_id, text))


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

    async def insert_pending_note(
        *, source_type: str, source_url: str | None = None, raw_text: str | None = ""
    ) -> int:
        async with sf() as session:
            note = Note(
                user_id=1,
                chat_id=1,
                is_group=False,
                visibility="private",
                source_type=source_type,
                source_url=source_url,
                raw_text=raw_text,
                status="pending",
            )
            session.add(note)
            await session.commit()
            created_ids.append(note.id)
            return note.id

    sf.insert_pending_text_note = insert_pending_text_note  # type: ignore[attr-defined]
    sf.insert_pending_note = insert_pending_note  # type: ignore[attr-defined]

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
    duration_before = (
        REGISTRY.get_sample_value(
            "notes_process_note_duration_seconds_count", {"source_type": "text"}
        )
        or 0.0
    )

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
    # 06-deployment.md, "Время обработки по source_type" — recorded on
    # success too, not only on the failure paths below.
    assert (
        REGISTRY.get_sample_value(
            "notes_process_note_duration_seconds_count", {"source_type": "text"}
        )
        == duration_before + 1
    )


async def test_process_note_marks_failed_on_embedding_error(factory):
    note_id = await factory.insert_pending_text_note()
    client = FakeEmbeddingClient(fail=EmbeddingServiceError(503, "model not loaded"))
    failed_before = (
        REGISTRY.get_sample_value("notes_status_failed_total", {"source_type": "text"}) or 0.0
    )

    await process_note_async(note_id, session_factory=factory, embedding_client=client)

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.status == "failed"
        assert "503" in note.error
        assert note.attempts == 1
    assert (
        REGISTRY.get_sample_value("notes_status_failed_total", {"source_type": "text"})
        == failed_before + 1
    )


async def test_process_note_sends_debug_notification_when_enabled(factory):
    note_id = await factory.insert_pending_text_note()
    async with factory() as session:
        await UserSettingsRepository(session).toggle_debug(user_id=1)
        await session.commit()
    sender = FakeSender()

    await process_note_async(
        note_id, session_factory=factory, embedding_client=FakeEmbeddingClient(), sender=sender
    )

    assert len(sender.sent) == 1
    chat_id, text = sender.sent[0]
    assert chat_id == 1
    assert "Обработка завершена" in text

    # user_settings rows outlive this fixture's per-test cleanup (it only
    # tracks notes) — reset so a later test doesn't inherit user_id=1 stuck
    # with debug on.
    async with factory() as session:
        await UserSettingsRepository(session).toggle_debug(user_id=1)
        await session.commit()


async def test_process_note_sends_nothing_when_debug_disabled(factory):
    note_id = await factory.insert_pending_text_note()
    sender = FakeSender()

    await process_note_async(
        note_id, session_factory=factory, embedding_client=FakeEmbeddingClient(), sender=sender
    )

    assert sender.sent == []


async def test_process_note_sends_no_debug_timing_on_failure_even_with_debug_enabled(factory):
    """The /debug timing message (render_debug_processing_done) is a
    success-only thing — a failure gets render_processing_failed instead
    (below), never both."""
    note_id = await factory.insert_pending_text_note()
    async with factory() as session:
        await UserSettingsRepository(session).toggle_debug(user_id=1)
        await session.commit()
    sender = FakeSender()
    client = FakeEmbeddingClient(fail=EmbeddingServiceError(503, "model not loaded"))

    await process_note_async(
        note_id, session_factory=factory, embedding_client=client, sender=sender
    )

    assert len(sender.sent) == 1
    assert "Обработка завершена" not in sender.sent[0][1]

    async with factory() as session:
        await UserSettingsRepository(session).toggle_debug(user_id=1)
        await session.commit()


async def test_process_note_notifies_the_user_on_embedding_failure(factory):
    """Previously only clients/alerts.py's ops-side alert fired on a
    failure — the person who actually sent the note got silence, no
    different from one still quietly processing."""
    note_id = await factory.insert_pending_text_note()
    sender = FakeSender()
    client = FakeEmbeddingClient(fail=EmbeddingServiceError(503, "model not loaded"))

    await process_note_async(
        note_id, session_factory=factory, embedding_client=client, sender=sender
    )

    assert len(sender.sent) == 1
    chat_id, text = sender.sent[0]
    assert chat_id == 1
    assert "Не получилось обработать заметку" in text


async def test_process_note_notifies_the_user_on_an_unexpected_exception(factory):
    note_id = await factory.insert_pending_note(source_type="page", source_url="https://x")
    sender = FakeSender()

    class BrokenExtractor:
        async def extract(self, note):
            raise ValueError("boom")

    await process_note_async(
        note_id,
        session_factory=factory,
        embedding_client=FakeEmbeddingClient(),
        extractors={"page": BrokenExtractor()},
        sender=sender,
    )

    assert len(sender.sent) == 1
    chat_id, text = sender.sent[0]
    assert chat_id == 1
    assert "Не получилось обработать заметку" in text


class FakeAlerts:
    def __init__(self) -> None:
        self.failed_calls: list[dict] = []

    async def reclaimed(self, **kw):
        pass

    async def abandoned(self, **kw):
        pass

    async def failed(self, **kw):
        self.failed_calls.append(kw)


async def test_process_note_alerts_on_a_real_exception(factory):
    """Unlike a stuck/reclaimed note (nothing to say why), a caught
    exception has a real reason — the alert should carry it."""
    note_id = await factory.insert_pending_text_note()
    client = FakeEmbeddingClient(fail=EmbeddingServiceError(503, "model not loaded"))
    alerts = FakeAlerts()

    await process_note_async(
        note_id, session_factory=factory, embedding_client=client, alerts=alerts
    )

    [call] = alerts.failed_calls
    assert call["note_id"] == note_id
    assert call["source_type"] == "text"
    assert "503" in call["error"]


async def test_process_note_never_alerts_on_success(factory):
    note_id = await factory.insert_pending_text_note()
    alerts = FakeAlerts()

    await process_note_async(
        note_id, session_factory=factory, embedding_client=FakeEmbeddingClient(), alerts=alerts
    )

    assert alerts.failed_calls == []


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


class FakeExtractor:
    source_type = "page"
    queue = "fast"

    def __init__(self, *, text="", title=None, lang=None, error=None):
        self._result = SimpleNamespace(
            text=text, title=title, lang=lang, meta=({"error": error} if error else {})
        )

    async def extract(self, note):
        return self._result


async def test_process_note_indexes_extracted_text_for_a_page_note(factory):
    note_id = await factory.insert_pending_note(
        source_type="page", source_url="https://example.com/a", raw_text="https://example.com/a"
    )
    extractor = FakeExtractor(text="the extracted page content", lang="en")
    client = FakeEmbeddingClient()

    await process_note_async(
        note_id,
        session_factory=factory,
        embedding_client=client,
        extractors={"page": extractor},
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.status == "done"
        assert note.extracted_text == "the extracted page content"
        assert note.lang == "en"
        chunks = (
            (await session.execute(select(NoteChunk).where(NoteChunk.note_id == note_id)))
            .scalars()
            .all()
        )
        assert "extracted page content" in chunks[0].chunk_text


async def test_process_note_degrades_to_raw_text_when_extraction_fails(factory):
    """03-ingest.md, "Деградация": a blocked/failed extractor doesn't fail
    the note — it indexes raw_text (the URL) instead, and still done."""
    note_id = await factory.insert_pending_note(
        source_type="page",
        source_url="https://evil.example/",
        raw_text="https://evil.example/",
    )
    extractor = FakeExtractor(text="", error="blocked address")
    client = FakeEmbeddingClient()

    await process_note_async(
        note_id,
        session_factory=factory,
        embedding_client=client,
        extractors={"page": extractor},
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.status == "done"
        assert note.extracted_text is None
        assert note.error == "blocked address"
        chunks = (
            (await session.execute(select(NoteChunk).where(NoteChunk.note_id == note_id)))
            .scalars()
            .all()
        )
        assert chunks[0].chunk_text == "https://evil.example/"


async def test_process_note_fails_when_nothing_can_be_indexed_at_all(factory):
    """Only when even raw_text is empty does degradation run out — the note
    is physically unfindable, so this is the one case that is a real
    failure."""
    note_id = await factory.insert_pending_note(
        source_type="page", source_url="https://evil.example/", raw_text=""
    )
    extractor = FakeExtractor(text="", error="blocked address")
    client = FakeEmbeddingClient()

    await process_note_async(
        note_id,
        session_factory=factory,
        embedding_client=client,
        extractors={"page": extractor},
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.status == "failed"


async def test_process_note_raises_for_a_source_type_with_no_extractor_yet(factory):
    note_id = await factory.insert_pending_note(
        source_type="instagram", source_url="https://instagram.com/p/x", raw_text="caption"
    )
    client = FakeEmbeddingClient()

    await process_note_async(
        note_id, session_factory=factory, embedding_client=client, extractors={}
    )

    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        # Caught by process_note_async's generic except clause, same as any
        # other extraction failure — not a crash of the worker process.
        assert note.status == "failed"
