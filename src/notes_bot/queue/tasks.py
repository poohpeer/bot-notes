"""RQ job functions — see docs/architecture/03-ingest.md and 08-roadmap.md, M2.

RQ runs job functions synchronously; each wraps an async implementation via
`asyncio.run`, per job. Dependencies (engine, embedding client) are built
lazily and cached at module scope — the worker process constructs them once
and reuses them across jobs, rather than reconnecting per note.

M2 scope: `source_type='text'` only. Other source types (page, youtube,
instagram, map, voice) are M3/M4 and raise `NotImplementedError` here for
now, rather than silently mis-indexing them.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from notes_bot.clients.embeddings import EmbeddingServiceError, HttpEmbeddingClient
from notes_bot.config import Settings, get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.models import Note
from notes_bot.db.repositories import ChunkRepository, NoteRepository
from notes_bot.domain.chunking import chunk

log = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def _get_session_factory(settings: Settings) -> async_sessionmaker:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_engine(settings)
        _session_factory = create_session_factory(_engine)
    return _session_factory


def _index_text(note: Note) -> str:
    """What gets chunked and embedded, by source_type — see
    02-data-model.md, "Текст, идущий в индекс". Only 'text' exists yet."""
    if note.source_type == "text":
        return note.raw_text or ""
    raise NotImplementedError(
        f"source_type={note.source_type!r} is not indexed yet — lands in M3/M4"
    )


async def process_note_async(
    note_id: int,
    *,
    session_factory: async_sessionmaker,
    embedding_client: HttpEmbeddingClient,
) -> None:
    async with session_factory() as session:
        note_repo = NoteRepository(session)
        note = await note_repo.get(note_id)
        if note is None:
            log.warning("process_note: note_id=%s no longer exists, skipping", note_id)
            return
        if note.status not in ("pending", "processing"):
            # Already done or failed — a re-enqueued or duplicate job (see
            # ADR-8) that arrived after the first run finished.
            log.info(
                "process_note: note_id=%s status=%s already settled, skipping",
                note_id,
                note.status,
            )
            return

        await note_repo.mark_processing(note_id)
        await session.commit()

    async with session_factory() as session:
        note_repo = NoteRepository(session)
        chunk_repo = ChunkRepository(session)
        note = await note_repo.get(note_id)
        assert note is not None  # just fetched above; nothing else deletes notes

        try:
            text = _index_text(note)
            chunks = chunk(text)
            if not chunks:
                raise ValueError("note has no text to index")

            vectors = await embedding_client.embed_passages([c.text for c in chunks])
            new_chunks = ChunkRepository.from_domain_chunks(chunks, vectors)

            await chunk_repo.replace_chunks(
                note_id, new_chunks, embedding_model=embedding_client.model_name
            )
            await note_repo.mark_done(note_id)
            await session.commit()
            log.info("process_note: note_id=%s done, chunks=%d", note_id, len(new_chunks))

        except EmbeddingServiceError as exc:
            await session.rollback()
            async with session_factory() as failure_session:
                await NoteRepository(failure_session).mark_failed(note_id, str(exc))
                await failure_session.commit()
            log.warning(
                "process_note: note_id=%s failed (retryable=%s): %s",
                note_id,
                exc.retryable,
                exc,
            )
        except Exception as exc:  # noqa: BLE001 — logged and recorded, not swallowed
            await session.rollback()
            async with session_factory() as failure_session:
                await NoteRepository(failure_session).mark_failed(note_id, str(exc))
                await failure_session.commit()
            log.exception("process_note: note_id=%s failed unexpectedly", note_id)


def process_note(note_id: int) -> None:
    """The RQ-registered entrypoint — `fast` queue, see 06-deployment.md."""
    settings = get_settings()
    session_factory = _get_session_factory(settings)
    embedding_client = HttpEmbeddingClient(
        settings.embeddings_url,
        model_name=settings.embedding_model_name,
        dim=settings.embedding_dim,
    )
    asyncio.run(
        process_note_async(
            note_id, session_factory=session_factory, embedding_client=embedding_client
        )
    )
