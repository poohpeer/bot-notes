"""RQ job functions — see docs/architecture/03-ingest.md and 08-roadmap.md.

RQ runs job functions synchronously; each wraps an async implementation via
`asyncio.run`, per job. Dependencies (engine, embedding client, extractors)
are built lazily and cached at module scope — the worker process constructs
them once and reuses them across jobs, rather than reconnecting per note.

`process_note` is the same job function for every source_type; only which
*queue* a note lands in decides whether a fast or heavy worker (and thus
which Docker image — see 06-deployment.md) ever picks it up. A heavy-only
extractor (voice, instagram) can safely live in this module even though it's
imported by the plain `app` image too: its heavy dependencies
(faster-whisper, real yt-dlp media download) are only ever imported lazily,
inside the client classes themselves, the first time a job actually needs
them — see clients/transcribe.py's module docstring. A fast worker never
receives a heavy-queued job in the first place, so that import never
happens there.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from notes_bot.clients.embeddings import EmbeddingServiceError, HttpEmbeddingClient
from notes_bot.clients.telegram_files import TelegramFileClient
from notes_bot.clients.transcribe import FasterWhisperClient
from notes_bot.config import Settings, get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.models import Note
from notes_bot.db.repositories import ChunkRepository, NoteRepository
from notes_bot.domain.chunking import chunk
from notes_bot.extractors.base import Extractor
from notes_bot.extractors.instagram import InstagramExtractor
from notes_bot.extractors.map import MapExtractor
from notes_bot.extractors.page import PageExtractor
from notes_bot.extractors.voice import VoiceExtractor
from notes_bot.extractors.youtube import YoutubeExtractor

log = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None
_extractors: dict[str, Extractor] | None = None


def _get_session_factory(settings: Settings) -> async_sessionmaker:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_engine(settings)
        _session_factory = create_session_factory(_engine)
    return _session_factory


def _get_extractors(settings: Settings) -> dict[str, Extractor]:
    """Built once per process and reused — each extractor is stateless (or
    only holds injected clients). One FasterWhisperClient instance is
    shared between voice and instagram so the model, once loaded, is loaded
    only once per worker — see clients/transcribe.py.
    """
    global _extractors
    if _extractors is None:
        transcription_client = FasterWhisperClient(model_size=settings.whisper_model)
        file_client = TelegramFileClient(settings.telegram_bot_token)
        _extractors = {
            "page": PageExtractor(),
            "youtube": YoutubeExtractor(),
            "map": MapExtractor(),
            "voice": VoiceExtractor(
                file_client=file_client, transcription_client=transcription_client
            ),
            "instagram": InstagramExtractor(
                transcription_client=transcription_client,
                max_download_bytes=settings.max_download_bytes,
                max_audio_seconds=settings.max_audio_seconds,
            ),
        }
    return _extractors


async def _extract_and_get_index_text(
    note: Note, note_repo: NoteRepository, extractors: dict[str, Extractor]
) -> str:
    """Runs the matching extractor (if any), persists what it found, and
    returns the text that actually gets chunked — see 02-data-model.md,
    "Текст, идущий в индекс", and 03-ingest.md, "Деградация".

    A plain 'text' note has no extractor: raw_text is already the text. For
    everything else, extracted_text wins when non-empty; an extractor that
    failed or found nothing degrades to raw_text (the URL itself, still
    findable) rather than failing the note.
    """
    if note.source_type == "text":
        return note.raw_text or ""

    extractor = extractors.get(note.source_type)
    if extractor is None:
        raise NotImplementedError(f"no extractor registered for source_type={note.source_type!r}")

    result = await extractor.extract(note)
    await note_repo.record_extraction(
        note.id,
        extracted_text=result.text or None,
        lang=result.lang,
        error=result.meta.get("error"),
    )
    return result.text or note.raw_text or ""


async def process_note_async(
    note_id: int,
    *,
    session_factory: async_sessionmaker,
    embedding_client: HttpEmbeddingClient,
    extractors: dict[str, Extractor] | None = None,
) -> None:
    # Empty, not a module-level default: a 'text' note never consults this
    # (see _extract_and_get_index_text), and every other source_type is
    # expected to pass its own — process_note() below always does, built
    # from settings via _get_extractors().
    extractors = extractors if extractors is not None else {}
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
            text = await _extract_and_get_index_text(note, note_repo, extractors)
            chunks = chunk(text)
            if not chunks:
                # Only reached with nothing left to index at all — not even
                # raw_text/the URL. This IS a real failure: the note is
                # physically unfindable — see "Деградация".
                raise ValueError("note has no text to index, even after degradation")

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
            note_id,
            session_factory=session_factory,
            embedding_client=embedding_client,
            extractors=_get_extractors(settings),
        )
    )
