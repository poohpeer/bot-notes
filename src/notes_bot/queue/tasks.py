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
import time
from html import escape as _esc

import redis.asyncio as async_redis
from redis import Redis
from rq import Queue
from sqlalchemy import and_
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from notes_bot.bot.keyboards import events_table_confirm_keyboard
from notes_bot.bot.render import (
    render_debug_processing_done,
    render_events_table_disabled,
    render_events_table_empty,
    render_events_table_preview,
    render_places,
    render_processing_failed,
    render_smart_answer_debug,
    render_smart_answer_failed,
)
from notes_bot.clients.alerts import AlertNotifier, NullNotifier, build_notifier
from notes_bot.clients.embeddings import EmbeddingServiceError, HttpEmbeddingClient
from notes_bot.clients.llm import (
    ImageInput,
    LLMClient,
    LLMServiceError,
    NullLLMClient,
    ProxyAILLMClient,
    extract_token_usage,
)
from notes_bot.clients.pending_events_cache import PendingEventsCache
from notes_bot.clients.prompts import load_prompt
from notes_bot.clients.telegram_files import TelegramFileClient
from notes_bot.clients.telegram_sender import Sender, TelegramSender
from notes_bot.clients.transcribe import FasterWhisperClient
from notes_bot.clients.translate import HttpTranslateClient, TranslateServiceError
from notes_bot.config import Settings, get_settings
from notes_bot.db.engine import create_engine, create_session_factory
from notes_bot.db.models import Note
from notes_bot.db.repositories import ChunkRepository, NoteRepository, UserSettingsRepository
from notes_bot.db.search import search_notes
from notes_bot.domain.acl import visibility_predicate
from notes_bot.domain.chunking import chunk
from notes_bot.enrich import (
    find_duplicate,
    generate_place,
    generate_places,
    generate_summary,
    generate_tags,
    generate_title,
)
from notes_bot.events_table import extract_events_table
from notes_bot.extractors.base import Extractor
from notes_bot.extractors.instagram import InstagramExtractor
from notes_bot.extractors.map import MapExtractor
from notes_bot.extractors.page import PageExtractor
from notes_bot.extractors.voice import VoiceExtractor
from notes_bot.extractors.youtube import YoutubeExtractor
from notes_bot.metrics import (
    EXTRACTOR_FAILURES_TOTAL,
    NOTES_ENRICH_FAILED_TOTAL,
    NOTES_STATUS_FAILED_TOTAL,
    PROCESS_NOTE_DURATION_SECONDS,
)

log = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None
_extractors: dict[str, Extractor] | None = None
_llm_queue: Queue | None = None
_alerts: AlertNotifier | None = None
_pending_events_cache: PendingEventsCache | None = None


def _get_pending_events_cache(settings: Settings) -> PendingEventsCache:
    global _pending_events_cache
    if _pending_events_cache is None:
        _pending_events_cache = PendingEventsCache(async_redis.from_url(settings.redis_url))
    return _pending_events_cache


def _get_session_factory(settings: Settings) -> async_sessionmaker:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_engine(settings)
        _session_factory = create_session_factory(_engine)
    return _session_factory


def _get_llm_queue(settings: Settings) -> Queue:
    global _llm_queue
    if _llm_queue is None:
        _llm_queue = Queue("llm", connection=Redis.from_url(settings.redis_url))
    return _llm_queue


def _get_alerts(settings: Settings) -> AlertNotifier:
    global _alerts
    if _alerts is None:
        _alerts = build_notifier(
            settings.alerts_telegram_bot_token,
            settings.alerts_telegram_chat_id,
            min_interval_minutes=settings.alerts_min_interval_minutes,
        )
    return _alerts


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

    'table_event' (03-ingest.md, "/events_table") is the other no-extractor
    case: its raw_text is already the final event text, produced by the
    extraction *call* itself (events_table.py), not something a further
    extractor step would improve on.
    """
    if note.source_type in ("text", "table_event"):
        return note.raw_text or ""

    extractor = extractors.get(note.source_type)
    if extractor is None:
        raise NotImplementedError(f"no extractor registered for source_type={note.source_type!r}")

    result = await extractor.extract(note)
    error = result.meta.get("error")
    if error:
        # 06-deployment.md, "Доля ошибок yt-dlp по источникам" — the metric
        # the doc says "заслуживает алерта": a degraded-but-not-failed
        # extraction (see this function's own docstring) is exactly the
        # early signal that a source started blocking scrapes.
        EXTRACTOR_FAILURES_TOTAL.labels(source_type=note.source_type).inc()
    await note_repo.record_extraction(
        note.id,
        extracted_text=result.text or None,
        lang=result.lang,
        error=error,
    )
    return result.text or note.raw_text or ""


async def process_note_async(
    note_id: int,
    *,
    session_factory: async_sessionmaker,
    embedding_client: HttpEmbeddingClient,
    extractors: dict[str, Extractor] | None = None,
    llm_queue: Queue | None = None,
    alerts: AlertNotifier | None = None,
    sender: Sender | None = None,
    llm_client: LLMClient | None = None,
    llm_timeout_s: float = 120.0,
    translate_client: HttpTranslateClient | None = None,
) -> None:
    # Empty, not a module-level default: a 'text' note never consults this
    # (see _extract_and_get_index_text), and every other source_type is
    # expected to pass its own — process_note() below always does, built
    # from settings via _get_extractors().
    extractors = extractors if extractors is not None else {}
    alerts = alerts if alerts is not None else NullNotifier()
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
        # Captured before any rollback below expires the ORM object —
        # reading note.source_type/source_url afterward would trigger an
        # implicit (and here, unawaited) lazy-load.
        source_type = note.source_type
        source_url = note.source_url
        user_id = note.user_id
        chat_id = note.chat_id

        started = time.perf_counter()
        # /debug's per-stage breakdown (render_debug_processing_done) — a
        # stage that's skipped this run (translate, when disabled or never
        # attempted) simply gets no entry, see render_stage_timings.
        stage_timings: dict[str, float] = {}
        try:
            stage_started = time.perf_counter()
            text = await _extract_and_get_index_text(note, note_repo, extractors)
            stage_timings["extract"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            chunks = chunk(text)
            stage_timings["chunk"] = time.perf_counter() - stage_started
            if not chunks:
                # Only reached with nothing left to index at all — not even
                # raw_text/the URL. This IS a real failure: the note is
                # physically unfindable — see "Деградация".
                raise ValueError("note has no text to index, even after degradation")

            texts_to_embed = [c.text for c in chunks]
            if translate_client is not None:
                # 04-search.md, "Перевод перед эмбеддингом": embed the
                # translated text so a Hebrew note is findable by a Russian
                # query and vice versa, but `chunks` (chunk.text, used for
                # display — render_search_card etc.) stays the original,
                # untranslated text. Best-effort: a down/slow translate
                # service must not fail note processing, only fall back to
                # the old same-language-only search behavior for this note.
                stage_started = time.perf_counter()
                try:
                    texts_to_embed = await translate_client.translate_passages(texts_to_embed)
                except TranslateServiceError as exc:
                    log.warning(
                        "process_note: note_id=%s translate failed, embedding original text: %s",
                        note_id,
                        exc,
                    )
                stage_timings["translate"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            vectors = await embedding_client.embed_passages(texts_to_embed)
            stage_timings["embed"] = time.perf_counter() - stage_started
            new_chunks = ChunkRepository.from_domain_chunks(chunks, vectors)

            stage_started = time.perf_counter()
            await chunk_repo.replace_chunks(
                note_id, new_chunks, embedding_model=embedding_client.model_name
            )
            await note_repo.mark_done(note_id)
            await session.commit()
            stage_timings["save"] = time.perf_counter() - stage_started
            log.info("process_note: note_id=%s done, chunks=%d", note_id, len(new_chunks))

            if sender is not None and source_type != "table_event":
                # /debug (03-ingest.md, "Debug: время обработки") — opt-in,
                # so this check is a cheap SELECT for every note whose owner
                # never turned it on, not a reason to skip the check.
                # table_event is excluded: /events_table creates many notes
                # in one confirm, each processed as its own independent
                # process_note job, so a per-note message here would flood
                # the chat with one "Обработка завершена" per event instead
                # of the single "Сохранил N заметок" confirm already sends.
                if await UserSettingsRepository(session).is_debug_enabled(user_id):
                    elapsed_s = time.perf_counter() - started
                    summary = None
                    if llm_client is not None:
                        try:
                            summary = await generate_summary(
                                llm_client, text, timeout_s=llm_timeout_s
                            )
                        except LLMServiceError as exc:
                            # Best-effort: this is a debug convenience, not
                            # the note's own processing — a failed/slow LLM
                            # call here must not turn into a failed note, or
                            # even a missing notification. Falls back to the
                            # raw text preview below.
                            log.warning(
                                "process_note: note_id=%s debug summary failed: %s",
                                note_id,
                                exc,
                            )
                    await sender.send(
                        chat_id,
                        render_debug_processing_done(
                            elapsed_s=elapsed_s,
                            summary=summary,
                            indexed_text=text,
                            stage_timings=stage_timings,
                        ),
                    )

            if llm_queue is not None and source_type != "table_event":
                # 03-ingest.md's sequence diagram: enrichment is queued
                # right after status=done, never blocking it — a note is
                # fully findable before enrich_note ever runs. Deterministic
                # job_id (ADR-8), same as process_note's own enqueue.
                #
                # table_event is the one exception: "/events_table" already
                # set tags (event type + the table's own topic_tags) via
                # NoteRepository.set_tags_and_skip_enrich, from the same
                # extraction call that produced the note text — enrich_note's
                # own generate_tags would overwrite them with a guess made
                # from one short line, with no knowledge of the table's
                # broader topic.
                llm_queue.enqueue(enrich_note, note_id, job_id=f"enrich_note-{note_id}")

        except EmbeddingServiceError as exc:
            await session.rollback()
            async with session_factory() as failure_session:
                await NoteRepository(failure_session).mark_failed(note_id, str(exc))
                await failure_session.commit()
            NOTES_STATUS_FAILED_TOTAL.labels(source_type=source_type).inc()
            log.warning(
                "process_note: note_id=%s failed (retryable=%s): %s",
                note_id,
                exc.retryable,
                exc,
            )
            await alerts.failed(
                note_id=note_id, source_type=source_type, source_url=source_url, error=str(exc)
            )
            if sender is not None:
                await sender.send(chat_id, render_processing_failed())
        except Exception as exc:  # noqa: BLE001 — logged and recorded, not swallowed
            await session.rollback()
            async with session_factory() as failure_session:
                await NoteRepository(failure_session).mark_failed(note_id, str(exc))
                await failure_session.commit()
            NOTES_STATUS_FAILED_TOTAL.labels(source_type=source_type).inc()
            log.exception("process_note: note_id=%s failed unexpectedly", note_id)
            await alerts.failed(
                note_id=note_id, source_type=source_type, source_url=source_url, error=str(exc)
            )
            if sender is not None:
                await sender.send(chat_id, render_processing_failed())
        finally:
            # 06-deployment.md, "Время обработки по source_type" —
            # separately shows the real cost of Whisper.
            PROCESS_NOTE_DURATION_SECONDS.labels(source_type=source_type).observe(
                time.perf_counter() - started
            )


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
            llm_queue=_get_llm_queue(settings),
            alerts=_get_alerts(settings),
            sender=TelegramSender(settings.telegram_bot_token),
            llm_client=_get_llm_client(settings),
            llm_timeout_s=settings.llm_enrich_timeout_s,
            translate_client=_get_translate_client(settings),
        )
    )


def _get_translate_client(settings: Settings) -> HttpTranslateClient | None:
    """None when TRANSLATE_ENABLED=false — process_note_async's own
    `if translate_client is not None` then embeds the original text
    untranslated, same as before this feature existed."""
    if not settings.translate_enabled:
        return None
    return HttpTranslateClient(
        settings.translate_url,
        target_lang=settings.translate_target_lang,
        timeout=settings.translate_timeout_s,
    )


def _get_llm_client(settings: Settings) -> LLMClient:
    """NullLLMClient when LLM_ENABLED=false — see 05-contracts.md. Built
    fresh per call (unlike the cached engine/extractors) since it's cheap
    and the flag could plausibly change between deploys without a code
    change; no reason to bake the choice in at first import."""
    if not settings.llm_enabled:
        return NullLLMClient()
    return ProxyAILLMClient(settings.proxy_ai_url)


async def enrich_note_async(
    note_id: int,
    *,
    session_factory: async_sessionmaker,
    llm_client: LLMClient,
    embedding_client: HttpEmbeddingClient,
    llm_enabled: bool,
    timeout_s: float,
    dupe_candidate_k: int = 50,
) -> None:
    """Never touches status/extracted_text/chunks (see
    NoteRepository.set_enrichment) — a note is already fully findable
    before this ever runs; this only adds title/tags/summary/structured
    fields and, for a `map` note, place details. See 03-ingest.md, "Шаг 6".
    """
    async with session_factory() as session:
        note_repo = NoteRepository(session)
        note = await note_repo.get(note_id)
        if note is None or note.status != "done":
            # Not indexed yet (still processing/failed) — nothing to
            # enrich; re-enqueued once process_note actually finishes.
            log.info("enrich_note: note_id=%s not done yet, skipping", note_id)
            return

        if not llm_enabled:
            await note_repo.mark_enrich_skipped(note_id)
            await session.commit()
            return

        await note_repo.mark_enrich_processing(note_id)
        await session.commit()

    async with session_factory() as session:
        note_repo = NoteRepository(session)
        chunk_repo = ChunkRepository(session)
        note = await note_repo.get(note_id)
        assert note is not None

        text = note.extracted_text or note.raw_text or ""

        try:
            title = await generate_title(llm_client, text, timeout_s=timeout_s)
            tags = await generate_tags(llm_client, text, timeout_s=timeout_s)
            summary = await generate_summary(llm_client, text, timeout_s=timeout_s)

            structured: dict = {}
            if note.source_type == "map":
                structured = await generate_place(llm_client, text, timeout_s=timeout_s)
            elif note.source_type in ("youtube", "instagram"):
                places = await generate_places(llm_client, text, timeout_s=timeout_s)
                if places:
                    structured = {"places": places}

            own_embedding = await chunk_repo.get_first_chunk_embedding(note_id)
            if own_embedding is not None:
                hits = await search_notes(
                    session,
                    query_vector=own_embedding,
                    acl_predicate=and_(Note.user_id == note.user_id, Note.id != note_id),
                    active_model=embedding_client.model_name,
                    candidate_k=dupe_candidate_k,
                    limit=5,
                    offset=0,
                )
                candidates = [(hit.note_id, hit.chunk_text) for hit in hits]
                duplicate_id = await find_duplicate(
                    llm_client, new_text=text, candidates=candidates, timeout_s=timeout_s
                )
                if duplicate_id is not None:
                    # Surfacing this to the user needs the same worker->bot
                    # notification channel noted as a gap in bot/handlers.py
                    # — not built yet. Persisted here so it isn't lost.
                    structured = {**structured, "possible_duplicate_of": duplicate_id}
                    log.info(
                        "enrich_note: note_id=%s possible duplicate of note_id=%s",
                        note_id,
                        duplicate_id,
                    )

            await note_repo.set_enrichment(
                note_id, title=title, summary=summary, tags=tags, structured=structured
            )
            await session.commit()
            log.info("enrich_note: note_id=%s done", note_id)

        except LLMServiceError as exc:
            await session.rollback()
            async with session_factory() as failure_session:
                await NoteRepository(failure_session).mark_enrich_failed(note_id, str(exc))
                await failure_session.commit()
            NOTES_ENRICH_FAILED_TOTAL.inc()
            log.warning(
                "enrich_note: note_id=%s failed (quota_exhausted=%s, retryable=%s): %s",
                note_id,
                exc.is_quota_exhausted,
                exc.retryable,
                exc,
            )
        except Exception as exc:  # noqa: BLE001 — logged and recorded, not swallowed
            await session.rollback()
            async with session_factory() as failure_session:
                await NoteRepository(failure_session).mark_enrich_failed(note_id, str(exc))
                await failure_session.commit()
            NOTES_ENRICH_FAILED_TOTAL.inc()
            log.exception("enrich_note: note_id=%s failed unexpectedly", note_id)


def enrich_note(note_id: int) -> None:
    """The RQ-registered entrypoint — `llm` queue, see 06-deployment.md.
    Listened to by notes-worker-fast (after `fast`, per that doc: "fast и
    llm в таком порядке — пользовательские задачи имеют приоритет над
    обогащением"), not notes-worker-heavy — enrichment is cheap HTTP calls
    to ai-proxy, not CPU-bound transcription.
    """
    settings = get_settings()
    session_factory = _get_session_factory(settings)
    embedding_client = HttpEmbeddingClient(
        settings.embeddings_url,
        model_name=settings.embedding_model_name,
        dim=settings.embedding_dim,
    )
    asyncio.run(
        enrich_note_async(
            note_id,
            session_factory=session_factory,
            llm_client=_get_llm_client(settings),
            embedding_client=embedding_client,
            llm_enabled=settings.llm_enabled,
            timeout_s=settings.llm_enrich_timeout_s,
        )
    )


async def smart_answer_async(
    *,
    user_id: int,
    chat_id: int,
    is_group_chat: bool,
    query_text: str,
    session_factory: async_sessionmaker,
    embedding_client: HttpEmbeddingClient,
    llm_client: LLMClient,
    llm_enabled: bool,
    timeout_s: float,
    sender: Sender,
    top_n: int = 10,
    max_distance: float | None = None,
    translate_client: HttpTranslateClient | None = None,
) -> None:
    """`smart_answer` — see 04-search.md, "/smart_search": the synthesis of
    an already-queried search whose raw hits were never shown (the handler
    sends only a pending marker before this runs). A failure here used to
    just log and return, on the reasoning that the user already had the raw
    results and losing only the synthesis was no big deal — with no raw
    results shown any more, that would leave the pending marker as a silent
    dead end, so every failure path below sends render_smart_answer_failed()
    instead. "No matching notes" is the one exception: the handler already
    checked hits were non-empty moments before enqueuing this, so a race
    that empties it between the two searches is the only way to reach it,
    and logging is enough for something that rare.

    ACL is enforced here, in the SQL that builds `hits`, before any note
    text reaches the LLM prompt — see 04-search.md: "единственный способ,
    которым приватная заметка может покинуть систему, - попадание в
    контекст LLM-запроса".
    """
    if not llm_enabled:
        log.info("smart_answer: user_id=%s LLM disabled, skipping", user_id)
        return

    # /debug (03-ingest.md's flag, reused — see render_smart_answer_debug):
    # a per-stage breakdown of where this run's time went, plus token usage
    # when ai-proxy's response carries it. A stage that never ran (translate,
    # when disabled) has no entry — see render_stage_timings.
    started = time.perf_counter()
    stage_timings: dict[str, float] = {}

    async with session_factory() as session:
        search_mode = "all"
        debug_enabled = False
        if not is_group_chat:
            settings_row = await UserSettingsRepository(session).get_or_create(user_id)
            search_mode = settings_row.search_mode
            debug_enabled = await UserSettingsRepository(session).is_debug_enabled(user_id)

        embed_text = query_text
        if translate_client is not None:
            # 04-search.md, "Перевод перед эмбеддингом" — same reasoning as
            # run_search's own translate-before-embed: without it, this
            # query vector never matches a note written in a different
            # language, since this is a separate embed_query call from
            # run_search's (smart_answer_async builds its own LLM context
            # independently of the handler's earlier "any hits at all?"
            # check). Best-effort: a down translate service falls back to
            # embedding the original query, not a failed answer.
            stage_started = time.perf_counter()
            try:
                embed_text = await translate_client.translate_query(query_text)
            except TranslateServiceError as exc:
                log.warning("smart_answer: user_id=%s translate failed: %s", user_id, exc)
            stage_timings["translate"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        try:
            query_vector = await embedding_client.embed_query(embed_text)
        except EmbeddingServiceError as exc:
            log.warning("smart_answer: user_id=%s embedding failed: %s", user_id, exc)
            await sender.send(chat_id, render_smart_answer_failed())
            return
        stage_timings["embed"] = time.perf_counter() - stage_started

        predicate = visibility_predicate(
            user_id=user_id, chat_id=chat_id, is_group_chat=is_group_chat, search_mode=search_mode
        )
        stage_started = time.perf_counter()
        hits = await search_notes(
            session,
            query_vector=query_vector,
            acl_predicate=predicate,
            active_model=embedding_client.model_name,
            candidate_k=200,
            limit=top_n,
            offset=0,
            max_distance=max_distance,
        )
        stage_timings["search"] = time.perf_counter() - stage_started

    if not hits:
        log.info("smart_answer: user_id=%s no matching notes, skipping", user_id)
        return

    context_block = "\n\n".join(
        f"[{h.note_id}] {h.title or ''}\n{h.chunk_text}".strip() for h in hits
    )
    user_prompt = f"Вопрос: {query_text}\n\nЗаметки:\n{context_block}"

    stage_started = time.perf_counter()
    try:
        result = await llm_client.complete(
            system=load_prompt("rag_answer"), user=user_prompt, timeout_s=timeout_s
        )
    except LLMServiceError as exc:
        log.warning(
            "smart_answer: user_id=%s ai-proxy failed (quota_exhausted=%s): %s",
            user_id,
            exc.is_quota_exhausted,
            exc,
        )
        await sender.send(chat_id, render_smart_answer_failed())
        return
    stage_timings["llm"] = time.perf_counter() - stage_started

    if not result.text:
        log.info("smart_answer: user_id=%s empty synthesis, skipping", user_id)
        await sender.send(chat_id, render_smart_answer_failed())
        return

    # render_places returns HTML (place names as <a href>, see its own
    # docstring) — this whole message is sent with parse_mode="HTML" below,
    # so every other dynamic piece (LLM-generated answer text; note
    # title/chunk_text/source_url, all arbitrary internet content) must be
    # escaped too, or a stray '<'/'&' anywhere in them would break parsing.
    source_lines = []
    for h in hits:
        line = f"[{h.note_id}] {_esc(h.title or h.chunk_text[:40])}"
        if h.source_url:
            line += f" — {_esc(h.source_url)}"
        source_lines.append(line)
        # /smart_search shows no raw cards any more (04-search.md) — this
        # is the only place a video note's extracted places (render.py's
        # render_places, from generate_places) ever reach the user.
        source_lines.extend(f"  {place_line}" for place_line in render_places(h.structured))
    sources = "\n".join(source_lines)
    answer = f"{_esc(result.text)}\n\nИсточники:\n{sources}"
    await sender.send(chat_id, answer, parse_mode="HTML")

    if debug_enabled:
        tokens_in, tokens_out = extract_token_usage(result.usage)
        await sender.send(
            chat_id,
            render_smart_answer_debug(
                elapsed_s=time.perf_counter() - started,
                stage_timings=stage_timings,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            ),
        )


def smart_answer(user_id: int, chat_id: int, is_group_chat: bool, query_text: str) -> None:
    """The RQ-registered entrypoint — `llm` queue, see 06-deployment.md."""
    settings = get_settings()
    session_factory = _get_session_factory(settings)
    embedding_client = HttpEmbeddingClient(
        settings.embeddings_url,
        model_name=settings.embedding_model_name,
        dim=settings.embedding_dim,
    )
    asyncio.run(
        smart_answer_async(
            user_id=user_id,
            chat_id=chat_id,
            is_group_chat=is_group_chat,
            query_text=query_text,
            session_factory=session_factory,
            embedding_client=embedding_client,
            llm_client=_get_llm_client(settings),
            llm_enabled=settings.llm_enabled,
            timeout_s=settings.llm_smart_search_timeout_s,
            sender=TelegramSender(settings.telegram_bot_token),
            max_distance=settings.search_max_distance,
            translate_client=_get_translate_client(settings),
        )
    )


async def parse_events_table_async(
    *,
    chat_id: int,
    user_id: int,
    is_group: bool,
    tab_name: str,
    images: list[ImageInput],
    llm_client: LLMClient,
    llm_enabled: bool,
    timeout_s: float,
    sender: Sender,
    pending_cache: PendingEventsCache,
) -> None:
    """`/events_table` (03-ingest.md) — the vision call itself, on the `llm`
    queue same as `smart_answer`: this is user-facing-but-LLM-heavy, not
    urgent the way `process_note` is. handlers.py buffers the Telegram
    album and downloads the photos; this only does the LLM call and the
    preview/cache step, so a slow ai-proxy round trip never blocks the
    single bot replica's polling loop.
    """
    if not llm_enabled:
        log.info("parse_events_table: chat_id=%s LLM disabled, skipping", chat_id)
        await sender.send(chat_id, render_events_table_disabled())
        return

    table = await extract_events_table(
        llm_client, images=images, tab_name=tab_name, timeout_s=timeout_s
    )
    if not table.events:
        await sender.send(chat_id, render_events_table_empty(tab_name))
        return

    session_id = await pending_cache.create(
        chat_id=chat_id,
        user_id=user_id,
        is_group=is_group,
        tab_name=tab_name,
        topic_tags=table.topic_tags,
        events=table.events,
    )
    # Insertion order, not alphabetical — matches the order events were
    # actually found, which is usually chronological (dates top to bottom).
    type_counts: dict[str, int] = {}
    for event in table.events:
        type_counts[event.tag] = type_counts.get(event.tag, 0) + 1

    await sender.send(
        chat_id,
        render_events_table_preview(
            tab_name=tab_name, topic_tags=table.topic_tags, type_counts=type_counts
        ),
        reply_markup=events_table_confirm_keyboard(session_id),
    )


def parse_events_table(
    *, chat_id: int, user_id: int, is_group: bool, tab_name: str, images: list[ImageInput]
) -> None:
    """The RQ-registered entrypoint — `llm` queue, see 06-deployment.md."""
    settings = get_settings()
    asyncio.run(
        parse_events_table_async(
            chat_id=chat_id,
            user_id=user_id,
            is_group=is_group,
            tab_name=tab_name,
            images=images,
            llm_client=_get_llm_client(settings),
            llm_enabled=settings.llm_enabled,
            timeout_s=settings.events_table_timeout_s,
            sender=TelegramSender(settings.telegram_bot_token),
            pending_cache=_get_pending_events_cache(settings),
        )
    )
