"""Core bot logic — no aiogram Message/CallbackQuery types here, so it is
testable without constructing Telegram objects. `bot/handlers.py` is the
thin aiogram-specific adapter on top of this.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from rq import Queue
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.bot.render import RenderableHit, render_search_debug_empty
from notes_bot.clients.conflicts_cache import ConflictsCache, PendingConflict
from notes_bot.clients.embeddings import HttpEmbeddingClient
from notes_bot.clients.pending_events_cache import PendingEventsCache
from notes_bot.clients.search_cache import SearchSessionCache
from notes_bot.clients.translate import HttpTranslateClient, TranslateServiceError
from notes_bot.config import Settings
from notes_bot.db.models import Note
from notes_bot.db.repositories import ChatSettingsRepository, NoteRepository, UserSettingsRepository
from notes_bot.db.search import search_notes
from notes_bot.domain.acl import visibility_predicate
from notes_bot.domain.classify import classify_text_message
from notes_bot.queue.queues import enqueue_process_note, enqueue_smart_answer

log = logging.getLogger(__name__)

# instagram and voice go to `heavy` — see 03-ingest.md, "Шаг 2. Приём в
# боте": "Выбрать очередь: instagram и voice → heavy, всё остальное → fast".
_HEAVY_SOURCE_TYPES = {"instagram", "voice"}


@dataclass(frozen=True)
class Deps:
    session_factory: async_sessionmaker
    embedding_client: HttpEmbeddingClient
    search_cache: SearchSessionCache
    fast_queue: Queue
    heavy_queue: Queue
    llm_queue: Queue
    settings: Settings
    # Needed to recognize "@botname" mentions in group messages — see
    # domain/group_capture.py. Resolved once via getMe at startup
    # (notes_bot.health.mark_ready_after_get_me already calls it; cli/bot.py
    # reuses that result rather than calling getMe twice).
    bot_username: str = ""
    # None when TRANSLATE_ENABLED=false — run_search then embeds query_text
    # as-is, same as before notes-translate existed (04-search.md, "Перевод
    # перед эмбеддингом").
    translate_client: HttpTranslateClient | None = None
    # /events_table (03-ingest.md) — holds a parsed table between the
    # preview message (sent by the worker, queue/tasks.py's
    # parse_events_table) and the Сохранить/Отмена tap, which runs here in
    # the bot process. None only in tests that don't exercise this command.
    pending_events_cache: PendingEventsCache | None = None
    # Same command's dedup step (03-ingest.md, "/events_table" - "уже
    # существует") — holds the "same date range + type, different text"
    # cases confirm_events_table couldn't resolve on its own, between the
    # per-conflict message and the tap that answers it.
    conflicts_cache: ConflictsCache | None = None


@dataclass(frozen=True)
class SaveNoteResult:
    created: bool
    note_id: int | None
    visibility: str | None
    source_type: str | None = None


async def save_note(
    deps: Deps, *, user_id: int, chat_id: int, is_group: bool, tg_message_id: int, text: str
) -> SaveNoteResult:
    """Fail-closed privacy (ADR-5): a private-chat note is saved with the
    user's default visibility immediately, before any button is pressed. A
    group note gets no visibility at all — group membership is its scope.

    Source type is classified from the message text (03-ingest.md, "Шаг 1")
    — a URL routes to page/youtube/map/instagram, plain text stays 'text'.
    """
    classification = classify_text_message(text)

    async with deps.session_factory() as session:
        visibility: str | None = None
        if not is_group:
            settings_row = await UserSettingsRepository(session).get_or_create(user_id)
            visibility = settings_row.default_visibility

        note = await NoteRepository(session).create_note(
            user_id=user_id,
            chat_id=chat_id,
            is_group=is_group,
            tg_message_id=tg_message_id,
            source_type=classification.source_type,
            source_url=classification.source_url,
            raw_text=text,
            visibility=visibility,
        )
        await session.commit()

    if note is None:
        # ADR-8: a retried Telegram delivery for a message already saved.
        return SaveNoteResult(created=False, note_id=None, visibility=None)

    is_heavy = classification.source_type in _HEAVY_SOURCE_TYPES
    target_queue = deps.heavy_queue if is_heavy else deps.fast_queue
    enqueue_process_note(
        target_queue,
        note.id,
        job_timeout=deps.settings.heavy_job_timeout_s if is_heavy else None,
    )

    return SaveNoteResult(
        created=True,
        note_id=note.id,
        visibility=visibility,
        source_type=classification.source_type,
    )


async def save_voice_note(
    deps: Deps,
    *,
    user_id: int,
    chat_id: int,
    is_group: bool,
    tg_message_id: int,
    file_id: str,
    caption: str | None,
) -> SaveNoteResult:
    """A voice message is a Telegram message *type*, not a URL to classify
    — the bot layer already knows it's `source_type='voice'` before this
    runs. `file_id` goes in source_url (see VoiceExtractor's docstring for
    why); `caption`, if any, is what gets edited/re-chunked on a later text
    edit — the transcript itself never is (03-ingest.md: editing is
    text-only)."""
    async with deps.session_factory() as session:
        visibility: str | None = None
        if not is_group:
            settings_row = await UserSettingsRepository(session).get_or_create(user_id)
            visibility = settings_row.default_visibility

        note = await NoteRepository(session).create_note(
            user_id=user_id,
            chat_id=chat_id,
            is_group=is_group,
            tg_message_id=tg_message_id,
            source_type="voice",
            source_url=file_id,
            raw_text=caption,
            visibility=visibility,
        )
        await session.commit()

    if note is None:
        return SaveNoteResult(created=False, note_id=None, visibility=None)

    enqueue_process_note(deps.heavy_queue, note.id, job_timeout=deps.settings.heavy_job_timeout_s)
    return SaveNoteResult(created=True, note_id=note.id, visibility=visibility, source_type="voice")


async def toggle_privacy(deps: Deps, *, note_id: int, user_id: int) -> str | None:
    async with deps.session_factory() as session:
        new_visibility = await NoteRepository(session).toggle_visibility(note_id, user_id)
        await session.commit()
    return new_visibility


_NEAR_MISS_COUNT = 3


@dataclass(frozen=True)
class SearchPageResult:
    hits: list[RenderableHit]
    has_more: bool
    session_id: str | None
    # Only ever set alongside `hits == []` — see render_search_debug_empty.
    # Built from data already fetched for the (empty) result itself, not a
    # second query, and only when the caller has /debug on.
    debug_info: str | None = None


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_STEM_LEN = 3
_MAX_STEM_LEN = 6


def _stems_match(subject: str, token: str) -> bool:
    """A shared-prefix check, not real morphology - good enough to tell
    "математика"/"mathematics" apart from "физика"/"physics" without a
    real stemmer, and forgiving of translate_client's own inconsistent
    endings ("math" vs "mathematics" for the same subject) by capping at
    `_MAX_STEM_LEN` rather than requiring one string to literally contain
    the other."""
    k = min(len(subject), len(token), _MAX_STEM_LEN)
    return k >= _MIN_STEM_LEN and subject[:k] == token[:k]


def _filter_table_event_hits_by_subject(hits: list, canonical_query: str) -> list:
    """/events_table (03-ingest.md) notes are short, templated exam-cluster
    listings ("экзамен - физика/искусство/кино/..."), so a plain vector
    search for "экзамен по математике" pulls in every exam note in the
    corpus about equally well - "экзамен" dominates the embedding, the
    specific subject barely moves it. When the query names a subject that
    some retrieved table_event hit's `structured["subjects"]` (set at
    parse time, in the same canonical language as `canonical_query` - see
    confirm_events_table/run_search) actually has but others don't, keep
    only the ones that do; every other hit, and every other source_type,
    is untouched.

    `canonical_query` must already be in that same canonical language
    (run_search passes `embed_text`, not the caller's raw `query_text`) -
    language-agnostic the same way the vector search itself is, not
    "matches only if you happen to type the note's own language". Never
    removes every table_event hit: if none of them name a subject the
    query mentions, there's nothing to discriminate on, so all are kept
    (better a broad answer than none)."""
    query_tokens = _WORD_RE.findall(canonical_query.lower())
    table_event_hits = [h for h in hits if h.source_type == "table_event"]
    if len(table_event_hits) < 2 or not query_tokens:
        return hits

    def subjects_of(hit) -> list[str]:
        return (hit.structured or {}).get("subjects") or []

    def matches(hit) -> bool:
        subjects = [s.lower() for s in subjects_of(hit) if s]
        return any(_stems_match(subject, token) for subject in subjects for token in query_tokens)

    matching_ids = {h.note_id for h in table_event_hits if matches(h)}
    if not matching_ids or len(matching_ids) == len(table_event_hits):
        return hits

    return [h for h in hits if h.source_type != "table_event" or h.note_id in matching_ids]


async def run_search(
    deps: Deps, *, user_id: int, chat_id: int, is_group_chat: bool, query_text: str
) -> SearchPageResult:
    page_size = deps.settings.search_page_size

    async with deps.session_factory() as session:
        search_mode = "all"
        if not is_group_chat:
            settings_row = await UserSettingsRepository(session).get_or_create(user_id)
            search_mode = settings_row.search_mode

        embed_text = query_text
        if deps.translate_client is not None:
            # 04-search.md, "Перевод перед эмбеддингом" — embed the
            # translated query so it lands near notes translated the same
            # way at index time (see queue/tasks.py's process_note_async),
            # regardless of which language either was written in. Rendered
            # hits (chunk_text/title below) are untouched — only the vector
            # used to find them changes. Best-effort: a down/slow translate
            # service falls back to the old same-language-only search for
            # this query, not a failed search.
            try:
                embed_text = await deps.translate_client.translate_query(query_text)
            except TranslateServiceError as exc:
                log.warning("run_search: translate failed, using original query: %s", exc)

        query_vector = await deps.embedding_client.embed_query(embed_text)
        predicate = visibility_predicate(
            user_id=user_id, chat_id=chat_id, is_group_chat=is_group_chat, search_mode=search_mode
        )
        # Fetched unfiltered and thresholded here in Python, not in SQL:
        # when nothing clears SEARCH_MAX_DISTANCE, a /debug caller still
        # wants to see what the closest candidates actually were, and that
        # data no longer exists once max_distance drops it at the SQL layer.
        max_distance = deps.settings.search_max_distance
        raw_hits = await search_notes(
            session,
            query_vector=query_vector,
            acl_predicate=predicate,
            active_model=deps.embedding_client.model_name,
            candidate_k=deps.settings.search_candidate_k,
            limit=50,
            offset=0,
            max_distance=None,
        )
        hits = (
            raw_hits
            if max_distance is None
            else [h for h in raw_hits if h.distance <= max_distance]
        )
        # embed_text, not query_text: structured["subjects"] is stored in
        # the same canonical translate_target_lang as embed_text (see
        # confirm_events_table), not in whichever language the caller
        # happened to type - language-agnostic the same way the vector
        # search itself already is, not just "matches if you type Russian".
        hits = _filter_table_event_hits_by_subject(hits, embed_text)

        if not hits:
            debug_info = None
            if await UserSettingsRepository(session).is_debug_enabled(user_id):
                near_misses = [
                    (h.title or h.chunk_text[:40], h.distance) for h in raw_hits[:_NEAR_MISS_COUNT]
                ]
                debug_info = render_search_debug_empty(near_misses, max_distance=max_distance)
            return SearchPageResult(hits=[], has_more=False, session_id=None, debug_info=debug_info)

    session_id = await deps.search_cache.create(
        user_id=user_id,
        query=query_text,
        mode=search_mode,
        scope_chat_id=chat_id if is_group_chat else None,
        note_ids=[h.note_id for h in hits],
        fragments=[h.chunk_text for h in hits],
    )
    # The cache's offset counts items already shown; this call serves the
    # first page immediately, so the next "show more" must start after it.
    await deps.search_cache.set_offset(user_id, session_id, page_size)

    page = hits[:page_size]
    return SearchPageResult(
        hits=[_to_renderable(h) for h in page],
        has_more=len(hits) > page_size,
        session_id=session_id,
    )


async def smart_search(
    deps: Deps, *, user_id: int, chat_id: int, is_group_chat: bool, query_text: str
) -> SearchPageResult:
    """`/smart_search` — see 04-search.md: runs the exact same synchronous
    search as `/search` (so it is never worse), then queues `smart_answer`
    in `llm` for the synthesized, sourced answer that follows as a
    separate message. No query rewriting (04-search.md explains why: it
    would cost a second CLI-provider round trip for a saving that doesn't
    pay for the extra latency).

    Silently skips queuing when LLM_ENABLED=false — see
    05-contracts.md/NullLLMClient: the smart answer just never arrives,
    same as any other ai-proxy failure."""
    page = await run_search(
        deps, user_id=user_id, chat_id=chat_id, is_group_chat=is_group_chat, query_text=query_text
    )
    if deps.settings.llm_enabled:
        enqueue_smart_answer(
            deps.llm_queue,
            user_id=user_id,
            chat_id=chat_id,
            is_group_chat=is_group_chat,
            query_text=query_text,
        )
    return page


@dataclass(frozen=True)
class ShowMoreResult:
    hits: list[RenderableHit]
    has_more: bool
    expired: bool


async def show_more(deps: Deps, *, user_id: int, session_id: str) -> ShowMoreResult:
    page_size = deps.settings.search_page_size

    cached = await deps.search_cache.get(user_id, session_id)
    if cached is None:
        return ShowMoreResult(hits=[], has_more=False, expired=True)

    id_slice = cached.note_ids[cached.offset : cached.offset + page_size]
    fragment_slice = cached.fragments[cached.offset : cached.offset + page_size]
    if not id_slice:
        return ShowMoreResult(hits=[], has_more=False, expired=False)

    is_group_chat = cached.scope_chat_id is not None
    predicate = visibility_predicate(
        user_id=user_id,
        chat_id=cached.scope_chat_id if is_group_chat else user_id,
        is_group_chat=is_group_chat,
        search_mode=cached.mode,
    )

    async with deps.session_factory() as session:
        visible = await NoteRepository(session).get_visible_by_ids(
            id_slice, acl_predicate=predicate
        )

    hits = [
        _to_renderable_from_note(visible[note_id], fragment)
        for note_id, fragment in zip(id_slice, fragment_slice, strict=True)
        if note_id in visible  # deleted or no-longer-visible between pages
    ]

    new_offset = cached.offset + page_size
    await deps.search_cache.set_offset(user_id, session_id, new_offset)

    return ShowMoreResult(hits=hits, has_more=new_offset < len(cached.note_ids), expired=False)


async def show_detail(
    deps: Deps, *, user_id: int, session_id: str, note_id: int
) -> RenderableHit | None:
    """ "Подробнее" under a /search summary card (render.render_search_summary_card)
    — expands to the full render_search_card for one hit. Re-checks ACL at
    click time rather than trusting the original search's snapshot, same
    reasoning as show_more: visibility can change between the search and
    the click. None covers every reason not to show it — expired session,
    a note_id that was never part of it (tampered callback_data), or one
    that is no longer visible — the caller doesn't need to tell those apart."""
    cached = await deps.search_cache.get(user_id, session_id)
    if cached is None:
        return None
    try:
        index = cached.note_ids.index(note_id)
    except ValueError:
        return None

    is_group_chat = cached.scope_chat_id is not None
    predicate = visibility_predicate(
        user_id=user_id,
        chat_id=cached.scope_chat_id if is_group_chat else user_id,
        is_group_chat=is_group_chat,
        search_mode=cached.mode,
    )

    async with deps.session_factory() as session:
        visible = await NoteRepository(session).get_visible_by_ids(
            [note_id], acl_predicate=predicate
        )

    note = visible.get(note_id)
    if note is None:
        return None
    return _to_renderable_from_note(note, cached.fragments[index])


def _to_renderable(hit) -> RenderableHit:
    return RenderableHit(
        note_id=hit.note_id,
        title=hit.title,
        source_type=hit.source_type,
        source_url=hit.source_url,
        tags=hit.tags,
        chunk_text=hit.chunk_text,
        # Delete button rendering is M6 scope; ownership isn't surfaced yet.
        is_owner=False,
        structured=hit.structured,
        summary=hit.summary,
    )


def _to_renderable_from_note(note, chunk_text: str) -> RenderableHit:
    return RenderableHit(
        note_id=note.id,
        title=note.title,
        source_type=note.source_type,
        source_url=note.source_url,
        tags=note.tags,
        chunk_text=chunk_text,
        is_owner=False,
        structured=note.structured,
        summary=note.summary,
    )


def _to_renderable_own(note) -> RenderableHit:
    """/list and /trash in a private chat only ever show the caller's own
    notes — unlike a search hit, ownership here is a given, not something
    to check."""
    return _note_to_renderable(note, is_owner=True)


def _note_to_renderable(note, *, is_owner: bool) -> RenderableHit:
    """Distinct from `_to_renderable` (search hits, above) — a plain
    `Note` row from /list-style pagination, not a `SearchHit`."""
    return RenderableHit(
        note_id=note.id,
        title=note.title,
        source_type=note.source_type,
        source_url=note.source_url,
        tags=note.tags,
        chunk_text=note.extracted_text or note.raw_text or "",
        is_owner=is_owner,
        structured=note.structured,
    )


@dataclass(frozen=True)
class NotesPage:
    hits: list[RenderableHit]
    has_more: bool
    next_offset: int


async def list_notes(deps: Deps, *, user_id: int, offset: int) -> NotesPage:
    """`/list` — plain SQL pagination by `created_at`, not the search
    session cache: there is no ANN ranking here to go stale between pages
    (04-search.md)."""
    page_size = deps.settings.search_page_size
    async with deps.session_factory() as session:
        # +1 to learn whether another page exists, same trick as search.
        notes = await NoteRepository(session).list_own(user_id, limit=page_size + 1, offset=offset)
    has_more = len(notes) > page_size
    page = notes[:page_size]
    return NotesPage(
        hits=[_to_renderable_own(n) for n in page],
        has_more=has_more,
        next_offset=offset + page_size,
    )


async def list_group_notes(
    deps: Deps, *, chat_id: int, viewer_user_id: int, offset: int
) -> NotesPage:
    """`/list` in a group (ADR-10) — every note captured in *this room*,
    not just the caller's own. `is_owner` is computed per note (viewer vs.
    the note's actual saver) so `_send_notes_page` only renders a delete
    button on notes the caller can actually delete — see
    `NoteRepository.list_group`'s own docstring for why this isn't
    `list_notes` with an extra filter."""
    page_size = deps.settings.search_page_size
    async with deps.session_factory() as session:
        notes = await NoteRepository(session).list_group(
            chat_id, limit=page_size + 1, offset=offset
        )
    has_more = len(notes) > page_size
    page = notes[:page_size]
    return NotesPage(
        hits=[_note_to_renderable(n, is_owner=n.user_id == viewer_user_id) for n in page],
        has_more=has_more,
        next_offset=offset + page_size,
    )


async def list_trash(deps: Deps, *, user_id: int, offset: int) -> NotesPage:
    page_size = deps.settings.search_page_size
    async with deps.session_factory() as session:
        notes = await NoteRepository(session).list_own_deleted(
            user_id, limit=page_size + 1, offset=offset
        )
    has_more = len(notes) > page_size
    page = notes[:page_size]
    return NotesPage(
        hits=[_to_renderable_own(n) for n in page],
        has_more=has_more,
        next_offset=offset + page_size,
    )


async def delete_note(deps: Deps, *, note_id: int, user_id: int) -> bool:
    async with deps.session_factory() as session:
        deleted = await NoteRepository(session).soft_delete(note_id, user_id)
        await session.commit()
    return deleted


async def restore_note(deps: Deps, *, note_id: int, user_id: int) -> bool:
    async with deps.session_factory() as session:
        restored = await NoteRepository(session).restore(note_id, user_id)
        await session.commit()
    return restored


async def edit_note_text(deps: Deps, *, note_id: int, user_id: int, new_text: str) -> bool:
    """03-ingest.md, "Редактирование": text-only, full re-chunk/re-embed.
    The note temporarily leaves search results (`status='pending'`) rather
    than show text that no longer matches its vectors."""
    async with deps.session_factory() as session:
        edited = await NoteRepository(session).edit_text(note_id, user_id, new_text)
        await session.commit()
    if edited:
        enqueue_process_note(deps.fast_queue, note_id)
    return edited


async def edit_note_by_message(
    deps: Deps, *, chat_id: int, tg_message_id: int, user_id: int, new_text: str
) -> bool:
    """Triggered by the user editing their original Telegram message
    in-place, per 03-ingest.md: "пользователь присылает новое сообщение как
    замену" — the most natural reading of that is an actual Telegram
    message edit, which needs no new UI and reuses the (chat_id,
    tg_message_id) pair ADR-8's idempotent insert already keys on."""
    async with deps.session_factory() as session:
        note_id = await NoteRepository(session).edit_text_by_message(
            chat_id=chat_id, tg_message_id=tg_message_id, user_id=user_id, new_text=new_text
        )
        await session.commit()
    if note_id is not None:
        enqueue_process_note(deps.fast_queue, note_id)
    return note_id is not None


async def set_group_capture_mode(deps: Deps, *, chat_id: int, mode: str) -> None:
    async with deps.session_factory() as session:
        await ChatSettingsRepository(session).set_capture_mode(chat_id, mode)
        await session.commit()


@dataclass(frozen=True)
class ConflictPreview:
    index: int
    old_text: str
    new_text: str


@dataclass(frozen=True)
class ConfirmEventsTableResult:
    ok: bool
    created: int = 0
    skipped_exact: int = 0
    conflicts_session_id: str | None = None
    conflicts: list[ConflictPreview] = field(default_factory=list)


async def confirm_events_table(
    deps: Deps, *, session_id: str, user_id: int
) -> ConfirmEventsTableResult:
    """ "✅ Сохранить" on the /events_table preview (03-ingest.md) — one note
    per parsed event, tagged with the event's own type (экзамен/праздник/
    мероприятие) plus the table's topic_tags from the same extraction
    call. `ok=False` on an expired/foreign session — expired is normal
    (30-minute TTL, same as SearchSessionCache), foreign never happens
    through the bot's own UI but is still checked, same as show_detail's
    own re-check.

    Dedup ("/events_table" - "уже существует"): each event is matched
    against existing `table_event` notes in the same chat by (date_start,
    date_end, type), never by text - the source table can legitimately
    change between two screenshots of the same tab (it says so itself:
    "הלוח עשוי להשתנות"). An exact text match is a true duplicate, skipped
    silently; a same-key-different-text match is a real conflict, deferred
    to `resolve_events_table_conflict` rather than guessed at here.

    The dedup snapshot is fetched once, *before* this run creates anything
    - two different events sharing a key within the *same* run (e.g. two
    unrelated entries on the same day, both classified the same type) are
    never checked against each other, only against notes that already
    existed before this run started. Every event in the current batch is
    added unconditionally."""
    if deps.pending_events_cache is None:
        return ConfirmEventsTableResult(ok=False)
    pending = await deps.pending_events_cache.get(session_id)
    if pending is None or pending.user_id != user_id:
        return ConfirmEventsTableResult(ok=False)

    async with deps.session_factory() as session:
        note_repo = NoteRepository(session)
        visibility: str | None = None
        if not pending.is_group:
            settings_row = await UserSettingsRepository(session).get_or_create(pending.user_id)
            visibility = settings_row.default_visibility

        existing_by_key: dict[tuple[str, str, str], Note] = {}
        for existing_note in await note_repo.list_table_events(pending.chat_id):
            structured_existing = existing_note.structured or {}
            key = (
                structured_existing.get("date_start"),
                structured_existing.get("date_end"),
                structured_existing.get("type"),
            )
            prior = existing_by_key.get(key)
            if prior is None or existing_note.id > prior.id:
                existing_by_key[key] = existing_note

        # `event.subjects` stays in the table's own language (goes into
        # `tags`, shown to the user) - `structured["subjects"]` needs a
        # canonical-language copy instead, so `_filter_table_event_hits_by_subject`
        # can compare it against `embed_text` regardless of which language
        # either the table or the search query happened to use. One batch
        # translate call for the whole run, not one per event.
        all_subjects = sorted({s for event in pending.extracted_events() for s in event.subjects})
        canonical_by_subject: dict[str, str] = {}
        if all_subjects and deps.translate_client is not None:
            try:
                translated = await deps.translate_client.translate_passages(all_subjects)
                canonical_by_subject = dict(zip(all_subjects, translated, strict=True))
            except TranslateServiceError as exc:
                log.warning(
                    "confirm_events_table: translate failed, subject filter stays "
                    "same-language-only for this batch: %s",
                    exc,
                )

        created_ids: list[int] = []
        skipped_exact = 0
        conflicts: list[PendingConflict] = []

        for event in pending.extracted_events():
            structured = {
                "date_start": event.date_start,
                "date_end": event.date_end,
                "type": event.type,
                # Only consumed by _filter_table_event_hits_by_subject
                # (04-search.md) - a generic vector search for "экзамен по
                # X" matches every exam note equally (they're all short,
                # templated "экзамен - список предметов" text), so search
                # needs the actual subject list to tell them apart. In the
                # canonical translate language, not event.subjects' own
                # (table's) language - see canonical_by_subject above.
                "subjects": [canonical_by_subject.get(s, s) for s in event.subjects],
            }
            tags = sorted({event.tag, *pending.topic_tags, *event.subjects})

            existing = existing_by_key.get((event.date_start, event.date_end, event.type))
            if existing is not None:
                if existing.raw_text == event.note_text:
                    skipped_exact += 1
                else:
                    conflicts.append(
                        PendingConflict(
                            note_id=existing.id,
                            old_text=existing.raw_text or "",
                            new_text=event.note_text,
                            new_tags=tags,
                            new_structured=structured,
                        )
                    )
                continue

            note = await note_repo.create_note(
                user_id=pending.user_id,
                chat_id=pending.chat_id,
                is_group=pending.is_group,
                # No natural Telegram message per event (one album produced
                # N events) — ADR-8's idempotency key doesn't apply here,
                # each confirm click is its own one-off batch.
                tg_message_id=None,
                source_type="table_event",
                raw_text=event.note_text,
                visibility=visibility,
            )
            if note is None:
                continue
            await note_repo.set_tags_and_skip_enrich(note.id, tags, structured=structured)
            created_ids.append(note.id)
        await session.commit()

    for note_id in created_ids:
        enqueue_process_note(deps.fast_queue, note_id)

    await deps.pending_events_cache.delete(session_id)

    conflicts_session_id = None
    if conflicts and deps.conflicts_cache is not None:
        conflicts_session_id = await deps.conflicts_cache.create(
            user_id=pending.user_id, conflicts=conflicts
        )

    return ConfirmEventsTableResult(
        ok=True,
        created=len(created_ids),
        skipped_exact=skipped_exact,
        conflicts_session_id=conflicts_session_id,
        conflicts=[
            ConflictPreview(index=i, old_text=c.old_text, new_text=c.new_text)
            for i, c in enumerate(conflicts)
        ],
    )


async def resolve_events_table_conflict(
    deps: Deps, *, session_id: str, index: int, keep_new: bool, user_id: int
) -> bool:
    """ "Оставить старое" / "Заменить новым" on one dedup conflict. Keeping
    the old note is a pure no-op (it's already there, untouched); replacing
    re-embeds it (`NoteRepository.replace_table_event` sets
    `status='pending'`, same as any other text edit) since its raw_text is
    now different."""
    if deps.conflicts_cache is None:
        return False
    pending = await deps.conflicts_cache.get(session_id)
    if pending is None or pending.user_id != user_id:
        return False
    items = pending.items()
    if not (0 <= index < len(items)):
        return False
    if not keep_new:
        return True

    conflict = items[index]
    async with deps.session_factory() as session:
        await NoteRepository(session).replace_table_event(
            conflict.note_id,
            raw_text=conflict.new_text,
            tags=conflict.new_tags,
            structured=conflict.new_structured,
        )
        await session.commit()
    enqueue_process_note(deps.fast_queue, conflict.note_id)
    return True


async def cancel_events_table(deps: Deps, *, session_id: str, user_id: int) -> bool:
    if deps.pending_events_cache is None:
        return False
    pending = await deps.pending_events_cache.get(session_id)
    if pending is None or pending.user_id != user_id:
        return False
    await deps.pending_events_cache.delete(session_id)
    return True
