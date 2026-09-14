"""Core bot logic — no aiogram Message/CallbackQuery types here, so it is
testable without constructing Telegram objects. `bot/handlers.py` is the
thin aiogram-specific adapter on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass

from rq import Queue
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.bot.render import RenderableHit
from notes_bot.clients.embeddings import HttpEmbeddingClient
from notes_bot.clients.search_cache import SearchSessionCache
from notes_bot.config import Settings
from notes_bot.db.repositories import NoteRepository, UserSettingsRepository
from notes_bot.db.search import search_notes
from notes_bot.domain.acl import visibility_predicate
from notes_bot.queue.queues import enqueue_process_note


@dataclass(frozen=True)
class Deps:
    session_factory: async_sessionmaker
    embedding_client: HttpEmbeddingClient
    search_cache: SearchSessionCache
    fast_queue: Queue
    settings: Settings


@dataclass(frozen=True)
class SaveTextNoteResult:
    created: bool
    note_id: int | None
    visibility: str | None


async def save_text_note(
    deps: Deps, *, user_id: int, chat_id: int, is_group: bool, tg_message_id: int, text: str
) -> SaveTextNoteResult:
    """Fail-closed privacy (ADR-5): a private-chat note is saved with the
    user's default visibility immediately, before any button is pressed. A
    group note gets no visibility at all — group membership is its scope."""
    async with deps.session_factory() as session:
        visibility: str | None = None
        if not is_group:
            settings_row = await UserSettingsRepository(session).get_or_create(user_id)
            visibility = settings_row.default_visibility

        note = await NoteRepository(session).create_text_note(
            user_id=user_id,
            chat_id=chat_id,
            is_group=is_group,
            tg_message_id=tg_message_id,
            raw_text=text,
            visibility=visibility,
        )
        await session.commit()

    if note is None:
        # ADR-8: a retried Telegram delivery for a message already saved.
        return SaveTextNoteResult(created=False, note_id=None, visibility=None)

    enqueue_process_note(deps.fast_queue, note.id)
    return SaveTextNoteResult(created=True, note_id=note.id, visibility=visibility)


async def toggle_privacy(deps: Deps, *, note_id: int, user_id: int) -> str | None:
    async with deps.session_factory() as session:
        new_visibility = await NoteRepository(session).toggle_visibility(note_id, user_id)
        await session.commit()
    return new_visibility


@dataclass(frozen=True)
class SearchPageResult:
    hits: list[RenderableHit]
    has_more: bool
    session_id: str | None


async def run_search(
    deps: Deps, *, user_id: int, chat_id: int, is_group_chat: bool, query_text: str
) -> SearchPageResult:
    page_size = deps.settings.search_page_size

    async with deps.session_factory() as session:
        search_mode = "all"
        if not is_group_chat:
            settings_row = await UserSettingsRepository(session).get_or_create(user_id)
            search_mode = settings_row.search_mode

        query_vector = await deps.embedding_client.embed_query(query_text)
        predicate = visibility_predicate(
            user_id=user_id, chat_id=chat_id, is_group_chat=is_group_chat, search_mode=search_mode
        )
        hits = await search_notes(
            session,
            query_vector=query_vector,
            acl_predicate=predicate,
            active_model=deps.embedding_client.model_name,
            candidate_k=deps.settings.search_candidate_k,
            limit=50,
            offset=0,
        )

    if not hits:
        return SearchPageResult(hits=[], has_more=False, session_id=None)

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
    )
