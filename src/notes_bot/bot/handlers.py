"""aiogram handlers — thin adapters over bot/logic.py. Parses/formats
Telegram objects and delegates every decision to the pure logic layer.

Private chats: text, page/YouTube/map/Instagram links, voice messages,
/search + /search_mine + /search_all, /list + /trash + delete + restore,
editing (via Telegram's own message-edit, see on_edited_message).
Groups: capture on mention/reply — both a new message and an edit of one,
see on_group_message_edited — (ADR-10), a welcome message on join,
/capture_all + /capture_mentions, the same /search, and /list scoped to
the room (on_list_group — never a member's own private-chat notes, see
NoteRepository.list_group). /trash stays private-only.

/events_table (both chat types): an album of screenshots with caption
"/events_table <tab name>" — buffered in-process (on_events_table_photo),
parsed on the `llm` queue (queue/tasks.py's parse_events_table), previewed
with a Сохранить/Отмена keyboard, confirmed here (on_events_table_confirm).

Known gap: 03-ingest.md's flow diagram has the worker notify the bot when a
note finishes processing, and — for voice specifically — "Бот отвечает на
голосовое распознанным текстом, чтобы пользователь сразу видел, что именно
уйдёт в индекс". That notification channel (worker process -> running bot
process) isn't built yet for any source_type, not just voice; this handler
only sends the immediate "saving" acknowledgement, the same as every other
note. Voice notes are still fully indexed and searchable once the heavy
worker finishes — this only affects the "see the transcript immediately"
UX touch.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from dataclasses import dataclass, field

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, ChatMemberUpdated, Message

from notes_bot.bot.keyboards import (
    delete_confirm_keyboard,
    list_item_keyboard,
    list_more_keyboard,
    privacy_keyboard,
    search_detail_keyboard,
    search_more_keyboard,
    trash_item_keyboard,
    trash_more_keyboard,
)
from notes_bot.bot.logic import (
    Deps,
    NotesPage,
    cancel_events_table,
    confirm_events_table,
    delete_note,
    edit_note_by_message,
    list_group_notes,
    list_notes,
    list_trash,
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
from notes_bot.bot.render import (
    render_capture_mode_changed,
    render_debug_toggle_confirmation,
    render_delete_confirmation_prompt,
    render_delete_refused,
    render_edit_saved,
    render_events_table_cancelled,
    render_events_table_expired,
    render_events_table_saved,
    render_events_table_started,
    render_events_table_usage,
    render_group_note_saved,
    render_group_welcome,
    render_list_empty,
    render_no_more_results,
    render_note_deleted,
    render_note_restored,
    render_note_saved,
    render_privacy_toggle_confirmation,
    render_private_only,
    render_search_card,
    render_search_expired,
    render_search_summary_card,
    render_smart_answer_pending,
    render_trash_empty,
    render_voice_note_saved,
)
from notes_bot.clients.llm import ImageInput
from notes_bot.clients.telegram_files import TelegramFileClient, TelegramFileError
from notes_bot.db.repositories import ChatSettingsRepository, UserSettingsRepository
from notes_bot.domain.group_capture import Entity, mentions_bot, should_capture_group_message
from notes_bot.queue.queues import enqueue_parse_events_table

log = logging.getLogger(__name__)

router = Router(name="notes")

_GROUP_TYPES = {"group", "supergroup"}


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Привет! Пришли мне текст — сохраню его как заметку с семантическим "
        "поиском. Команды: /search <запрос>, /smart_search <запрос>, "
        "/search_mine, /search_all, /list, /trash."
    )


@router.message(Command("search_mine"))
async def on_search_mine(message: Message, deps: Deps) -> None:
    await _set_search_mode(message, deps, "mine_only")


@router.message(Command("search_all"))
async def on_search_all(message: Message, deps: Deps) -> None:
    await _set_search_mode(message, deps, "all")


async def _set_search_mode(message: Message, deps: Deps, mode: str) -> None:
    async with deps.session_factory() as session:
        await UserSettingsRepository(session).set_search_mode(message.from_user.id, mode)
        await session.commit()
    label = "только свои" if mode == "mine_only" else "все"
    await message.answer(f"Режим поиска: {label}.")


@router.message(Command("debug"))
async def on_debug(message: Message, deps: Deps) -> None:
    """Toggle — see 03-ingest.md, "Debug: время обработки". The
    notification itself is sent by process_note in the worker, not here;
    this only flips the setting it reads."""
    async with deps.session_factory() as session:
        enabled = await UserSettingsRepository(session).toggle_debug(message.from_user.id)
        await session.commit()
    await message.answer(render_debug_toggle_confirmation(enabled=enabled))


@router.message(Command("search"))
async def on_search(message: Message, deps: Deps) -> None:
    query_text = (message.text or "").partition(" ")[2].strip()
    if not query_text:
        await message.answer("Использование: /search <запрос>")
        return

    is_group_chat = message.chat.type in _GROUP_TYPES
    page = await run_search(
        deps,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        is_group_chat=is_group_chat,
        query_text=query_text,
    )
    await _send_page(message, page.hits, page.has_more, page.session_id, debug_info=page.debug_info)


@router.message(Command("smart_search"))
async def on_smart_search(message: Message, deps: Deps) -> None:
    query_text = (message.text or "").partition(" ")[2].strip()
    if not query_text:
        await message.answer("Использование: /smart_search <запрос>")
        return

    is_group_chat = message.chat.type in _GROUP_TYPES
    page = await smart_search(
        deps,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        is_group_chat=is_group_chat,
        query_text=query_text,
    )
    if not page.hits:
        await message.answer(render_no_more_results())
        if page.debug_info:
            await message.answer(page.debug_info)
        return
    if deps.settings.llm_enabled:
        # No raw search cards — /smart_search's whole point is the LLM's
        # synthesized answer, not the notes behind it (04-search.md); a
        # pending marker instead of silence, since the answer itself is a
        # separate message the worker sends 20-40s later (queue/tasks.py's
        # smart_answer).
        await message.answer(render_smart_answer_pending())
    else:
        # LLM_ENABLED=false means no synthesis is ever coming — the raw
        # cards are the only answer this query will get, same as /search.
        await _send_page(message, page.hits, page.has_more, page.session_id)


@router.callback_query(F.data.startswith("more:"))
async def on_show_more(callback: CallbackQuery, deps: Deps) -> None:
    session_id = callback.data.removeprefix("more:")
    result = await show_more(deps, user_id=callback.from_user.id, session_id=session_id)
    await callback.answer()

    if result.expired:
        await callback.message.answer(render_search_expired())
        return
    if not result.hits:
        await callback.message.answer(render_no_more_results())
        return
    await _send_page(callback.message, result.hits, result.has_more, session_id)


async def _send_page(
    message: Message,
    hits,
    has_more: bool,
    session_id: str | None,
    *,
    debug_info: str | None = None,
) -> None:
    if not hits:
        await message.answer(render_no_more_results())
        if debug_info:
            await message.answer(debug_info)
        return
    for hit in hits:
        # Short by default (render_search_summary_card) with a "Подробнее"
        # button — session_id is always set here (never None alongside a
        # non-empty hits list, see run_search/show_more), which is what the
        # button needs to re-look-up this exact hit later.
        await message.answer(
            render_search_summary_card(hit),
            reply_markup=search_detail_keyboard(session_id, hit.note_id),
        )
    if has_more and session_id:
        await message.answer("Ещё?", reply_markup=search_more_keyboard(session_id))


@router.callback_query(F.data.startswith("detail:"))
async def on_show_detail(callback: CallbackQuery, deps: Deps) -> None:
    session_id, _, note_id_raw = callback.data.removeprefix("detail:").partition(":")
    try:
        note_id = int(note_id_raw)
    except ValueError:
        await callback.answer()
        return

    hit = await show_detail(
        deps, user_id=callback.from_user.id, session_id=session_id, note_id=note_id
    )
    await callback.answer()
    if hit is None:
        await callback.message.answer(render_search_expired())
        return
    await callback.message.answer(render_search_card(hit), parse_mode="HTML")


@router.callback_query(F.data.startswith("vis:"))
async def on_toggle_privacy(callback: CallbackQuery, deps: Deps) -> None:
    note_id = int(callback.data.removeprefix("vis:"))
    new_visibility = await toggle_privacy(deps, note_id=note_id, user_id=callback.from_user.id)
    await callback.answer()
    if new_visibility is None:
        # Not the owner, or the note is gone — no ownership hint given back,
        # same as every other mutation in this bot (04-search.md).
        return
    await callback.message.edit_text(
        render_privacy_toggle_confirmation(new_visibility),
        reply_markup=privacy_keyboard(note_id, new_visibility),
    )


@router.message(F.chat.type == "private", F.text, ~F.text.startswith("/"))
async def on_note_message(message: Message, deps: Deps) -> None:
    """Any non-command text — save_note classifies it (plain text vs a
    page/YouTube/map/Instagram link) and routes accordingly."""
    result = await save_note(
        deps,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        is_group=False,
        tg_message_id=message.message_id,
        text=message.text,
    )
    if not result.created:
        # ADR-8: a retried delivery of a message already saved. Silent —
        # answering again would look like a duplicate save to the user.
        return
    await message.answer(
        render_note_saved(visibility=result.visibility),
        reply_markup=privacy_keyboard(result.note_id, result.visibility),
    )


@router.message(F.chat.type == "private", F.voice)
async def on_voice_message(message: Message, deps: Deps) -> None:
    result = await save_voice_note(
        deps,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        is_group=False,
        tg_message_id=message.message_id,
        file_id=message.voice.file_id,
        caption=message.caption,
    )
    if not result.created:
        return
    await message.answer(
        render_voice_note_saved(visibility=result.visibility),
        reply_markup=privacy_keyboard(result.note_id, result.visibility),
    )


@router.edited_message(F.chat.type == "private", F.text)
async def on_edited_message(message: Message, deps: Deps) -> None:
    """The user edited their own message in Telegram — 03-ingest.md,
    "Редактирование": "пользователь присылает новое сообщение как замену".
    Silent on a note that can't be edited (wrong owner, not a text note, or
    never saved as a note at all — plenty of edited messages aren't) rather
    than commenting on every edit anyone makes in the chat."""
    edited = await edit_note_by_message(
        deps,
        chat_id=message.chat.id,
        tg_message_id=message.message_id,
        user_id=message.from_user.id,
        new_text=message.text,
    )
    if edited:
        await message.reply(render_edit_saved())


# ---------------------------------------------------------------------------
# Groups (ADR-10)
# ---------------------------------------------------------------------------


@router.my_chat_member(F.chat.type.in_(_GROUP_TYPES))
async def on_bot_membership_changed(event: ChatMemberUpdated) -> None:
    old_status = event.old_chat_member.status
    new_status = event.new_chat_member.status
    if old_status in ("left", "kicked") and new_status in ("member", "administrator"):
        await event.bot.send_message(event.chat.id, render_group_welcome())


@router.message(Command("capture_all"), F.chat.type.in_(_GROUP_TYPES))
async def on_capture_all(message: Message, deps: Deps) -> None:
    await set_group_capture_mode(deps, chat_id=message.chat.id, mode="all")
    await message.answer(render_capture_mode_changed("all"))


@router.message(Command("capture_mentions"), F.chat.type.in_(_GROUP_TYPES))
async def on_capture_mentions(message: Message, deps: Deps) -> None:
    await set_group_capture_mode(deps, chat_id=message.chat.id, mode="mentions_and_replies")
    await message.answer(render_capture_mode_changed("mentions_and_replies"))


@router.message(F.chat.type.in_(_GROUP_TYPES), F.text, ~F.text.startswith("/"))
async def on_group_message(message: Message, deps: Deps) -> None:
    await _maybe_capture_group_message(message, deps, event="message")


@router.edited_message(F.chat.type.in_(_GROUP_TYPES), F.text, ~F.text.startswith("/"))
async def on_group_message_edited(message: Message, deps: Deps) -> None:
    """A message that starts as a plain notice and gets edited to add
    "@botname" (or turned into a reply — not applicable here, Telegram
    doesn't let you add a reply-link after sending) must still be
    capturable, or "мешенить постфактум" quietly does nothing — observed
    live: a group member edited a message to prepend the bot's mention,
    and it never reached any handler (aiogram logged the update itself as
    "not handled" — no filter matched at all, since only the private-chat
    edited_message handler existed before this one)."""
    await _maybe_capture_group_message(message, deps, event="edited_message")


async def _maybe_capture_group_message(message: Message, deps: Deps, *, event: str) -> None:
    async with deps.session_factory() as session:
        capture_mode = await ChatSettingsRepository(session).get_capture_mode(message.chat.id)

    entities = [
        Entity(type=e.type, offset=e.offset, length=e.length) for e in (message.entities or [])
    ]
    is_mention = mentions_bot(message.text, entities, deps.bot_username)
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == message.bot.id
    )
    should_capture = should_capture_group_message(
        capture_mode=capture_mode, mentions_bot=is_mention, is_reply_to_bot=is_reply_to_bot
    )
    # Diagnostic trail for "почему это не сохранилось" reports — the only
    # prior signal was aiogram's own "Update is/isn't handled" log line,
    # which doesn't say *why* a matched handler chose not to capture.
    log.info(
        "group capture: event=%s chat_id=%s msg_id=%s mode=%s mentions_bot=%s "
        "is_reply_to_bot=%s capture=%s",
        event,
        message.chat.id,
        message.message_id,
        capture_mode,
        is_mention,
        is_reply_to_bot,
        should_capture,
    )
    if not should_capture:
        return

    result = await save_note(
        deps,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        is_group=True,
        tg_message_id=message.message_id,
        text=message.text,
    )
    if not result.created:
        return
    await message.reply(render_group_note_saved())


# ---------------------------------------------------------------------------
# /list, /trash, delete, restore
# ---------------------------------------------------------------------------


@router.message(Command("list"), F.chat.type == "private")
async def on_list(message: Message, deps: Deps) -> None:
    # Private chat: every one of the caller's own notes, including private
    # ones — see on_list_group below for the group variant, which is
    # scoped to the room instead (never a user's private DMs' notes).
    page = await list_notes(deps, user_id=message.from_user.id, offset=0)
    await _send_notes_page(message, page, is_trash=False)


@router.message(Command("list"), F.chat.type.in_(_GROUP_TYPES))
async def on_list_group(message: Message, deps: Deps) -> None:
    # Deliberately NOT list_notes(user_id=...): that would broadcast the
    # caller's own private DMs' notes into the room. list_group_notes is
    # scoped to notes captured in *this* chat_id only (NoteRepository.
    # list_group's own docstring) — what ADR-10 group capture actually put
    # there, from any member.
    page = await list_group_notes(
        deps, chat_id=message.chat.id, viewer_user_id=message.from_user.id, offset=0
    )
    await _send_notes_page(message, page, is_trash=False)


@router.callback_query(F.data.startswith("listmore:"))
async def on_list_more(callback: CallbackQuery, deps: Deps) -> None:
    offset = int(callback.data.removeprefix("listmore:"))
    if callback.message.chat.type in _GROUP_TYPES:
        page = await list_group_notes(
            deps,
            chat_id=callback.message.chat.id,
            viewer_user_id=callback.from_user.id,
            offset=offset,
        )
    else:
        page = await list_notes(deps, user_id=callback.from_user.id, offset=offset)
    await callback.answer()
    await _send_notes_page(callback.message, page, is_trash=False)


@router.message(Command("trash"), F.chat.type == "private")
async def on_trash(message: Message, deps: Deps) -> None:
    page = await list_trash(deps, user_id=message.from_user.id, offset=0)
    await _send_notes_page(message, page, is_trash=True)


@router.message(Command("trash"), F.chat.type.in_(_GROUP_TYPES))
async def on_trash_group(message: Message) -> None:
    # A room's shared trash of soft-deleted notes isn't something every
    # member should see — unlike /list, this stays private-only. Previously
    # this command just silently did nothing in a group; an explicit reply
    # beats a "why doesn't this work" report.
    await message.reply(render_private_only())


@router.callback_query(F.data.startswith("trashmore:"))
async def on_trash_more(callback: CallbackQuery, deps: Deps) -> None:
    offset = int(callback.data.removeprefix("trashmore:"))
    page = await list_trash(deps, user_id=callback.from_user.id, offset=offset)
    await callback.answer()
    await _send_notes_page(callback.message, page, is_trash=True)


async def _send_notes_page(message: Message, page: NotesPage, *, is_trash: bool) -> None:
    if not page.hits:
        await message.answer(render_trash_empty() if is_trash else render_list_empty())
        return
    for hit in page.hits:
        if is_trash:
            keyboard = trash_item_keyboard(hit.note_id)
        else:
            # Group /list can show notes saved by other members — only the
            # actual owner gets a delete button (soft_delete's own WHERE
            # clause is the real check; this is the same rendering
            # convenience as render_search_card's is_owner gate).
            keyboard = list_item_keyboard(hit.note_id) if hit.is_owner else None
        await message.answer(render_search_card(hit), reply_markup=keyboard, parse_mode="HTML")
    if page.has_more:
        more_keyboard = (
            trash_more_keyboard(page.next_offset)
            if is_trash
            else list_more_keyboard(page.next_offset)
        )
        await message.answer("Ещё?", reply_markup=more_keyboard)


@router.callback_query(F.data.startswith("del:"))
async def on_delete_request(callback: CallbackQuery) -> None:
    note_id = int(callback.data.removeprefix("del:"))
    await callback.answer()
    await callback.message.answer(
        render_delete_confirmation_prompt(), reply_markup=delete_confirm_keyboard(note_id)
    )


@router.callback_query(F.data.startswith("delyes:"))
async def on_delete_confirmed(callback: CallbackQuery, deps: Deps) -> None:
    note_id = int(callback.data.removeprefix("delyes:"))
    deleted = await delete_note(deps, note_id=note_id, user_id=callback.from_user.id)
    await callback.answer()
    await callback.message.edit_text(render_note_deleted() if deleted else render_delete_refused())


@router.callback_query(F.data.startswith("delno:"))
async def on_delete_cancelled(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.delete()


@router.callback_query(F.data.startswith("restore:"))
async def on_restore(callback: CallbackQuery, deps: Deps) -> None:
    note_id = int(callback.data.removeprefix("restore:"))
    restored = await restore_note(deps, note_id=note_id, user_id=callback.from_user.id)
    await callback.answer()
    await callback.message.edit_text(
        render_note_restored() if restored else render_delete_refused()
    )


# ---------------------------------------------------------------------------
# /events_table — see docs/architecture/03-ingest.md, "/events_table"
# ---------------------------------------------------------------------------

# One or more screenshots sent as an album arrive as separate `message`
# updates sharing `media_group_id`, with no signal from Telegram for "this
# is the last one" — buffered here (single bot replica, per cli/bot.py's own
# module docstring, so in-process state is safe) and flushed after a short
# gap in arrivals. A lone photo (no album) gets a synthetic key instead of
# `None`, so two single-photo messages arriving close together in the same
# chat never share a buffer entry.
_ALBUM_DEBOUNCE_S = 2.0
_EVENTS_TABLE_CAPTION_RE = re.compile(r"^/events_table(?:@\w+)?(?:\s+(.*))?$", re.DOTALL)


@dataclass
class _PendingAlbum:
    file_ids: list[str] = field(default_factory=list)
    # Whichever message in the group actually carried the caption — that's
    # the one whose text tells us this is an /events_table submission at
    # all, and the one we reply to once parsing starts.
    caption_message: Message | None = None
    any_message: Message | None = None


_album_buffers: dict[str, _PendingAlbum] = {}
_album_tasks: dict[str, asyncio.Task] = {}


@router.message(Command("events_table"))
async def on_events_table_usage(message: Message) -> None:
    # Only ever matches a plain text message — a photo's caption lives in
    # `message.caption`, not `message.text`, so a captioned photo never
    # reaches this handler at all; see on_events_table_photo below.
    await message.answer(render_events_table_usage())


@router.message(F.photo)
async def on_events_table_photo(message: Message, deps: Deps) -> None:
    group_key = message.media_group_id or f"single:{message.chat.id}:{message.message_id}"
    pending = _album_buffers.setdefault(group_key, _PendingAlbum())
    # Telegram lists each photo at several resolutions; [-1] is the largest.
    pending.file_ids.append(message.photo[-1].file_id)
    pending.any_message = message
    if message.caption:
        pending.caption_message = message

    previous_task = _album_tasks.get(group_key)
    if previous_task is not None:
        previous_task.cancel()
    _album_tasks[group_key] = asyncio.create_task(_flush_events_table_album(group_key, deps))


async def _flush_events_table_album(group_key: str, deps: Deps) -> None:
    try:
        await asyncio.sleep(_ALBUM_DEBOUNCE_S)
    except asyncio.CancelledError:
        return  # a later photo in the same album superseded this timer
    finally:
        _album_tasks.pop(group_key, None)

    pending = _album_buffers.pop(group_key, None)
    if pending is None or pending.caption_message is None:
        return  # no /events_table caption ever arrived — an ordinary photo

    match = _EVENTS_TABLE_CAPTION_RE.match(pending.caption_message.caption or "")
    if not match:
        return
    tab_name = (match.group(1) or "").strip()
    reply_target = pending.caption_message
    if not tab_name:
        await reply_target.answer(render_events_table_usage())
        return

    file_client = TelegramFileClient(deps.settings.telegram_bot_token)
    images: list[ImageInput] = []
    for file_id in pending.file_ids:
        try:
            data = await file_client.download(file_id)
        except TelegramFileError as exc:
            log.warning("events_table: failed to download photo %s: %s", file_id, exc)
            continue
        images.append(
            ImageInput(media_type="image/jpeg", data_base64=base64.b64encode(data).decode())
        )
    if not images:
        await reply_target.answer(render_events_table_usage())
        return

    await reply_target.answer(render_events_table_started(tab_name))
    enqueue_parse_events_table(
        deps.llm_queue,
        chat_id=reply_target.chat.id,
        user_id=reply_target.from_user.id,
        is_group=reply_target.chat.type in _GROUP_TYPES,
        tab_name=tab_name,
        images=images,
        job_timeout=int(deps.settings.events_table_timeout_s) + 60,
    )


@router.callback_query(F.data.startswith("evtable_yes:"))
async def on_events_table_confirm(callback: CallbackQuery, deps: Deps) -> None:
    session_id = callback.data.removeprefix("evtable_yes:")
    result = await confirm_events_table(deps, session_id=session_id, user_id=callback.from_user.id)
    await callback.answer()
    if not result.ok:
        await callback.message.answer(render_events_table_expired())
        return
    await callback.message.edit_text(render_events_table_saved(result.count))


@router.callback_query(F.data.startswith("evtable_no:"))
async def on_events_table_cancel(callback: CallbackQuery, deps: Deps) -> None:
    session_id = callback.data.removeprefix("evtable_no:")
    cancelled = await cancel_events_table(
        deps, session_id=session_id, user_id=callback.from_user.id
    )
    await callback.answer()
    if not cancelled:
        await callback.message.answer(render_events_table_expired())
        return
    await callback.message.edit_text(render_events_table_cancelled())
