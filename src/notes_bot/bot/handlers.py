"""aiogram handlers — thin adapters over bot/logic.py. Parses/formats
Telegram objects and delegates every decision to the pure logic layer.

Private chats: text, page/YouTube/map/Instagram links, voice messages,
/search + /search_mine + /search_all, /list + /trash + delete + restore,
editing (via Telegram's own message-edit, see on_edited_message).
Groups: capture on mention/reply (ADR-10), a welcome message on join,
/capture_all + /capture_mentions, and the same /search.

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

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, ChatMemberUpdated, Message

from notes_bot.bot.keyboards import (
    delete_confirm_keyboard,
    list_item_keyboard,
    list_more_keyboard,
    privacy_keyboard,
    search_more_keyboard,
    trash_item_keyboard,
    trash_more_keyboard,
)
from notes_bot.bot.logic import (
    Deps,
    NotesPage,
    delete_note,
    edit_note_by_message,
    list_notes,
    list_trash,
    restore_note,
    run_search,
    save_note,
    save_voice_note,
    set_group_capture_mode,
    show_more,
    smart_search,
    toggle_privacy,
)
from notes_bot.bot.render import (
    render_capture_mode_changed,
    render_delete_confirmation_prompt,
    render_delete_refused,
    render_edit_saved,
    render_group_note_saved,
    render_group_welcome,
    render_list_empty,
    render_no_more_results,
    render_note_deleted,
    render_note_restored,
    render_note_saved,
    render_privacy_toggle_confirmation,
    render_search_card,
    render_search_expired,
    render_smart_answer_pending,
    render_trash_empty,
    render_voice_note_saved,
)
from notes_bot.db.repositories import ChatSettingsRepository, UserSettingsRepository
from notes_bot.domain.group_capture import Entity, mentions_bot, should_capture_group_message

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
    await _send_page(message, page.hits, page.has_more, page.session_id)


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
    await _send_page(message, page.hits, page.has_more, page.session_id)
    if page.hits and deps.settings.llm_enabled:
        # 04-search.md's mermaid: "Обычная выдача сразу + пометка «готовлю
        # умный ответ»" — the synthesis itself is a separate message later,
        # from the worker (see queue/tasks.py's smart_answer).
        await message.answer(render_smart_answer_pending())


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


async def _send_page(message: Message, hits, has_more: bool, session_id: str | None) -> None:
    if not hits:
        await message.answer(render_no_more_results())
        return
    for hit in hits:
        await message.answer(render_search_card(hit))
    if has_more and session_id:
        await message.answer("Ещё?", reply_markup=search_more_keyboard(session_id))


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

    if not should_capture_group_message(
        capture_mode=capture_mode, mentions_bot=is_mention, is_reply_to_bot=is_reply_to_bot
    ):
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
    # Private-chat only: this lists every one of the caller's own notes,
    # including private ones — running it in a group would broadcast their
    # titles/snippets to everyone there. (listmore:/trashmore: callbacks
    # only ever originate from a message this handler itself sent, so they
    # need no separate chat-type guard.)
    page = await list_notes(deps, user_id=message.from_user.id, offset=0)
    await _send_notes_page(message, page, is_trash=False)


@router.callback_query(F.data.startswith("listmore:"))
async def on_list_more(callback: CallbackQuery, deps: Deps) -> None:
    offset = int(callback.data.removeprefix("listmore:"))
    page = await list_notes(deps, user_id=callback.from_user.id, offset=offset)
    await callback.answer()
    await _send_notes_page(callback.message, page, is_trash=False)


@router.message(Command("trash"), F.chat.type == "private")
async def on_trash(message: Message, deps: Deps) -> None:
    page = await list_trash(deps, user_id=message.from_user.id, offset=0)
    await _send_notes_page(message, page, is_trash=True)


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
        keyboard = trash_item_keyboard(hit.note_id) if is_trash else list_item_keyboard(hit.note_id)
        await message.answer(render_search_card(hit), reply_markup=keyboard)
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
