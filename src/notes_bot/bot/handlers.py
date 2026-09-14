"""aiogram handlers — thin adapters over bot/logic.py. Parses/formats
Telegram objects and delegates every decision to the pure logic layer.

Private chats only: text, page, YouTube, and map links, classified by
domain.classify (M3), /search + /search_mine + /search_all (M2). Voice
messages and Instagram links aren't wired in yet (M4); group capture
(ADR-10) is M6.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message

from notes_bot.bot.keyboards import privacy_keyboard, search_more_keyboard
from notes_bot.bot.logic import Deps, run_search, save_note, show_more, toggle_privacy
from notes_bot.bot.render import (
    render_no_more_results,
    render_note_saved,
    render_privacy_toggle_confirmation,
    render_search_card,
    render_search_expired,
)
from notes_bot.db.repositories import UserSettingsRepository

router = Router(name="notes")


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Привет! Пришли мне текст — сохраню его как заметку с семантическим "
        "поиском. Команды: /search <запрос>, /search_mine, /search_all."
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

    is_group_chat = message.chat.type != "private"
    page = await run_search(
        deps,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        is_group_chat=is_group_chat,
        query_text=query_text,
    )
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
