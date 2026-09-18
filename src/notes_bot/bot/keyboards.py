"""Inline keyboards — pure construction, no aiogram Bot/dispatcher needed to
build or test one. Callback data formats:

    vis:{note_id}         toggle privacy (ADR-5: flips whatever is stored)
    more:{session_id}     "show more" search pagination, see 04-search.md
    detail:{session_id}:{note_id}   expand a /search summary card to the full one
    listmore:{offset}     "show more" for /list, plain SQL offset
    trashmore:{offset}    "show more" for /trash, plain SQL offset
    del:{note_id}         delete — asks for confirmation first
    delyes:{note_id}      confirmed delete
    delno:{note_id}       cancelled delete
    restore:{note_id}     restore from /trash
    evtable_yes:{session_id}   /events_table — confirm, create the notes
    evtable_no:{session_id}    /events_table — cancel, discard the parse
"""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def privacy_keyboard(note_id: int, visibility: str) -> InlineKeyboardMarkup:
    # The button shows the CURRENT state as a toggle target, not the state
    # itself — pressing "🌐 сделать публичной" is what makes it public.
    label = "🌐 сделать публичной" if visibility == "private" else "🔒 сделать приватной"
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=label, callback_data=f"vis:{note_id}")]]
    )


def search_more_keyboard(session_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Показать ещё 5", callback_data=f"more:{session_id}")]
        ]
    )


def search_detail_keyboard(session_id: str, note_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Подробнее", callback_data=f"detail:{session_id}:{note_id}")]
        ]
    )


def list_item_keyboard(note_id: int) -> InlineKeyboardMarkup:
    """/list — every item is the caller's own, so a delete button is always
    safe to render (04-search.md: rendering is a convenience, the real
    check is the WHERE clause on the mutation itself)."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="🗑 удалить", callback_data=f"del:{note_id}")]]
    )


def delete_confirm_keyboard(note_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Да, удалить", callback_data=f"delyes:{note_id}"),
                InlineKeyboardButton(text="Отмена", callback_data=f"delno:{note_id}"),
            ]
        ]
    )


def events_table_confirm_keyboard(session_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Сохранить", callback_data=f"evtable_yes:{session_id}"
                ),
                InlineKeyboardButton(text="❌ Отмена", callback_data=f"evtable_no:{session_id}"),
            ]
        ]
    )


def trash_item_keyboard(note_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="♻️ восстановить", callback_data=f"restore:{note_id}")]
        ]
    )


def list_more_keyboard(offset: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Показать ещё", callback_data=f"listmore:{offset}")]
        ]
    )


def trash_more_keyboard(offset: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Показать ещё", callback_data=f"trashmore:{offset}")]
        ]
    )
