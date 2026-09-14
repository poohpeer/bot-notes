"""Inline keyboards — pure construction, no aiogram Bot/dispatcher needed to
build or test one. Callback data formats:

    vis:{note_id}        toggle privacy (ADR-5: flips whatever is stored)
    more:{session_id}    "show more" pagination, see 04-search.md
    del:{note_id}        delete, owner-only render (rendered nowhere yet —
                          /list and deletion land in M6)
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
