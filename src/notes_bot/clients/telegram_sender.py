"""Sends one message via the Bot API from a worker process — see
queue/tasks.py's `smart_answer`. The worker has no Dispatcher/long-polling
loop and doesn't need one just to call sendMessage; a Bot instance built
for a single call is enough, closed right after.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from aiogram import Bot

if TYPE_CHECKING:
    from aiogram.types import InlineKeyboardMarkup


class Sender(Protocol):
    async def send(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None: ...


class TelegramSender:
    def __init__(self, bot_token: str) -> None:
        self._bot_token = bot_token

    async def send(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        bot = Bot(token=self._bot_token)
        try:
            await bot.send_message(chat_id, text, parse_mode=parse_mode, reply_markup=reply_markup)
        finally:
            await bot.session.close()
