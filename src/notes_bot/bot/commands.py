"""The Telegram slash-command menu — what autocompletes when a user types
"/". Purely cosmetic (every command here still works if typed without ever
appearing in the menu, per handlers.py's own filters); this only controls
what gets suggested, and to whom.

Scoped by chat type because /list and /trash only make sense in a private
chat (handlers.py gates them on `F.chat.type == "private"`, see their own
docstrings on why), and /capture_all + /capture_mentions only make sense in
a group. Offering the other kind in a chat where it's a no-op is confusing,
not just untidy.
"""

from __future__ import annotations

import logging

from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeAllGroupChats, BotCommandScopeAllPrivateChats

log = logging.getLogger(__name__)

_COMMON_COMMANDS = [
    BotCommand(command="start", description="Что я умею"),
    BotCommand(command="search", description="Поиск по заметкам"),
    BotCommand(command="smart_search", description="Поиск с ответом от LLM"),
    BotCommand(command="search_mine", description="Искать только свои заметки"),
    BotCommand(command="search_all", description="Искать заметки всех"),
    BotCommand(command="debug", description="Вкл/выкл время обработки заметок"),
    BotCommand(command="events_table", description="Разобрать скриншот таблицы (расписание)"),
]

PRIVATE_COMMANDS = [
    *_COMMON_COMMANDS,
    BotCommand(command="list", description="Список заметок"),
    BotCommand(command="trash", description="Корзина"),
]

GROUP_COMMANDS = [
    *_COMMON_COMMANDS,
    BotCommand(command="capture_all", description="Сохранять все сообщения в чате"),
    BotCommand(command="capture_mentions", description="Сохранять только упоминания бота"),
]


async def set_bot_commands(bot: Bot) -> None:
    # No BotCommandScopeDefault: leaving it unset means group *and* private
    # chats both fall back to whichever scope is more specific
    # (all_private_chats / all_group_chats below), which is exactly the
    # split wanted here — unlike bot-organizer, nothing here needs an
    # audience default has to catch.
    await bot.set_my_commands(PRIVATE_COMMANDS, scope=BotCommandScopeAllPrivateChats())
    await bot.set_my_commands(GROUP_COMMANDS, scope=BotCommandScopeAllGroupChats())
    log.info(
        "bot command menu set (%d private, %d group)",
        len(PRIVATE_COMMANDS),
        len(GROUP_COMMANDS),
    )
