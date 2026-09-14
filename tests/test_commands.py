"""notes_bot.bot.commands — the slash-command menu. Asserted against the
real handlers rather than a second hand-maintained list, so the two cannot
drift apart the way bot-organizer's did (a menu commands.py's own docstring
explicitly calls out as the failure mode being avoided here).
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import AsyncMock

from aiogram.types import BotCommandScopeAllGroupChats, BotCommandScopeAllPrivateChats

from notes_bot.bot import handlers
from notes_bot.bot.commands import GROUP_COMMANDS, PRIVATE_COMMANDS, set_bot_commands


def _registered_command_names() -> set[str]:
    source = inspect.getsource(handlers)
    names = set(re.findall(r'Command\("(\w+)"\)', source))
    if "CommandStart()" in source:
        names.add("start")
    return names


def test_every_menu_command_has_a_handler():
    menu_commands = {c.command for c in (*PRIVATE_COMMANDS, *GROUP_COMMANDS)}
    assert menu_commands <= _registered_command_names()


def test_every_command_has_a_description():
    assert all(c.description for c in (*PRIVATE_COMMANDS, *GROUP_COMMANDS))


def test_private_only_commands_are_not_offered_in_groups():
    group_names = {c.command for c in GROUP_COMMANDS}
    assert "list" not in group_names
    assert "trash" not in group_names


def test_group_only_commands_are_not_offered_in_private():
    private_names = {c.command for c in PRIVATE_COMMANDS}
    assert "capture_all" not in private_names
    assert "capture_mentions" not in private_names


async def test_startup_actually_sends_both_menus():
    """The list being right is half of it — a missing set_my_commands call
    would leave every test above green while Telegram kept offering
    whatever the menu held before."""
    bot = AsyncMock()

    await set_bot_commands(bot)

    assert bot.set_my_commands.await_count == 2
    scopes_seen = {type(call.kwargs["scope"]) for call in bot.set_my_commands.await_args_list}
    assert scopes_seen == {BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats}
