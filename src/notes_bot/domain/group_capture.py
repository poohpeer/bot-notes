"""Whether a group message should be saved — see docs/architecture/03-ingest.md,
"Шаг 2. Приём в боте", and ADR-10 in 07-decisions.md.

Pure functions: no aiogram types, no I/O — the handler extracts the plain
facts (capture_mode from chat_settings, whether the message mentions the
bot or replies to it) and this decides.
"""

from __future__ import annotations

from typing import NamedTuple


class Entity(NamedTuple):
    type: str
    offset: int
    length: int


def mentions_bot(text: str, entities: list[Entity], bot_username: str) -> bool:
    """A `mention` entity ("@name") whose text matches the bot's own
    username — not a `text_mention` (a mention-by-name-without-@, which
    Telegram only emits for users without a username; a bot always has
    one, so that entity type never applies here).
    """
    target = f"@{bot_username}".lower()
    for entity in entities:
        if entity.type != "mention":
            continue
        mention_text = text[entity.offset : entity.offset + entity.length]
        if mention_text.lower() == target:
            return True
    return False


def should_capture_group_message(
    *, capture_mode: str, mentions_bot: bool, is_reply_to_bot: bool
) -> bool:
    """ADR-10: default `mentions_and_replies` only saves what's addressed
    to the bot; `all` saves everything (and needs Telegram privacy mode
    off to even receive it — see ADR-10's note on BotFather settings,
    enforced by Telegram itself, not this function)."""
    if capture_mode == "all":
        return True
    return mentions_bot or is_reply_to_bot
