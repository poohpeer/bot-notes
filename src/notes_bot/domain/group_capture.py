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
    # Telegram counts entity offsets in UTF-16 code units, so a single emoji
    # earlier in the text shifts a slice by Python characters by one and the
    # mention silently stops matching — slice the UTF-16 encoding instead.
    units = text.encode("utf-16-le")
    for entity in entities:
        if entity.type != "mention":
            continue
        mention_text = units[entity.offset * 2 : (entity.offset + entity.length) * 2].decode(
            "utf-16-le", errors="ignore"
        )
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


def should_save_photo_note(*, is_group: bool, mentions_bot: bool) -> bool:
    """A photo with no "/events_table" caption becomes a note — see
    03-ingest.md, "Фото и скриншоты". In a private chat every photo is
    meant for the bot. In a group only one that mentions it in the caption:
    deliberately stricter than should_capture_group_message (no reply-to-bot,
    and /capture_all does not widen it) — a group's pictures are mostly for
    the people in it, and a vision call plus a stored note per picture is
    not something to do on a guess."""
    return (not is_group) or mentions_bot
