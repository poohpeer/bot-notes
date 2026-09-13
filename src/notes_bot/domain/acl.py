"""Visibility predicates — see docs/architecture/04-search.md, "ACL-предикаты",
and ADR-12 in 07-decisions.md.

The single place where a mistake means a private note leaks between users.
Kept out of SQL strings and covered by a table-driven test matrix
(author, chat, visibility, is_group, search mode) for exactly that reason —
every other module composes these predicates rather than writing its own
WHERE clause.
"""

from __future__ import annotations

from typing import Literal

from sqlalchemy import ColumnElement, and_, or_

from notes_bot.db.models import Note

SearchMode = Literal["all", "mine_only"]


def visibility_predicate(
    *, user_id: int, chat_id: int, is_group_chat: bool, search_mode: SearchMode
) -> ColumnElement[bool]:
    """What a search from this actor, in this chat, is allowed to see.

    Three scenarios, word for word from docs/architecture/04-search.md:

    1. Search from a group: strictly within that group.
    2. Search from a private chat, mode 'all' (default): the user's own
       notes, plus public notes from other private chats — group notes never
       leak out through this branch (`is_group == False`).
    3. Search from a private chat, mode 'mine_only': only the user's own.
    """
    if is_group_chat:
        return Note.chat_id == chat_id

    own = Note.user_id == user_id
    if search_mode == "mine_only":
        return own

    public_from_private_chats = and_(Note.visibility == "public", Note.is_group.is_(False))
    return or_(own, public_from_private_chats)
