"""Table-driven ACL tests — the single place a bug here leaks a private note
between users. See ADR-12 and docs/architecture/04-search.md.

Requires a real Postgres (the CHECK constraints and column types are part of
what's under test) — see tests/conftest.py.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from notes_bot.db.models import Note
from notes_bot.domain.acl import visibility_predicate

pytestmark = pytest.mark.asyncio

ALICE = 1
BOB = 2
GROUP_TRIP = 100
GROUP_OTHER = 200


async def _seed(session) -> dict[str, Note]:
    notes = {
        "alice_private": Note(
            user_id=ALICE,
            chat_id=ALICE,
            is_group=False,
            visibility="private",
            source_type="text",
            raw_text="alice private",
        ),
        "alice_public": Note(
            user_id=ALICE,
            chat_id=ALICE,
            is_group=False,
            visibility="public",
            source_type="text",
            raw_text="alice public",
        ),
        "bob_private": Note(
            user_id=BOB,
            chat_id=BOB,
            is_group=False,
            visibility="private",
            source_type="text",
            raw_text="bob private",
        ),
        "bob_public": Note(
            user_id=BOB,
            chat_id=BOB,
            is_group=False,
            visibility="public",
            source_type="text",
            raw_text="bob public",
        ),
        "alice_in_trip_group": Note(
            user_id=ALICE,
            chat_id=GROUP_TRIP,
            is_group=True,
            visibility=None,
            source_type="text",
            raw_text="alice in trip group",
        ),
        "bob_in_trip_group": Note(
            user_id=BOB,
            chat_id=GROUP_TRIP,
            is_group=True,
            visibility=None,
            source_type="text",
            raw_text="bob in trip group",
        ),
        "bob_in_other_group": Note(
            user_id=BOB,
            chat_id=GROUP_OTHER,
            is_group=True,
            visibility=None,
            source_type="text",
            raw_text="bob in other group",
        ),
    }
    session.add_all(notes.values())
    await session.flush()
    return notes


async def _visible_labels(session, notes: dict[str, Note], predicate) -> set[str]:
    result = await session.execute(select(Note.id).where(predicate))
    visible_ids = {row[0] for row in result}
    by_id = {n.id: label for label, n in notes.items()}
    return {by_id[i] for i in visible_ids}


async def test_group_search_sees_only_that_group(db_session):
    notes = await _seed(db_session)
    predicate = visibility_predicate(
        user_id=ALICE, chat_id=GROUP_TRIP, is_group_chat=True, search_mode="all"
    )
    assert await _visible_labels(db_session, notes, predicate) == {
        "alice_in_trip_group",
        "bob_in_trip_group",
    }


async def test_group_search_does_not_leak_a_different_group(db_session):
    notes = await _seed(db_session)
    predicate = visibility_predicate(
        user_id=BOB, chat_id=GROUP_OTHER, is_group_chat=True, search_mode="all"
    )
    assert await _visible_labels(db_session, notes, predicate) == {"bob_in_other_group"}


async def test_private_chat_all_mode_sees_own_and_public(db_session):
    """Own notes (including ones sent into a group — see
    test_authors_own_group_note_is_visible_in_their_personal_search) plus
    public notes from other private chats. Group notes authored by someone
    else stay out — that's the "never leaks" test below."""
    notes = await _seed(db_session)
    predicate = visibility_predicate(
        user_id=ALICE, chat_id=ALICE, is_group_chat=False, search_mode="all"
    )
    assert await _visible_labels(db_session, notes, predicate) == {
        "alice_private",
        "alice_public",
        "alice_in_trip_group",
        "bob_public",
    }


async def test_private_chat_mine_only_excludes_others_public_notes(db_session):
    notes = await _seed(db_session)
    predicate = visibility_predicate(
        user_id=ALICE, chat_id=ALICE, is_group_chat=False, search_mode="mine_only"
    )
    assert await _visible_labels(db_session, notes, predicate) == {
        "alice_private",
        "alice_public",
        "alice_in_trip_group",
    }


async def test_private_chat_all_mode_never_leaks_a_group_note_authored_by_someone_else(
    db_session,
):
    """Bob's own group note is visible to him via the `user_id` branch (see
    test_authors_own_group_note_is_visible_in_their_personal_search) — what
    must never leak through the private-chat 'all' branch is a group note
    authored by someone else, to someone who was never in that group."""
    notes = await _seed(db_session)
    predicate = visibility_predicate(
        user_id=BOB, chat_id=BOB, is_group_chat=False, search_mode="all"
    )
    assert "alice_in_trip_group" not in await _visible_labels(db_session, notes, predicate)


async def test_authors_own_group_note_is_visible_in_their_personal_search(db_session):
    """A note the user sent into a group is visible to them in personal
    search too — via the `user_id` branch, per 04-search.md — but not to
    other group members outside the group."""
    notes = await _seed(db_session)
    predicate = visibility_predicate(
        user_id=ALICE, chat_id=ALICE, is_group_chat=False, search_mode="mine_only"
    )
    assert "alice_in_trip_group" in await _visible_labels(db_session, notes, predicate)
