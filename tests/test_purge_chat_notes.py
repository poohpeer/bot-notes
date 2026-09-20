"""tools/purge_chat_notes.py isn't an installed package module — it's
loaded by file path, same as tests/test_import_telegram_export.py does
for its own tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.db.models import Note
from notes_bot.db.repositories import NoteRepository

pytestmark = pytest.mark.asyncio

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location(
    "purge_chat_notes", _TOOLS_DIR / "purge_chat_notes.py"
)
purge_chat_notes = importlib.util.module_from_spec(_spec)
sys.modules["purge_chat_notes"] = purge_chat_notes
_spec.loader.exec_module(purge_chat_notes)

purge = purge_chat_notes.purge

CHAT_ID = -1007777


@pytest.fixture
async def factory(db_engine):
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    yield sf
    async with sf() as session:
        await session.execute(delete(Note).where(Note.chat_id == CHAT_ID))
        await session.commit()


async def _make_note(factory, *, user_id: int, tg_message_id: int) -> int:
    async with factory() as session:
        repo = NoteRepository(session)
        note = await repo.create_text_note(
            user_id=user_id,
            chat_id=CHAT_ID,
            is_group=True,
            tg_message_id=tg_message_id,
            raw_text="x",
            visibility=None,
        )
        await session.commit()
        return note.id


async def test_dry_run_counts_without_deleting(factory):
    note_id = await _make_note(factory, user_id=1, tg_message_id=1)

    count = await purge(session_factory=factory, chat_id=CHAT_ID, dry_run=True)

    assert count == 1
    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert note.deleted_at is None


async def test_purge_deletes_notes_from_every_author(factory):
    a = await _make_note(factory, user_id=1, tg_message_id=2)
    b = await _make_note(factory, user_id=2, tg_message_id=3)

    count = await purge(session_factory=factory, chat_id=CHAT_ID, dry_run=False)

    assert count == 2
    async with factory() as session:
        repo = NoteRepository(session)
        assert (await repo.get(a)).deleted_at is not None
        assert (await repo.get(b)).deleted_at is not None
