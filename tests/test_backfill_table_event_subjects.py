"""tools/backfill_table_event_subjects.py isn't an installed package
module — it's loaded by file path, same as
tests/test_import_telegram_export.py does for its own tool."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.db.models import Note
from notes_bot.db.repositories import NoteRepository

pytestmark = pytest.mark.asyncio

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location(
    "backfill_table_event_subjects", _TOOLS_DIR / "backfill_table_event_subjects.py"
)
backfill_table_event_subjects = importlib.util.module_from_spec(_spec)
sys.modules["backfill_table_event_subjects"] = backfill_table_event_subjects
_spec.loader.exec_module(backfill_table_event_subjects)

backfill = backfill_table_event_subjects.backfill

CHAT_ID = -1008888


@dataclass
class _FakeResult:
    parsed: dict


class ScriptedLLMClient:
    def __init__(self, by_text: dict[str, list[str]]) -> None:
        self._by_text = by_text

    async def complete(self, *, system, user, json_schema=None, history=None, timeout_s=60.0):
        return _FakeResult(parsed={"subjects": self._by_text.get(user, [])})


class FakeTranslateClient:
    async def translate_passages(self, texts):
        return [f"[en] {t}" for t in texts]


@pytest.fixture
async def factory(db_engine):
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    yield sf
    async with sf() as session:
        await session.execute(delete(Note).where(Note.chat_id == CHAT_ID))
        await session.commit()


async def _make_table_event_note(factory, *, raw_text: str) -> int:
    async with factory() as session:
        repo = NoteRepository(session)
        note = await repo.create_note(
            user_id=1,
            chat_id=CHAT_ID,
            is_group=False,
            tg_message_id=None,
            source_type="table_event",
            raw_text=raw_text,
            visibility="private",
        )
        await repo.set_tags_and_skip_enrich(
            note.id,
            ["экзамен"],
            structured={"date_start": "01/09/2026", "date_end": "01/09/2026", "type": "exam"},
        )
        await session.commit()
        return note.id


async def test_backfill_fills_subjects_and_canonical_subjects(factory):
    note_id = await _make_table_event_note(factory, raw_text="01/09/2026: מבחן מתמטיקה")
    llm = ScriptedLLMClient({"01/09/2026: מבחן מתמטיקה": ["מתמטיקה"]})

    updated = await backfill(
        session_factory=factory,
        llm=llm,
        translate_client=FakeTranslateClient(),
        dry_run=False,
        timeout_s=10,
    )

    assert updated == 1
    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert "מתמטיקה" in note.tags
        assert note.structured["subjects"] == ["[en] מתמטיקה"]


async def test_backfill_skips_notes_that_already_have_subjects(factory):
    note_id = await _make_table_event_note(factory, raw_text="01/09/2026: מבחן מתמטיקה")
    async with factory() as session:
        await NoteRepository(session).merge_table_event_subjects(
            note_id, subjects=["מתמטיקה"], canonical_subjects=["mathematics"]
        )
        await session.commit()

    llm = ScriptedLLMClient({})  # would return [] for anything - proves it's never called
    updated = await backfill(
        session_factory=factory, llm=llm, translate_client=None, dry_run=False, timeout_s=10
    )

    assert updated == 0


async def test_backfill_dry_run_writes_nothing(factory):
    note_id = await _make_table_event_note(factory, raw_text="01/09/2026: מבחן מתמטיקה")
    llm = ScriptedLLMClient({"01/09/2026: מבחן מתמטיקה": ["מתמטיקה"]})

    updated = await backfill(
        session_factory=factory, llm=llm, translate_client=None, dry_run=True, timeout_s=10
    )

    assert updated == 0
    async with factory() as session:
        note = await NoteRepository(session).get(note_id)
        assert "subjects" not in (note.structured or {})
