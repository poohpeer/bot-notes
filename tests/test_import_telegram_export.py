"""tools/import_telegram_export.py isn't an installed package module — it's
loaded by file path, same as running it as a script."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
import redis as sync_redis
from rq import Queue
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from notes_bot.db.models import Note

pytestmark = pytest.mark.asyncio

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location(
    "import_telegram_export", _TOOLS_DIR / "import_telegram_export.py"
)
import_telegram_export = importlib.util.module_from_spec(_spec)
sys.modules["import_telegram_export"] = import_telegram_export
_spec.loader.exec_module(import_telegram_export)

import_export = import_telegram_export.import_export

CHAT_ID = -1009999


@pytest_asyncio.fixture
async def factory(db_engine):
    sf = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    yield sf
    async with sf() as session:
        await session.execute(delete(Note).where(Note.chat_id == CHAT_ID))
        await session.commit()


@pytest.fixture
def rq_queues():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    conn = sync_redis.from_url(url)
    fast = Queue("fast", connection=conn)
    heavy = Queue("heavy", connection=conn)
    fast.empty()
    heavy.empty()
    yield fast, heavy
    fast.empty()
    heavy.empty()
    conn.close()


def _msg(msg_id, from_id, text, date_unixtime="1700000000"):
    return {
        "id": msg_id,
        "type": "message",
        "date_unixtime": date_unixtime,
        "from": f"User {from_id}",
        "from_id": f"user{from_id}",
        "text": text,
    }


async def test_imports_text_messages_and_enqueues_them(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    messages = [
        _msg(1, 111, "a plain recipe note"),
        _msg(
            2, 222, [{"type": "plain", "text": "a "}, "run of ", {"type": "bold", "text": "text"}]
        ),
    ]

    report = await import_export(
        messages=messages,
        chat_id=CHAT_ID,
        user_map={},
        session_factory=factory,
        fast_queue=fast_queue,
        heavy_queue=heavy_queue,
        dry_run=False,
    )
    assert report.imported == 2
    assert report.skipped_duplicate == 0
    assert report.skipped_non_text == 0
    assert report.skipped_no_user == 0

    async with factory() as session:
        rows = (
            (await session.execute(delete(Note).where(Note.chat_id == CHAT_ID).returning(Note)))
            .scalars()
            .all()
        )
        await session.commit()
    by_id = {n.structured["import_id"]: n for n in rows}
    assert by_id["tg-export:1"].raw_text == "a plain recipe note"
    assert by_id["tg-export:2"].raw_text == "a run of text"
    for note in rows:
        assert note.is_group is True
        assert note.visibility is None
        assert note.tg_message_id is None

    assert len(fast_queue.job_ids) == 2


async def test_reimporting_the_same_export_is_a_noop(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    messages = [_msg(1, 111, "hello")]

    first = await import_export(
        messages=messages,
        chat_id=CHAT_ID,
        user_map={},
        session_factory=factory,
        fast_queue=fast_queue,
        heavy_queue=heavy_queue,
        dry_run=False,
    )
    assert first.imported == 1

    second = await import_export(
        messages=messages,
        chat_id=CHAT_ID,
        user_map={},
        session_factory=factory,
        fast_queue=fast_queue,
        heavy_queue=heavy_queue,
        dry_run=False,
    )
    assert second.imported == 0
    assert second.skipped_duplicate == 1

    async with factory() as session:
        result = await session.execute(delete(Note).where(Note.chat_id == CHAT_ID).returning(Note))
        rows = result.scalars().all()
        await session.commit()
    assert len(rows) == 1


async def test_skips_non_text_and_authorless_messages(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    messages = [
        {"id": 1, "type": "service", "text": "joined the group"},
        {"id": 2, "type": "message", "text": ""},
        {"id": 3, "type": "message", "from": "Deleted Account", "from_id": None, "text": "hi"},
    ]

    report = await import_export(
        messages=messages,
        chat_id=CHAT_ID,
        user_map={},
        session_factory=factory,
        fast_queue=fast_queue,
        heavy_queue=heavy_queue,
        dry_run=False,
    )
    assert report.imported == 0
    assert report.skipped_non_text == 2
    assert report.skipped_no_user == 1


async def test_user_map_resolves_authors_missing_from_id(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    messages = [
        {
            "id": 5,
            "type": "message",
            "date_unixtime": "1700000000",
            "from": "Grandma",
            "from_id": None,
            "text": "soup",
        }
    ]

    report = await import_export(
        messages=messages,
        chat_id=CHAT_ID,
        user_map={"Grandma": 555},
        session_factory=factory,
        fast_queue=fast_queue,
        heavy_queue=heavy_queue,
        dry_run=False,
    )
    assert report.imported == 1
    async with factory() as session:
        result = await session.execute(delete(Note).where(Note.chat_id == CHAT_ID).returning(Note))
        rows = result.scalars().all()
        await session.commit()
    assert rows[0].user_id == 555


async def test_dry_run_writes_nothing(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    messages = [_msg(1, 111, "hello")]

    report = await import_export(
        messages=messages,
        chat_id=CHAT_ID,
        user_map={},
        session_factory=factory,
        fast_queue=fast_queue,
        heavy_queue=heavy_queue,
        dry_run=True,
    )
    assert report.imported == 1

    async with factory() as session:
        result = await session.execute(delete(Note).where(Note.chat_id == CHAT_ID).returning(Note))
        rows = result.scalars().all()
        await session.commit()
    assert rows == []
    assert fast_queue.job_ids == []


async def test_a_url_message_is_classified_and_routed_like_a_live_message(factory, rq_queues):
    fast_queue, heavy_queue = rq_queues
    messages = [_msg(1, 111, "check this out https://www.instagram.com/p/abc123/")]

    await import_export(
        messages=messages,
        chat_id=CHAT_ID,
        user_map={},
        session_factory=factory,
        fast_queue=fast_queue,
        heavy_queue=heavy_queue,
        dry_run=False,
    )
    async with factory() as session:
        result = await session.execute(delete(Note).where(Note.chat_id == CHAT_ID).returning(Note))
        rows = result.scalars().all()
        await session.commit()
    assert rows[0].source_type == "instagram"
    assert fast_queue.job_ids == []
    assert len(heavy_queue.job_ids) == 1
