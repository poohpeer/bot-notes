from __future__ import annotations

import os

import pytest
import redis.asyncio as redis

from notes_bot.clients.llm import ImageInput, LLMResult
from notes_bot.clients.pending_events_cache import PendingEventsCache
from notes_bot.queue.tasks import parse_events_table_async

pytestmark = pytest.mark.asyncio


class ScriptedLLMClient:
    def __init__(self, parsed):
        self._parsed = parsed

    async def complete(
        self, *, system, user, json_schema=None, history=None, timeout_s=60.0, images=None
    ):
        return LLMResult(text="", parsed=self._parsed, model="scripted", usage=None)


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    async def send(self, chat_id, text, *, parse_mode=None, reply_markup=None):
        self.sent.append((chat_id, text, reply_markup))


def _image() -> ImageInput:
    return ImageInput(media_type="image/jpeg", data_base64="Zm9v")


@pytest.fixture
async def pending_cache():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    client = redis.from_url(url)
    yield PendingEventsCache(client)
    await client.flushdb()
    await client.aclose()


async def test_skips_entirely_when_llm_disabled(pending_cache):
    sender = FakeSender()
    await parse_events_table_async(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="t",
        images=[_image()],
        llm_client=ScriptedLLMClient({"topic_tags": [], "events": []}),
        llm_enabled=False,
        timeout_s=10,
        sender=sender,
        pending_cache=pending_cache,
    )
    assert len(sender.sent) == 1
    assert "выключен" in sender.sent[0][1]


async def test_sends_empty_message_when_nothing_parsed(pending_cache):
    sender = FakeSender()
    await parse_events_table_async(
        chat_id=1,
        user_id=1,
        is_group=False,
        tab_name="שכבה י'",
        images=[_image()],
        llm_client=ScriptedLLMClient({"topic_tags": [], "events": []}),
        llm_enabled=True,
        timeout_s=10,
        sender=sender,
        pending_cache=pending_cache,
    )
    assert len(sender.sent) == 1
    assert "שכבה י'" in sender.sent[0][1]
    assert sender.sent[0][2] is None  # no keyboard — nothing to confirm


async def test_sends_a_preview_with_keyboard_and_caches_the_session(pending_cache):
    sender = FakeSender()
    llm = ScriptedLLMClient(
        {
            "topic_tags": ["школа"],
            "events": [
                {"date_start": "01/09/2026", "type": "exam", "text": "a"},
                {"date_start": "02/09/2026", "type": "exam", "text": "b"},
                {"date_start": "11/09/2026", "type": "holiday", "text": "c"},
            ],
        }
    )

    await parse_events_table_async(
        chat_id=1,
        user_id=42,
        is_group=True,
        tab_name="שכבה י'",
        images=[_image()],
        llm_client=llm,
        llm_enabled=True,
        timeout_s=10,
        sender=sender,
        pending_cache=pending_cache,
    )

    assert len(sender.sent) == 1
    chat_id, text, reply_markup = sender.sent[0]
    assert chat_id == 1
    assert "3" in text  # total events found
    assert "экзамен" in text
    assert "праздник" in text
    assert "#школа" in text
    assert reply_markup is not None

    # The session_id lives inside the confirm button's callback_data.
    session_id = reply_markup.inline_keyboard[0][0].callback_data.removeprefix("evtable_yes:")
    pending = await pending_cache.get(session_id)
    assert pending is not None
    assert pending.user_id == 42
    assert pending.is_group is True
    assert pending.tab_name == "שכבה י'"
    assert len(pending.events) == 3
