"""notes_bot.clients.alerts — same shape as ai-proxy's own notify.py tests."""

from __future__ import annotations

import httpx
import pytest

from notes_bot.clients.alerts import Notifier, NullNotifier, build_notifier

pytestmark = pytest.mark.asyncio


class Telegram:
    """Stands in for the Bot API, recording what would have been posted."""

    def __init__(self, status: int = 200, raises: Exception | None = None) -> None:
        self.status, self.raises = status, raises
        self.sent: list[dict] = []

    async def post(self, url, data=None):
        if self.raises:
            raise self.raises
        self.sent.append(data)
        return httpx.Response(self.status, text="", request=httpx.Request("POST", url))

    async def aclose(self):
        pass


def _notifier(client, **kw):
    return Notifier("token", "-100123", client=client, **kw)


async def test_a_reclaim_names_the_note_the_reason_and_what_to_expect():
    tg = Telegram()

    await _notifier(tg).reclaimed(
        note_id=11,
        source_type="instagram",
        source_url="https://instagram.com/reel/x",
        attempt=1,
        max_attempts=3,
        stuck_minutes=30,
    )

    text = tg.sent[0]["text"]
    assert "#11 (instagram)" in text
    assert "https://instagram.com/reel/x" in text
    assert "30 мин" in text
    assert "1/3" in text
    assert tg.sent[0]["chat_id"] == "-100123"


async def test_the_last_allowed_reclaim_says_so():
    tg = Telegram()

    await _notifier(tg).reclaimed(
        note_id=11,
        source_type="instagram",
        source_url=None,
        attempt=3,
        max_attempts=3,
        stuck_minutes=30,
    )

    assert "последняя попытка" in tg.sent[0]["text"]


async def test_a_note_with_no_url_still_gets_identified():
    tg = Telegram()

    await _notifier(tg).abandoned(
        note_id=5,
        source_type="voice",
        source_url=None,
        attempts=3,
        reason="воркер не отвечал",
    )

    text = tg.sent[0]["text"]
    assert "#5 (voice)" in text
    assert "воркер не отвечал" in text
    assert "переслать заново" in text


async def test_a_failed_note_carries_its_real_error():
    tg = Telegram()

    await _notifier(tg).failed(
        note_id=7,
        source_type="text",
        source_url=None,
        error="503 model not loaded",
    )

    assert "503 model not loaded" in tg.sent[0]["text"]


async def test_the_same_note_and_outcome_is_sent_once_per_interval():
    from datetime import timedelta

    tg = Telegram()
    n = _notifier(tg, min_interval=timedelta(hours=1))

    await n.abandoned(note_id=1, source_type="text", source_url=None, attempts=3, reason="x")
    await n.abandoned(note_id=1, source_type="text", source_url=None, attempts=3, reason="x")

    assert len(tg.sent) == 1


async def test_a_different_note_is_its_own_message():
    tg = Telegram()
    n = _notifier(tg)

    await n.abandoned(note_id=1, source_type="text", source_url=None, attempts=3, reason="x")
    await n.abandoned(note_id=2, source_type="text", source_url=None, attempts=3, reason="x")

    assert len(tg.sent) == 2


async def test_a_refusal_from_telegram_does_not_raise():
    tg = Telegram(status=400)

    await _notifier(tg).failed(note_id=1, source_type="text", source_url=None, error="x")


async def test_an_unreachable_telegram_does_not_raise():
    tg = Telegram(raises=httpx.ConnectError("no route"))

    await _notifier(tg).failed(note_id=1, source_type="text", source_url=None, error="x")


async def test_no_token_means_no_notifier_rather_than_a_crash():
    n = build_notifier(None, None, min_interval_minutes=5)
    assert isinstance(n, NullNotifier)


async def test_the_null_notifier_accepts_every_call():
    n = NullNotifier()
    await n.reclaimed(
        note_id=1, source_type="text", source_url=None, attempt=1, max_attempts=3, stuck_minutes=30
    )
    await n.abandoned(note_id=1, source_type="text", source_url=None, attempts=3, reason="x")
    await n.failed(note_id=1, source_type="text", source_url=None, error="x")
