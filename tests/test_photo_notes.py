"""A photo or album without the "/events_table" caption becomes a note — see
bot/handlers.py, _flush_events_table_album / _save_photo_note, and
03-ingest.md, "Фото и скриншоты". The aiogram objects are stand-ins: only
the attributes the handler reads exist on them."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from notes_bot.bot import handlers
from notes_bot.bot.logic import SaveNoteResult

pytestmark = pytest.mark.asyncio

BOT_USERNAME = "notesbot"


def _photo_message(
    message_id,
    *,
    file_id="f",
    chat_type="private",
    caption=None,
    caption_entities=None,
    media_group_id=None,
    from_user=True,
):
    return SimpleNamespace(
        message_id=message_id,
        chat=SimpleNamespace(id=500, type=chat_type),
        from_user=SimpleNamespace(id=7) if from_user else None,
        caption=caption,
        caption_entities=caption_entities,
        photo=[SimpleNamespace(file_id=f"{file_id}-small"), SimpleNamespace(file_id=file_id)],
        media_group_id=media_group_id,
        answer=AsyncMock(),
        reply=AsyncMock(),
    )


def _mention(caption, handle=f"@{BOT_USERNAME}"):
    offset = len(caption[: caption.index(handle)].encode("utf-16-le")) // 2
    return [SimpleNamespace(type="mention", offset=offset, length=len(handle))]


def _deps():
    return SimpleNamespace(
        bot_username=BOT_USERNAME,
        settings=SimpleNamespace(telegram_bot_token="token", events_table_timeout_s=10.0),
        llm_queue=object(),
    )


@pytest.fixture
def saved(monkeypatch):
    """Replaces the DB-touching save with a recorder."""
    save = AsyncMock(
        return_value=SaveNoteResult(
            created=True, note_id=42, visibility="private", source_type="image"
        )
    )
    monkeypatch.setattr(handlers, "save_image_note", save)
    return save


def _album(*messages):
    pending = handlers._PendingAlbum()
    for message in messages:
        pending.file_ids.append(message.photo[-1].file_id)
        pending.any_message = message
        if pending.first_message is None or message.message_id < pending.first_message.message_id:
            pending.first_message = message
        if message.caption:
            pending.caption_message = message
    return pending


async def test_a_private_photo_without_a_caption_becomes_a_note(saved):
    message = _photo_message(10, file_id="AgACf1")

    await handlers._save_photo_note(_album(message), _deps())

    saved.assert_awaited_once()
    kwargs = saved.await_args.kwargs
    assert kwargs["file_ids"] == ["AgACf1"]
    assert kwargs["caption"] is None
    assert kwargs["is_group"] is False
    assert (kwargs["user_id"], kwargs["chat_id"], kwargs["tg_message_id"]) == (7, 500, 10)
    message.answer.assert_awaited_once()
    assert "Читаю изображение" in message.answer.await_args.args[0]
    assert message.answer.await_args.kwargs["reply_markup"] is not None


async def test_a_private_photo_keeps_its_caption(saved):
    message = _photo_message(10, caption="вот меню")

    await handlers._save_photo_note(_album(message), _deps())

    assert saved.await_args.kwargs["caption"] == "вот меню"


async def test_a_group_photo_without_the_mention_is_ignored(saved):
    message = _photo_message(10, chat_type="supergroup", caption="смотрите какой закат")

    await handlers._save_photo_note(_album(message), _deps())

    saved.assert_not_awaited()
    message.reply.assert_not_awaited()
    message.answer.assert_not_awaited()


async def test_a_group_photo_with_no_caption_at_all_is_ignored(saved):
    message = _photo_message(10, chat_type="group")

    await handlers._save_photo_note(_album(message), _deps())

    saved.assert_not_awaited()


async def test_a_group_photo_that_mentions_the_bot_becomes_a_note(saved):
    caption = "@notesbot запомни"
    message = _photo_message(
        10, chat_type="supergroup", caption=caption, caption_entities=_mention(caption)
    )

    await handlers._save_photo_note(_album(message), _deps())

    kwargs = saved.await_args.kwargs
    assert kwargs["is_group"] is True
    assert kwargs["caption"] == caption
    message.reply.assert_awaited_once()
    message.answer.assert_not_awaited()


async def test_a_mention_after_an_emoji_in_the_caption_still_counts(saved):
    caption = "🎉 @notesbot запомни"
    message = _photo_message(
        10, chat_type="supergroup", caption=caption, caption_entities=_mention(caption)
    )

    await handlers._save_photo_note(_album(message), _deps())

    saved.assert_awaited_once()


async def test_a_group_photo_mentioning_another_bot_is_ignored(saved):
    caption = "@otherbot запомни"
    message = _photo_message(
        10,
        chat_type="supergroup",
        caption=caption,
        caption_entities=_mention(caption, "@otherbot"),
    )

    await handlers._save_photo_note(_album(message), _deps())

    saved.assert_not_awaited()


async def test_an_uncaptioned_album_is_one_note_anchored_on_its_first_photo(saved):
    first = _photo_message(21, file_id="f1", media_group_id="g")
    second = _photo_message(22, file_id="f2", media_group_id="g")
    third = _photo_message(23, file_id="f3", media_group_id="g")

    # Arrival order must not matter for the idempotency key.
    await handlers._save_photo_note(_album(second, first, third), _deps())

    kwargs = saved.await_args.kwargs
    assert kwargs["tg_message_id"] == 21
    assert sorted(kwargs["file_ids"]) == ["f1", "f2", "f3"]
    first.answer.assert_awaited_once()
    second.answer.assert_not_awaited()


async def test_a_captioned_album_is_anchored_on_the_message_that_carries_the_caption(saved):
    first = _photo_message(21, file_id="f1", media_group_id="g")
    second = _photo_message(22, file_id="f2", media_group_id="g", caption="меню")

    await handlers._save_photo_note(_album(first, second), _deps())

    kwargs = saved.await_args.kwargs
    assert kwargs["tg_message_id"] == 22
    assert kwargs["caption"] == "меню"


async def test_a_redelivered_photo_is_silent(saved):
    saved.return_value = SaveNoteResult(created=False, note_id=None, visibility=None)
    message = _photo_message(10)

    await handlers._save_photo_note(_album(message), _deps())

    message.answer.assert_not_awaited()
    message.reply.assert_not_awaited()


async def test_a_photo_with_no_sender_is_ignored(saved):
    message = _photo_message(10, from_user=False)

    await handlers._save_photo_note(_album(message), _deps())

    saved.assert_not_awaited()


# --- the real entry point: buffering + the decision after the debounce ------


@pytest.fixture
def fast_debounce(monkeypatch):
    monkeypatch.setattr(handlers, "_ALBUM_DEBOUNCE_S", 0.02)
    handlers._album_buffers.clear()
    handlers._album_tasks.clear()
    yield
    handlers._album_buffers.clear()
    handlers._album_tasks.clear()


async def test_an_album_arriving_as_separate_messages_is_saved_once(saved, fast_debounce):
    deps = _deps()
    for message_id, file_id in ((31, "f1"), (32, "f2"), (33, "f3")):
        await handlers._buffer_events_table_photo(
            _photo_message(message_id, file_id=file_id, media_group_id="g"), deps
        )

    await asyncio.sleep(0.15)

    saved.assert_awaited_once()
    assert saved.await_args.kwargs["file_ids"] == ["f1", "f2", "f3"]
    assert saved.await_args.kwargs["tg_message_id"] == 31


async def test_the_events_table_caption_still_starts_the_table_flow(
    saved, fast_debounce, monkeypatch
):
    downloads: list[str] = []

    class FakeFileClient:
        def __init__(self, token):
            pass

        async def download(self, file_id):
            downloads.append(file_id)
            return b"jpeg"

    enqueued: list[dict] = []
    monkeypatch.setattr(handlers, "TelegramFileClient", FakeFileClient)
    monkeypatch.setattr(
        handlers, "enqueue_parse_events_table", lambda queue, **kw: enqueued.append(kw)
    )
    message = _photo_message(40, file_id="t1", caption="/events_table математика")

    await handlers._buffer_events_table_photo(message, _deps())
    await asyncio.sleep(0.15)

    saved.assert_not_awaited()
    assert downloads == ["t1"]
    [job] = enqueued
    assert job["tab_name"] == "математика"
    assert job["chat_id"] == 500


async def test_a_private_photo_buffered_for_real_becomes_a_note(saved, fast_debounce):
    await handlers._buffer_events_table_photo(_photo_message(50, file_id="p1"), _deps())

    await asyncio.sleep(0.15)

    saved.assert_awaited_once()
    assert saved.await_args.kwargs["file_ids"] == ["p1"]
