from __future__ import annotations

import httpx
import pytest

from notes_bot.clients.telegram_files import TelegramFileClient, TelegramFileError


def _client(handler, **kw) -> TelegramFileClient:
    return TelegramFileClient("test-token", transport=httpx.MockTransport(handler), **kw)


async def test_downloads_a_small_file():
    def handler(request: httpx.Request) -> httpx.Response:
        if "getFile" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "result": {"file_id": "abc", "file_size": 5, "file_path": "voice/abc.oga"},
                },
            )
        return httpx.Response(200, content=b"hello")

    result = await _client(handler).download("abc")
    assert result == b"hello"


async def test_uses_the_bot_token_and_file_path_in_the_download_url():
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if "getFile" in str(request.url):
            return httpx.Response(
                200,
                json={"ok": True, "result": {"file_path": "voice/xyz.oga", "file_size": 3}},
            )
        return httpx.Response(200, content=b"abc")

    await _client(handler).download("abc")
    assert "getFile" in seen_urls[0]
    assert seen_urls[1].endswith("/file/bottest-token/voice/xyz.oga")


async def test_rejects_a_file_reported_over_the_limit_before_downloading():
    downloaded = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal downloaded
        if "getFile" in str(request.url):
            return httpx.Response(
                200,
                json={"ok": True, "result": {"file_path": "voice/big.oga", "file_size": 999}},
            )
        downloaded = True
        return httpx.Response(200, content=b"x" * 999)

    with pytest.raises(TelegramFileError, match="Bot API download limit"):
        await _client(handler, max_file_bytes=100).download("abc")
    assert downloaded is False


async def test_aborts_mid_download_if_the_body_exceeds_the_limit_regardless_of_file_size():
    """Defense in depth: getFile's file_size shouldn't be the only thing
    standing between this and an unbounded download."""

    async def _gen():
        for _ in range(100):
            yield b"x" * 50

    def handler(request: httpx.Request) -> httpx.Response:
        if "getFile" in str(request.url):
            # No file_size in the response at all.
            return httpx.Response(200, json={"ok": True, "result": {"file_path": "voice/x.oga"}})
        return httpx.Response(200, content=_gen())

    with pytest.raises(TelegramFileError, match="exceeds"):
        await _client(handler, max_file_bytes=100).download("abc")


async def test_getfile_not_ok_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "file not found"})

    with pytest.raises(TelegramFileError, match="file not found"):
        await _client(handler).download("missing")


async def test_getfile_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(TelegramFileError, match="404"):
        await _client(handler).download("abc")


async def test_download_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if "getFile" in str(request.url):
            return httpx.Response(
                200, json={"ok": True, "result": {"file_path": "voice/x.oga", "file_size": 3}}
            )
        return httpx.Response(500)

    with pytest.raises(TelegramFileError, match="500"):
        await _client(handler).download("abc")
