from __future__ import annotations

import json

import httpx
import pytest

from notes_bot.clients.translate import HttpTranslateClient, TranslateServiceError


def _client(handler) -> HttpTranslateClient:
    return HttpTranslateClient(
        "http://notes-translate:8000",
        target_lang="eng_Latn",
        transport=httpx.MockTransport(handler),
    )


async def test_translate_passages_sends_target_lang_and_returns_texts():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/translate"
        body = json.loads(request.read())
        assert body == {"texts": ["привет", "мир"], "target_lang": "eng_Latn"}
        return httpx.Response(
            200,
            json={
                "model": "nllb",
                "items": [
                    {"text": "hello", "source_lang": "ru", "translated": True},
                    {"text": "world", "source_lang": "ru", "translated": True},
                ],
            },
        )

    texts = await _client(handler).translate_passages(["привет", "мир"])
    assert texts == ["hello", "world"]


async def test_translate_query_returns_a_single_text():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert body == {"texts": ["q"], "target_lang": "eng_Latn"}
        return httpx.Response(
            200,
            json={
                "model": "nllb",
                "items": [{"text": "translated q", "source_lang": "ru", "translated": True}],
            },
        )

    text = await _client(handler).translate_query("q")
    assert text == "translated q"


async def test_non_200_raises_translate_service_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "model not loaded"})

    with pytest.raises(TranslateServiceError) as exc_info:
        await _client(handler).translate_query("q")
    assert exc_info.value.status_code == 503


async def test_connection_error_raises_translate_service_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TranslateServiceError):
        await _client(handler).translate_query("q")


async def test_non_json_error_body_falls_back_to_raw_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(TranslateServiceError, match="boom"):
        await _client(handler).translate_query("q")
