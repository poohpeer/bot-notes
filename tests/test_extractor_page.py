from __future__ import annotations

import socket

import httpx
import pytest

from notes_bot.db.models import Note
from notes_bot.extractors.page import PageExtractor

pytestmark = pytest.mark.asyncio

HTML = """<!DOCTYPE html><html lang="ru"><head>
<title>Хинкальная на Руставели — обзор</title></head><body>
<h1>Хинкальная на Руставели</h1>
<p>Отличное место для хинкали в центре города. Рекомендую попробовать хинкали
с мясом и сыром, а также хачапури по-аджарски.</p>
<p>Работает допоздна, есть доставка.</p>
</body></html>"""


def _note(url: str) -> Note:
    return Note(
        id=1,
        user_id=1,
        chat_id=1,
        is_group=False,
        visibility="private",
        source_type="page",
        source_url=url,
    )


@pytest.fixture
def dns(monkeypatch):
    mapping = {"good.example": "93.184.216.34", "evil.example": "127.0.0.1"}

    def _get(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _get)
    return mapping


async def test_extracts_title_and_main_text(dns):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=HTML.encode())

    extractor = PageExtractor(transport=httpx.MockTransport(handler))
    result = await extractor.extract(_note("http://good.example/page"))

    assert result.title == "Хинкальная на Руставели"
    assert "Отличное место для хинкали" in result.text
    # Boilerplate/navigation isn't part of this page, but the point stands:
    # only the article content is returned, not the raw HTML.
    assert "<html" not in result.text


async def test_ssrf_blocked_address_degrades_to_empty_result(dns):
    """Never raises — 03-ingest.md's degradation rule is the caller's to
    apply, not the extractor's to enforce by crashing the job."""
    extractor = PageExtractor()
    result = await extractor.extract(_note("http://evil.example/"))
    assert result.text == ""
    assert "error" in result.meta


async def test_empty_page_degrades_to_empty_result(dns):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html></html>")

    extractor = PageExtractor(transport=httpx.MockTransport(handler))
    result = await extractor.extract(_note("http://good.example/"))
    assert result.text == ""
    assert result.meta["error"]


async def test_missing_source_url_degrades_to_empty_result(dns):
    extractor = PageExtractor()
    result = await extractor.extract(_note(None))
    assert result.text == ""
    assert result.meta["error"] == "no source_url"


async def test_non_html_content_type_is_blocked(dns):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")

    extractor = PageExtractor(transport=httpx.MockTransport(handler))
    result = await extractor.extract(_note("http://good.example/file.pdf"))
    assert result.text == ""
    assert "content-type" in result.meta["error"]
