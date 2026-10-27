from __future__ import annotations

import socket

import httpx
import pytest

from notes_bot.db.models import Note
from notes_bot.extractors.page import PageExtractor, looks_like_a_wall

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


def _serve(html: str) -> PageExtractor:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=html.encode())

    return PageExtractor(transport=httpx.MockTransport(handler))


async def test_a_javascript_wall_is_not_taken_for_the_pages_content(dns):
    """Live: TripAdvisor answered with this sentence and it was stored as the
    note's text (note 62)."""
    html = (
        "<html><head><title>tripadvisor.ru</title></head>"
        "<body><p>Please enable JS and disable any ad blocker</p></body></html>"
    )
    result = await _serve(html).extract(_note("http://good.example/restaurant"))

    assert result.text == ""
    assert result.meta["unreadable"] is True
    assert result.meta["error"]


async def test_a_real_page_is_not_flagged_as_a_wall(dns):
    result = await _serve(HTML).extract(_note("http://good.example/page"))

    assert "Отличное место для хинкали" in result.text
    assert "unreadable" not in result.meta


async def test_an_ordinary_degradation_is_not_marked_unreadable(dns):
    result = await _serve("<html></html>").extract(_note("http://good.example/"))

    assert result.meta["error"]
    assert "unreadable" not in result.meta


@pytest.mark.parametrize(
    "text",
    [
        "Please enable JS and disable any ad blocker",
        "Please enable JavaScript to continue.",
        "Just a moment...",
        "Checking your browser before accessing example.com",
        "Verify you are human by completing the action below.",
        "Включите JavaScript, чтобы продолжить",
        "Access denied",
    ],
)
async def test_wall_texts_are_recognised(text):
    assert looks_like_a_wall(text) is True


async def test_a_long_article_that_mentions_a_captcha_is_not_a_wall():
    article = "Как работает captcha и зачем нужен JavaScript на сайте. " * 20
    assert len(article) > 400
    assert looks_like_a_wall(article) is False


async def test_short_ordinary_text_and_empty_text_are_not_walls():
    assert looks_like_a_wall("Хинкальная на Руставели, работает допоздна.") is False
    assert looks_like_a_wall("") is False
    assert looks_like_a_wall("   ") is False
