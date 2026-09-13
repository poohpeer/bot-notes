from __future__ import annotations

import socket
from urllib.parse import quote

import httpx
import pytest

from notes_bot.db.models import Note
from notes_bot.extractors.map import MapExtractor, _place_name_from_url

# asyncio_mode = "auto" (pyproject.toml) picks up async def tests on its
# own — no module-wide marker, since this file also has plain sync tests.


def _note(url: str | None) -> Note:
    return Note(
        id=1,
        user_id=1,
        chat_id=1,
        is_group=False,
        visibility="private",
        source_type="map",
        source_url=url,
    )


@pytest.fixture
def dns(monkeypatch):
    mapping = {
        "maps.app.goo.gl": "93.184.216.34",
        "www.google.com": "93.184.216.35",
        "evil.example": "127.0.0.1",
    }

    def _get(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _get)
    return mapping


def test_place_name_parsed_from_full_maps_url():
    url = "https://www.google.com/maps/place/Хинкальная+на+Руставели/@41.7,44.8,17z"
    assert _place_name_from_url(url) == "Хинкальная на Руставели"


def test_place_name_missing_when_url_has_no_place_segment():
    assert _place_name_from_url("https://www.google.com/maps/@41.7,44.8,17z") is None


async def test_resolves_short_link_and_extracts_place_name(dns):
    # A real Location header is percent-encoded (raw non-ASCII isn't a
    # legal header value) — quote it the way a server actually would.
    encoded_name = quote("Хинкальная на Руставели").replace("%20", "+")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.35":  # already at the resolved google.com
            return httpx.Response(200)
        return httpx.Response(
            302,
            headers={
                "location": f"https://www.google.com/maps/place/{encoded_name}/@41.7,44.8,17z"
            },
        )

    extractor = MapExtractor(transport=httpx.MockTransport(handler))
    result = await extractor.extract(_note("https://maps.app.goo.gl/xyz"))

    assert result.text == "Хинкальная на Руставели"
    assert result.title == "Хинкальная на Руставели"


async def test_never_reads_the_page_body(dns):
    """03-ingest.md: "Страница не скрапится" — only the redirect chain and
    final URL matter."""
    read_attempted = False

    async def _gen():
        nonlocal read_attempted
        read_attempted = True
        yield b"should not be read"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_gen(),
        )

    # No /maps/place/ segment at the final URL in this test — the point is
    # only that the body generator is never touched, regardless of outcome.
    extractor = MapExtractor(transport=httpx.MockTransport(handler))
    await extractor.extract(_note("https://maps.app.goo.gl/xyz"))
    assert read_attempted is False


async def test_ssrf_blocked_address_degrades_to_empty_result(dns):
    extractor = MapExtractor()
    result = await extractor.extract(_note("http://evil.example/"))
    assert result.text == ""
    assert "error" in result.meta


async def test_missing_source_url_degrades_to_empty_result(dns):
    extractor = MapExtractor()
    result = await extractor.extract(_note(None))
    assert result.text == ""
    assert result.meta["error"] == "no source_url"


async def test_unparseable_final_url_degrades_to_empty_result(dns):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)  # no redirect, final URL has no /place/ segment

    extractor = MapExtractor(transport=httpx.MockTransport(handler))
    result = await extractor.extract(_note("https://maps.app.goo.gl/xyz"))
    assert result.text == ""
    assert "error" in result.meta
