"""SSRF-fetcher tests — the DNS layer is mocked (deterministic, no real
network) while everything downstream (pinning, headers, SNI, redirects,
content-type, size limit) goes through the real fetch() logic against
httpx.MockTransport.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from notes_bot.clients.fetch import SsrfBlocked, fetch

pytestmark = pytest.mark.asyncio


def _fake_getaddrinfo(mapping: dict[str, str]):
    def _get(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror(f"no mock entry for {host}")
        ip = mapping[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    return _get


@pytest.fixture
def dns(monkeypatch):
    mapping: dict[str, str] = {}
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(mapping))
    return mapping


async def test_rejects_disallowed_scheme(dns):
    with pytest.raises(SsrfBlocked, match="scheme"):
        await fetch("ftp://example.com/file")


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "169.254.169.254",  # cloud metadata / link-local
        "169.254.1.1",  # link-local
        "10.0.0.5",  # RFC1918
        "172.16.0.1",  # RFC1918
        "192.168.1.1",  # RFC1918
        "0.0.0.0",  # unspecified
        "::1",  # IPv6 loopback
        "fd00::1",  # IPv6 unique-local
        "fe80::1",  # IPv6 link-local
    ],
)
async def test_blocks_every_disallowed_address_class(dns, ip):
    dns["evil.example"] = ip
    with pytest.raises(SsrfBlocked, match="blocked address"):
        await fetch("http://evil.example/")


async def test_dns_resolution_failure_is_blocked_not_a_crash(dns):
    with pytest.raises(SsrfBlocked, match="could not resolve"):
        await fetch("http://no-such-mock-entry.example/")


async def test_allows_a_public_address_and_returns_the_body(dns):
    dns["good.example"] = "93.184.216.34"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html></html>")

    result = await fetch("http://good.example/page", transport=httpx.MockTransport(handler))
    assert result.body == b"<html></html>"
    assert result.content_type == "text/html"


async def test_connects_to_the_pinned_ip_with_original_host_header(dns):
    dns["good.example"] = "93.184.216.34"
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host_in_url"] = request.url.host
        seen["host_header"] = request.headers.get("host")
        return httpx.Response(200, content=b"ok")

    await fetch("http://good.example/page", transport=httpx.MockTransport(handler))
    assert seen["host_in_url"] == "93.184.216.34"
    assert seen["host_header"] == "good.example"


async def test_https_sets_sni_hostname_to_the_original_host(dns):
    dns["good.example"] = "93.184.216.34"
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"ok")

    await fetch("https://good.example/page", transport=httpx.MockTransport(handler))
    assert seen["sni"] == "good.example"


async def test_redirect_is_followed_and_revalidated(dns):
    dns["start.example"] = "93.184.216.34"
    dns["end.example"] = "93.184.216.35"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            return httpx.Response(302, headers={"location": "http://end.example/final"})
        return httpx.Response(200, content=b"final page")

    result = await fetch("http://start.example/", transport=httpx.MockTransport(handler))
    assert result.body == b"final page"
    assert result.final_url == "http://end.example/final"


async def test_redirect_to_a_blocked_address_is_refused(dns):
    dns["start.example"] = "93.184.216.34"
    dns["internal.example"] = "10.0.0.1"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://internal.example/secret"})

    with pytest.raises(SsrfBlocked, match="blocked address"):
        await fetch("http://start.example/", transport=httpx.MockTransport(handler))


async def test_too_many_redirects_is_refused(dns):
    dns["loop.example"] = "93.184.216.34"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://loop.example/next"})

    with pytest.raises(SsrfBlocked, match="too many redirects"):
        await fetch("http://loop.example/", transport=httpx.MockTransport(handler))


async def test_relative_redirect_resolves_against_the_original_host(dns):
    dns["start.example"] = "93.184.216.34"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "/b"})
        return httpx.Response(200, content=b"page b")

    result = await fetch("http://start.example/a", transport=httpx.MockTransport(handler))
    assert result.body == b"page b"
    assert result.final_url == "http://start.example/b"


async def test_content_type_restriction_rejects_disallowed(dns):
    dns["good.example"] = "93.184.216.34"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/octet-stream"})

    with pytest.raises(SsrfBlocked, match="content-type"):
        await fetch(
            "http://good.example/",
            allowed_content_types={"text/html", "text/plain"},
            transport=httpx.MockTransport(handler),
        )


async def test_content_type_restriction_allows_listed_type(dns):
    dns["good.example"] = "93.184.216.34"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"hi")

    result = await fetch(
        "http://good.example/",
        allowed_content_types={"text/html", "text/plain"},
        transport=httpx.MockTransport(handler),
    )
    assert result.body == b"hi"


async def test_body_over_the_size_limit_is_refused(dns):
    dns["big.example"] = "93.184.216.34"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 1000)

    with pytest.raises(SsrfBlocked, match="max_body_bytes"):
        await fetch(
            "http://big.example/",
            max_body_bytes=100,
            transport=httpx.MockTransport(handler),
        )


async def test_size_limit_aborts_mid_stream_without_reading_everything(dns):
    """Doc requirement: streamed read with abort on excess, not a size
    check performed only after the whole body is already in memory."""
    dns["big.example"] = "93.184.216.34"
    chunks_yielded = 0

    async def _gen():
        nonlocal chunks_yielded
        for _ in range(1000):
            chunks_yielded += 1
            yield b"x" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_gen())

    with pytest.raises(SsrfBlocked, match="max_body_bytes"):
        await fetch(
            "http://big.example/",
            max_body_bytes=2000,
            transport=httpx.MockTransport(handler),
        )
    # 1000 chunks of 1000 bytes would be ~1MB; aborting after the limit
    # means only a handful were ever pulled from the generator.
    assert chunks_yielded < 10


async def test_read_body_false_skips_reading_the_body(dns):
    dns["good.example"] = "93.184.216.34"
    read_attempted = False

    async def _gen():
        nonlocal read_attempted
        read_attempted = True
        yield b"should not be read"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_gen())

    result = await fetch(
        "http://good.example/", read_body=False, transport=httpx.MockTransport(handler)
    )
    assert result.body == b""
    assert read_attempted is False
