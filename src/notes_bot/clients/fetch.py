"""SSRF-safe HTTP fetcher — see docs/architecture/03-ingest.md, "SSRF-safe
фетчер", and ADR-7. Every outgoing request for a user-supplied URL (the
`page` extractor and the map resolver) goes through this, never a bare
httpx call.

The user dictates the URL and this bot runs inside the cluster next to
Postgres and Redis — without these checks it is a ready-made port scanner
and cloud-metadata reader for anyone who sends it a link.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass

import httpx

_ALLOWED_SCHEMES = {"http", "https"}
_MAX_REDIRECTS = 5


class SsrfBlocked(Exception):
    """The URL, or something it redirected to, resolves to a blocked address
    or uses a scheme/content-type this fetcher refuses."""


@dataclass(frozen=True)
class FetchResult:
    final_url: str
    content_type: str | None
    body: bytes


def _is_blocked_ip(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    # is_private already covers RFC1918 (IPv4) and unique-local (IPv6
    # fc00::/7); is_link_local covers 169.254.0.0/16 — including the cloud
    # metadata address 169.254.169.254 — and IPv6 link-local. is_loopback
    # covers 127.0.0.0/8 and ::1. is_reserved and is_multicast round out
    # ranges that were never meant to be dialed as a normal HTTP peer.
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


async def _resolve_and_pin(host: str) -> str:
    """Resolves `host` and returns one IP, only after confirming every
    address it resolves to is safe.

    Checking *every* resolved address, not just the one that gets used,
    matters: a multi-homed DNS answer where only some addresses are
    internal would otherwise pass by chance depending on resolution order.
    Returning a single pinned IP (rather than re-resolving at connect time)
    is what closes the DNS-rebinding gap described in 03-ingest.md — the
    name could be repointed at an internal address the instant after this
    check if the actual TCP connection resolved it again.
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.run_in_executor(None, socket.getaddrinfo, host, None)
    except socket.gaierror as exc:
        raise SsrfBlocked(f"could not resolve {host}: {exc}") from exc

    ips = {info[4][0] for info in infos}
    if not ips:
        raise SsrfBlocked(f"{host} resolved to no addresses")
    for ip in ips:
        if _is_blocked_ip(ip):
            raise SsrfBlocked(f"{host} resolves to a blocked address ({ip})")
    return next(iter(ips))


async def fetch(
    url: str,
    *,
    allowed_content_types: set[str] | None = None,
    max_body_bytes: int = 10_000_000,
    connect_timeout_s: float = 5.0,
    total_timeout_s: float = 20.0,
    method: str = "GET",
    read_body: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FetchResult:
    """Fetches `url`, following up to `_MAX_REDIRECTS` redirects, each
    re-validated from scratch — a redirect is just as capable of pointing
    somewhere internal as the original URL.

    `transport` is for tests (httpx.MockTransport); production never passes
    it, so requests always go over the real network to the pinned IP.
    """
    timeout = httpx.Timeout(total_timeout_s, connect=connect_timeout_s)
    current = url
    redirects = 0

    async with httpx.AsyncClient(
        timeout=timeout, transport=transport, follow_redirects=False
    ) as client:
        while True:
            parsed = httpx.URL(current)
            if parsed.scheme not in _ALLOWED_SCHEMES:
                raise SsrfBlocked(f"scheme {parsed.scheme!r} is not allowed")

            ip = await _resolve_and_pin(parsed.host)
            pinned = parsed.copy_with(host=ip)
            headers = {"Host": parsed.host}
            extensions = {"sni_hostname": parsed.host} if parsed.scheme == "https" else {}

            # stream=True: client.request() reads the whole body eagerly
            # before returning, which would download an unbounded body
            # before the size check below ever runs. Streaming is what lets
            # that check actually abort mid-transfer.
            request = client.build_request(method, pinned, headers=headers, extensions=extensions)
            response = await client.send(request, stream=True)

            if response.is_redirect:
                redirects += 1
                if redirects > _MAX_REDIRECTS:
                    raise SsrfBlocked(f"too many redirects (>{_MAX_REDIRECTS})")
                location = response.headers.get("location")
                if not location:
                    raise SsrfBlocked("redirect with no Location header")
                # Resolve relative to the URL as the user gave it, not the
                # pinned IP — the redirect target is a hostname, not an IP.
                current = str(parsed.join(location))
                await response.aclose()
                continue

            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if allowed_content_types is not None and content_type not in allowed_content_types:
                await response.aclose()
                raise SsrfBlocked(f"content-type {content_type!r} is not allowed")

            body = bytearray()
            if read_body:
                # The map resolver only wants the final URL after redirects
                # (03-ingest.md: "HTTP HEAD/GET без тела. ... Страница не
                # скрапится") — read_body=False skips buffering a body
                # nothing will use.
                async for piece in response.aiter_bytes():
                    body.extend(piece)
                    if len(body) > max_body_bytes:
                        await response.aclose()
                        raise SsrfBlocked(f"body exceeds max_body_bytes={max_body_bytes}")
            await response.aclose()

            return FetchResult(
                final_url=str(parsed), content_type=content_type or None, body=bytes(body)
            )
