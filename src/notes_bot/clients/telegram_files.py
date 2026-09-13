"""Downloads a file from Telegram's Bot API — used by the `voice` extractor.
See docs/architecture/03-ingest.md, "Ограничения тяжёлого пути": a hard 20MB
cap, which is also the Bot API's own download limit, not a choice this bot
makes.

Not the SSRF-safe fetcher (clients/fetch.py): the target here is always
api.telegram.org, never a user-supplied host, so that fetcher's
DNS-pinning/blocklist machinery is solving a problem that doesn't exist on
this path.
"""

from __future__ import annotations

import httpx

_DEFAULT_MAX_FILE_BYTES = 20_000_000


class TelegramFileError(Exception):
    """Covers both a file that's too large and any other Bot API failure —
    callers degrade the note either way, so there's no reason to force two
    except clauses on them."""


class TelegramFileClient:
    def __init__(
        self,
        bot_token: str,
        *,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        api_base: str = "https://api.telegram.org",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._bot_token = bot_token
        self._max_file_bytes = max_file_bytes
        self._api_base = api_base.rstrip("/")
        self._transport = transport  # tests only
        self._timeout_s = timeout_s

    async def download(self, file_id: str) -> bytes:
        async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
            info = await self._get_file_info(client, file_id)

            file_size = info.get("file_size")
            if file_size is not None and file_size > self._max_file_bytes:
                raise TelegramFileError(
                    f"file {file_id} is {file_size} bytes, over the "
                    f"{self._max_file_bytes}-byte Bot API download limit"
                )

            file_path = info.get("file_path")
            if not file_path:
                raise TelegramFileError(f"getFile for {file_id} returned no file_path")

            url = f"{self._api_base}/file/bot{self._bot_token}/{file_path}"
            body = bytearray()
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise TelegramFileError(f"download failed with status {response.status_code}")
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    # Defense in depth: file_size from getFile should already
                    # have caught this, but a server that lies (or omits the
                    # field) shouldn't get an unbounded download anyway.
                    if len(body) > self._max_file_bytes:
                        raise TelegramFileError(f"download exceeds {self._max_file_bytes} bytes")
            return bytes(body)

    async def _get_file_info(self, client: httpx.AsyncClient, file_id: str) -> dict:
        url = f"{self._api_base}/bot{self._bot_token}/getFile"
        response = await client.get(url, params={"file_id": file_id})
        if response.status_code != 200:
            raise TelegramFileError(f"getFile failed with status {response.status_code}")
        data = response.json()
        if not data.get("ok"):
            raise TelegramFileError(f"getFile returned not ok: {data.get('description')}")
        return data["result"]
