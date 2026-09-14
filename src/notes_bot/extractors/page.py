"""`page` extractor — see docs/architecture/03-ingest.md, "Шаг 3. Экстракторы".

Fetches through the SSRF-safe fetcher (never a bare HTTP call) and hands the
HTML to trafilatura. Any failure — blocked address, unreachable host,
unparseable page, no extractable content — degrades to an empty result
rather than raising: the orchestrator (queue/tasks.py) falls back to
indexing raw_text (the URL itself), per "Деградация".
"""

from __future__ import annotations

import logging

import httpx
import trafilatura

from notes_bot.clients.fetch import SsrfBlocked, fetch
from notes_bot.db.models import Note
from notes_bot.extractors.base import ExtractResult

log = logging.getLogger(__name__)

_ALLOWED_CONTENT_TYPES = {"text/html", "text/plain"}


class PageExtractor:
    source_type = "page"
    queue = "fast"

    def __init__(
        self,
        *,
        max_body_bytes: int = 10_000_000,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._max_body_bytes = max_body_bytes
        self._transport = transport  # tests only

    async def extract(self, note: Note) -> ExtractResult:
        url = note.source_url
        if not url:
            return ExtractResult(text="", title=None, lang=None, meta={"error": "no source_url"})

        try:
            fetched = await fetch(
                url,
                allowed_content_types=_ALLOWED_CONTENT_TYPES,
                max_body_bytes=self._max_body_bytes,
                transport=self._transport,
            )
        except SsrfBlocked as exc:
            log.warning("page extractor: note_id=%s fetch blocked: %s", note.id, exc)
            return ExtractResult(text="", title=None, lang=None, meta={"error": str(exc)})

        html = fetched.body.decode("utf-8", errors="replace")
        try:
            data = trafilatura.bare_extraction(html, with_metadata=True, url=fetched.final_url)
        except Exception as exc:  # noqa: BLE001 — a parser failure degrades, doesn't crash the job
            log.warning("page extractor: note_id=%s trafilatura failed: %s", note.id, exc)
            return ExtractResult(text="", title=None, lang=None, meta={"error": str(exc)})

        if data is None or not data.text:
            return ExtractResult(
                text="",
                title=None,
                lang=None,
                meta={"error": "no extractable content", "final_url": fetched.final_url},
            )

        return ExtractResult(
            text=data.text,
            title=data.title,
            lang=data.language,
            meta={"final_url": fetched.final_url},
        )
