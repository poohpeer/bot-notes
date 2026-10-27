"""`page` extractor — see docs/architecture/03-ingest.md, "Шаг 3. Экстракторы".

Fetches through the SSRF-safe fetcher (never a bare HTTP call) and hands the
HTML to trafilatura. Any failure — blocked address, unreachable host,
unparseable page, no extractable content, a JavaScript/bot wall in place of
content — degrades to an empty result rather than raising: the orchestrator
(queue/tasks.py) falls back to indexing raw_text (the URL itself), per
"Деградация".
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

# What a page says instead of its content when it needs a JavaScript runtime
# or has decided it is talking to a bot: TripAdvisor answered note 62 with
# "Please enable JS and disable any ad blocker", and that sentence was stored
# as the note's text — indexed, searchable, and the only thing the note held.
# Matched only on short text (_WALL_MAX_CHARS): a real article that happens to
# discuss JavaScript or captchas is long, a wall is a sentence or two.
_WALL_MARKERS = (
    "enable js",
    "enable javascript",
    "javascript is disabled",
    "javascript is required",
    "javascript must be enabled",
    "javascript to be enabled",
    "включите javascript",
    "включите js",
    "требуется javascript",
    "checking your browser",
    "just a moment",
    "verify you are human",
    "are you a robot",
    "are you human",
    "access denied",
    "captcha",
)
_WALL_MAX_CHARS = 400


def looks_like_a_wall(text: str) -> bool:
    """True for the stub a page returns in place of its content — see
    _WALL_MARKERS. A heuristic, deliberately narrow: a false positive costs a
    note its page text (it degrades to the URL), a miss is today's behaviour."""
    stripped = text.strip()
    if not stripped or len(stripped) > _WALL_MAX_CHARS:
        return False
    lowered = stripped.lower()
    return any(marker in lowered for marker in _WALL_MARKERS)


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

        if looks_like_a_wall(data.text):
            # Not content: keeping it would index the wall's own sentence as
            # what the page says. `unreadable` lets the orchestrator tell the
            # sender, who can still send a screenshot of the page.
            log.warning("page extractor: note_id=%s got a JS/bot wall, not content", note.id)
            return ExtractResult(
                text="",
                title=None,
                lang=None,
                meta={
                    "error": "page needs JavaScript or blocks automated readers",
                    "unreadable": True,
                    "final_url": fetched.final_url,
                },
            )

        return ExtractResult(
            text=data.text,
            title=data.title,
            lang=data.language,
            meta={"final_url": fetched.final_url},
        )
