"""`map` extractor — see docs/architecture/03-ingest.md, "Шаг 3. Экстракторы".

Only resolves the redirect chain of a short map link (maps.app.goo.gl and
similar) and pulls a place name out of the final Google Maps URL — the page
itself is never scraped, so `fetch(..., read_body=False)` never buffers a
body nothing will use.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import unquote_plus

import httpx

from notes_bot.clients.fetch import SsrfBlocked, fetch
from notes_bot.db.models import Note
from notes_bot.extractors.base import ExtractResult

log = logging.getLogger(__name__)

# .../maps/place/<url-encoded name>/@lat,lng,zoom... — the shape every
# Google Maps place link resolves to, short link or not.
_PLACE_RE = re.compile(r"/maps/place/([^/]+)")


def _place_name_from_url(url: str) -> str | None:
    match = _PLACE_RE.search(url)
    if not match:
        return None
    return unquote_plus(match.group(1)).replace("+", " ")


class MapExtractor:
    source_type = "map"
    queue = "fast"

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport  # tests only

    async def extract(self, note: Note) -> ExtractResult:
        url = note.source_url
        if not url:
            return ExtractResult(text="", title=None, lang=None, meta={"error": "no source_url"})

        try:
            fetched = await fetch(url, read_body=False, transport=self._transport)
        except SsrfBlocked as exc:
            log.warning("map extractor: note_id=%s fetch blocked: %s", note.id, exc)
            return ExtractResult(text="", title=None, lang=None, meta={"error": str(exc)})

        name = _place_name_from_url(fetched.final_url)
        if not name:
            return ExtractResult(
                text="",
                title=None,
                lang=None,
                meta={"error": "could not parse a place name", "final_url": fetched.final_url},
            )

        return ExtractResult(
            text=name, title=name, lang=None, meta={"final_url": fetched.final_url}
        )
