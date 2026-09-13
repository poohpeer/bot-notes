"""URL classification — see docs/architecture/03-ingest.md, "Шаг 1.
Классификация источника". Pure function, no I/O: the actual network call
only happens later, inside the matched extractor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

_URL_RE = re.compile(r"https?://\S+")

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
_INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com"}
# maps.google.* covers every ccTLD Google Maps has used (maps.google.com,
# maps.google.co.il, ...); the other two are the short-link forms.
_MAP_SHORT_HOSTS = {"maps.app.goo.gl"}


@dataclass(frozen=True)
class Classification:
    source_type: str
    source_url: str | None


def classify_text_message(text: str) -> Classification:
    """`voice` is a Telegram message *type*, decided by the bot layer before
    this ever runs — this only classifies the URL (if any) inside a text
    message. Extra prose alongside the URL is not this function's concern:
    03-ingest.md keeps both, URL in source_url and the whole message in
    raw_text — that split happens where the note is saved, not here.
    """
    match = _URL_RE.search(text)
    if not match:
        return Classification(source_type="text", source_url=None)

    url = match.group(0)
    host = (urlparse(url).hostname or "").lower()

    if host in _YOUTUBE_HOSTS:
        return Classification(source_type="youtube", source_url=url)
    if host in _INSTAGRAM_HOSTS:
        return Classification(source_type="instagram", source_url=url)
    if _is_map_link(url, host):
        return Classification(source_type="map", source_url=url)
    return Classification(source_type="page", source_url=url)


def _is_map_link(url: str, host: str) -> bool:
    if host in _MAP_SHORT_HOSTS:
        return True
    if host == "goo.gl" and urlparse(url).path.startswith("/maps"):
        return True
    return host.startswith("maps.google.")
