"""URL classification — see docs/architecture/03-ingest.md, "Шаг 1.
Классификация источника". Pure function, no I/O: the actual network call
only happens later, inside the matched extractor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

_URL_RE = re.compile(r"https?://\S+")

# `\S+` above is greedy and stops only at whitespace, so a URL quoted or
# followed by punctuation in prose — "https://google.com", (see
# https://x.com/y) — pulls that character in too. A real production
# example: iOS/macOS autocorrect turns a straight quote into a curly one
# (”), which then makes it into urlparse's hostname and breaks DNS
# resolution outright ("Invalid IDNA hostname"). Stripped after matching,
# not folded into the regex, so this can stay a plain character class
# instead of a lookahead.
_TRAILING_PUNCTUATION = ".,;:!?\"'“”‘’«»\\]}>"

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

    url = _strip_trailing_punctuation(match.group(0))
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


def _strip_trailing_punctuation(url: str) -> str:
    while url and url[-1] in _TRAILING_PUNCTUATION:
        url = url[:-1]
    # ')' is handled separately from the plain character class above: a
    # trailing ')' is only noise if it isn't balanced by a '(' earlier in
    # the URL itself — e.g. a bare wiki link
    # (https://en.wikipedia.org/wiki/Foo_(bar)) has a real one.
    while url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    return url
