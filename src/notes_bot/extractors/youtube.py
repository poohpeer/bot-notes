"""`youtube` extractor — see docs/architecture/03-ingest.md, "Шаг 3.
Экстракторы". Metadata-only: title/description via yt-dlp, subtitles (when
present) via youtube-transcript-api. No speech recognition — that only
happens for `voice` and `instagram` (M4), on the `heavy` queue.

Both yt-dlp and youtube-transcript-api are synchronous libraries that make
their own network calls; each call is wrapped in `asyncio.to_thread` so it
doesn't block the event loop. Real clients (`_YtDlpMetadataClient`,
`_YoutubeTranscriptApiClient`) are injected behind small Protocols so tests
run against fakes — this repo has no network access to verify against the
real services.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from notes_bot.db.models import Note
from notes_bot.extractors.base import ExtractResult

log = logging.getLogger(__name__)

_SHORTS_RE = re.compile(r"/shorts/([\w-]{6,})")


class MetadataClient(Protocol):
    def get_metadata(self, url: str) -> dict[str, Any]: ...


class TranscriptClient(Protocol):
    def get_transcript(self, video_id: str) -> list[dict[str, Any]] | None: ...


def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        return video_id or None

    query_v = parse_qs(parsed.query).get("v")
    if query_v:
        return query_v[0]

    match = _SHORTS_RE.search(parsed.path)
    if match:
        return match.group(1)

    return None


class YoutubeExtractor:
    source_type = "youtube"
    queue = "fast"

    def __init__(
        self,
        *,
        metadata_client: MetadataClient | None = None,
        transcript_client: TranscriptClient | None = None,
    ) -> None:
        self._metadata_client = metadata_client or _YtDlpMetadataClient()
        self._transcript_client = transcript_client or _YoutubeTranscriptApiClient()

    async def extract(self, note: Note) -> ExtractResult:
        url = note.source_url
        if not url:
            return ExtractResult(text="", title=None, lang=None, meta={"error": "no source_url"})

        video_id = extract_video_id(url)
        if not video_id:
            return ExtractResult(
                text="", title=None, lang=None, meta={"error": "could not parse video id"}
            )

        try:
            metadata = await asyncio.to_thread(self._metadata_client.get_metadata, url)
        except Exception as exc:  # noqa: BLE001 — degrades, doesn't crash the job
            log.warning("youtube extractor: note_id=%s metadata failed: %s", note.id, exc)
            return ExtractResult(text="", title=None, lang=None, meta={"error": str(exc)})

        title = metadata.get("title")
        description = metadata.get("description") or ""

        transcript_text = ""
        has_subtitles = False
        try:
            segments = await asyncio.to_thread(self._transcript_client.get_transcript, video_id)
        except Exception as exc:  # noqa: BLE001 — no transcript is normal, not fatal
            log.info("youtube extractor: note_id=%s no transcript: %s", note.id, exc)
            segments = None
        if segments:
            has_subtitles = True
            transcript_text = " ".join(s["text"] for s in segments if s.get("text"))

        text = "\n\n".join(p for p in (title, description, transcript_text) if p)
        if not text:
            return ExtractResult(
                text="",
                title=None,
                lang=None,
                meta={"error": "no title, description, or transcript"},
            )

        return ExtractResult(
            text=text,
            title=title,
            lang=metadata.get("language"),
            meta={
                "has_subtitles": has_subtitles,
                "duration": metadata.get("duration"),
                "uploader": metadata.get("uploader"),
            },
        )


class _YtDlpMetadataClient:
    def get_metadata(self, url: str) -> dict[str, Any]:
        import yt_dlp

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)


class _YoutubeTranscriptApiClient:
    def get_transcript(self, video_id: str) -> list[dict[str, Any]] | None:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        try:
            # ru/en/he cover this bot's target languages (see
            # 05-contracts.md's prompt requirements); fetch() only accepts
            # an explicit priority list, not "any language".
            return api.fetch(video_id, languages=("ru", "en", "he")).to_raw_data()
        except Exception:
            # The video has captions, just not in any of those three —
            # take whichever one exists rather than nothing at all.
            transcripts = list(api.list(video_id))
            if not transcripts:
                raise
            return transcripts[0].fetch().to_raw_data()
