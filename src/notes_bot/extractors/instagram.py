"""`instagram` extractor — see docs/architecture/03-ingest.md, "Шаг 3.
Экстракторы": caption, then the audio track, then a transcript;
concatenation of both.

Downloads real media (unlike the youtube extractor's metadata-only calls),
so this is `heavy`-queue only — it needs ffmpeg and faster-whisper, which
only `notes-app-heavy` carries. The downloaded file is always removed in
`finally`, same reasoning as the voice extractor: a pod can restart mid-job.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Protocol

from notes_bot.clients.transcribe import TranscriptionClient
from notes_bot.db.models import Note
from notes_bot.extractors.base import ExtractResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedMedia:
    caption: str | None
    audio_path: str | None
    duration: float | None


class Downloader(Protocol):
    def download(self, url: str, *, max_bytes: int) -> DownloadedMedia: ...


class InstagramExtractor:
    source_type = "instagram"
    queue = "heavy"

    def __init__(
        self,
        *,
        transcription_client: TranscriptionClient,
        downloader: Downloader | None = None,
        max_download_bytes: int = 104_857_600,
        max_audio_seconds: int = 1200,
    ) -> None:
        self._transcription_client = transcription_client
        self._downloader = downloader or _YtDlpDownloader()
        self._max_download_bytes = max_download_bytes
        self._max_audio_seconds = max_audio_seconds

    async def extract(self, note: Note) -> ExtractResult:
        url = note.source_url
        if not url:
            return ExtractResult(text="", title=None, lang=None, meta={"error": "no source_url"})

        try:
            media = await asyncio.to_thread(
                self._downloader.download, url, max_bytes=self._max_download_bytes
            )
        except Exception as exc:  # noqa: BLE001 — degrades, doesn't crash the job
            log.warning("instagram extractor: note_id=%s download failed: %s", note.id, exc)
            return ExtractResult(text="", title=None, lang=None, meta={"error": str(exc)})

        transcript = ""
        transcription_skipped = None
        try:
            if media.audio_path and media.duration and media.duration > self._max_audio_seconds:
                # 03-ingest.md's second-choice fallback for R2 (Whisper too
                # slow): skip transcription for long posts rather than tying
                # up the heavy queue for the length of the task timeout.
                transcription_skipped = "duration exceeds max_audio_seconds"
                log.info(
                    "instagram extractor: note_id=%s skipping transcription (%ss > %ss)",
                    note.id,
                    media.duration,
                    self._max_audio_seconds,
                )
            elif media.audio_path:
                transcript = await asyncio.to_thread(
                    self._transcription_client.transcribe, media.audio_path
                )
        except Exception as exc:  # noqa: BLE001 — the caption alone still has value
            log.warning("instagram extractor: note_id=%s transcription failed: %s", note.id, exc)
        finally:
            if media.audio_path:
                # Removes the whole scratch directory _YtDlpDownloader made
                # for this download, not just the final .wav — yt-dlp's own
                # pre-postprocessing file can be left behind alongside it.
                shutil_rmtree_quietly(os.path.dirname(media.audio_path))

        caption = (media.caption or "").strip()
        combined = "\n\n".join(p for p in (caption, transcript) if p)
        if not combined:
            return ExtractResult(
                text="",
                title=None,
                lang=None,
                meta={"error": "no caption and no transcript"},
            )

        meta = {"duration": media.duration}
        if transcription_skipped:
            meta["transcription_skipped"] = transcription_skipped
        return ExtractResult(text=combined, title=None, lang=None, meta=meta)


class _YtDlpDownloader:
    def download(self, url: str, *, max_bytes: int) -> DownloadedMedia:
        import yt_dlp

        tmp_dir = tempfile.mkdtemp()
        try:
            opts = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
                "max_filesize": max_bytes,
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)

            # FFmpegExtractAudio renames the downloaded file's extension to
            # the target codec; prepare_filename() reflects the pre-
            # postprocessing name, not the post-conversion one.
            base, _ext = os.path.splitext(ydl.prepare_filename(info))
            audio_path = base + ".wav"
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"expected extracted audio at {audio_path}")

            return DownloadedMedia(
                caption=info.get("description"),
                audio_path=audio_path,
                duration=info.get("duration"),
            )
        except Exception:
            shutil_rmtree_quietly(tmp_dir)
            raise


def shutil_rmtree_quietly(path: str) -> None:
    import shutil

    with contextlib.suppress(OSError):
        shutil.rmtree(path)
