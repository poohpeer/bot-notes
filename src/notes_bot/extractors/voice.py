"""`voice` extractor — see docs/architecture/03-ingest.md, "Голосовые
заметки" and ADR-13. Downloads through the Bot API (not the internet — no
SSRF fetcher involved), transcribes with the same faster-whisper model used
for Instagram audio.

The file never touches disk longer than the transcription itself needs:
written to a NamedTemporaryFile, always removed in `finally` — "Под может
перезапуститься" (03-ingest.md, "Ограничения тяжёлого пути") is the reason
cleanup can't wait for normal process exit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from notes_bot.clients.telegram_files import TelegramFileClient, TelegramFileError
from notes_bot.clients.transcribe import TranscriptionClient
from notes_bot.db.models import Note
from notes_bot.extractors.base import ExtractResult

log = logging.getLogger(__name__)


class VoiceExtractor:
    """Note.source_url holds the Telegram file_id for a voice note — there
    is no dedicated column for it, and source_url is otherwise unused for
    this source_type (voice files aren't fetched by URL).

    03-ingest.md, "Голосовые заметки": a caption on the voice message "тоже
    идёт в индекс" (also goes into the index) — folded into the returned
    text here, the same way YoutubeExtractor combines title/description/
    transcript, so the orchestrator's extracted-text-wins-over-raw-text
    logic in queue/tasks.py stays uniform across every extractor.
    """

    source_type = "voice"
    queue = "heavy"

    def __init__(
        self, *, file_client: TelegramFileClient, transcription_client: TranscriptionClient
    ) -> None:
        self._file_client = file_client
        self._transcription_client = transcription_client

    async def extract(self, note: Note) -> ExtractResult:
        file_id = note.source_url
        if not file_id:
            return ExtractResult(text="", title=None, lang=None, meta={"error": "no file_id"})

        try:
            audio_bytes = await self._file_client.download(file_id)
        except TelegramFileError as exc:
            log.warning("voice extractor: note_id=%s download failed: %s", note.id, exc)
            return ExtractResult(text="", title=None, lang=None, meta={"error": str(exc)})

        fd, tmp_path = tempfile.mkstemp(suffix=".oga")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio_bytes)
            text = await asyncio.to_thread(self._transcription_client.transcribe, tmp_path)
        except Exception as exc:  # noqa: BLE001 — degrades, doesn't crash the job
            log.warning("voice extractor: note_id=%s transcription failed: %s", note.id, exc)
            return ExtractResult(text="", title=None, lang=None, meta={"error": str(exc)})
        finally:
            os.unlink(tmp_path)

        caption = (note.raw_text or "").strip()
        combined = "\n\n".join(p for p in (caption, text) if p)
        if not combined:
            return ExtractResult(
                text="",
                title=None,
                lang=None,
                meta={"error": "empty transcript and no caption"},
            )
        return ExtractResult(text=combined, title=None, lang=None, meta={})
