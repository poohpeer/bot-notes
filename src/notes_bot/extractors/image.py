"""`image` extractor — see docs/architecture/03-ingest.md, "Фото и скриншоты".

A photo or screenshot sent without the "/events_table" caption. Downloads
through the Bot API (like `voice`, no SSRF fetcher involved) and reads it with
one vision call: the text visible on it, verbatim, plus a short description so
a picture with little or no text is still findable by what it shows.

`images` go to ai-proxy's `claude_code` provider — `codex` rejects them
(clients/llm.py, `ImageInput`) — which `ProxyAILLMClient` already arranges.
"""

from __future__ import annotations

import base64
import logging

from notes_bot.clients.llm import ImageInput, LLMClient, LLMServiceError
from notes_bot.clients.prompts import load_prompt
from notes_bot.clients.telegram_files import TelegramFileClient, TelegramFileError
from notes_bot.db.models import Note
from notes_bot.extractors.base import ExtractResult

log = logging.getLogger(__name__)

_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["text", "description"],
}

# ai-proxy's `prompt` field has a min_length of 1 (events_table.py hit the
# same), and the instructions live in the system prompt.
_USER_MESSAGE = "Прочитай изображение(я)."


def split_file_ids(source_url: str | None) -> list[str]:
    """`Note.source_url` of an image note: one Telegram file_id per line (an
    album is one note with several). Shared with the bot layer, which builds
    the value, so the two ends cannot drift."""
    return [line.strip() for line in (source_url or "").split("\n") if line.strip()]


def join_file_ids(file_ids: list[str]) -> str:
    return "\n".join(file_ids)


class ImageExtractor:
    """`Note.source_url` holds the file_id(s) and `Note.raw_text` the caption,
    if any — the same split `voice` uses. The caption is folded into the
    returned text so the orchestrator's extracted-text-wins-over-raw-text rule
    (queue/tasks.py) stays uniform across every extractor."""

    source_type = "image"
    queue = "heavy"

    def __init__(
        self, *, file_client: TelegramFileClient, llm_client: LLMClient, timeout_s: float
    ) -> None:
        self._file_client = file_client
        self._llm_client = llm_client
        self._timeout_s = timeout_s

    async def extract(self, note: Note) -> ExtractResult:
        file_ids = split_file_ids(note.source_url)
        if not file_ids:
            return ExtractResult(text="", title=None, lang=None, meta={"error": "no file_id"})

        images: list[ImageInput] = []
        for file_id in file_ids:
            try:
                data = await self._file_client.download(file_id)
            except TelegramFileError as exc:
                # One lost frame of an album is not a reason to drop the rest.
                log.warning("image extractor: note_id=%s download failed: %s", note.id, exc)
                continue
            images.append(
                ImageInput(media_type="image/jpeg", data_base64=base64.b64encode(data).decode())
            )
        if not images:
            return ExtractResult(
                text=self._caption(note), title=None, lang=None, meta={"error": "download failed"}
            )

        try:
            result = await self._llm_client.complete(
                system=load_prompt("image_note"),
                user=_USER_MESSAGE,
                json_schema=_IMAGE_SCHEMA,
                images=images,
                timeout_s=self._timeout_s,
            )
        except LLMServiceError as exc:
            log.warning("image extractor: note_id=%s vision call failed: %s", note.id, exc)
            return ExtractResult(
                text=self._caption(note), title=None, lang=None, meta={"error": str(exc)}
            )

        if result.parsed is None:
            # LLM_ENABLED=false (NullLLMClient) and a reply that wasn't JSON
            # both land here — nothing was read off the picture.
            return ExtractResult(
                text=self._caption(note), title=None, lang=None, meta={"error": "no vision reply"}
            )

        description = self._clean(result.parsed.get("description"))
        visible_text = self._clean(result.parsed.get("text"))
        combined = "\n\n".join(p for p in (self._caption(note), description, visible_text) if p)
        if not combined:
            return ExtractResult(
                text="", title=None, lang=None, meta={"error": "nothing recognised on the image"}
            )
        return ExtractResult(text=combined, title=None, lang=None, meta={})

    @staticmethod
    def _caption(note: Note) -> str:
        return (note.raw_text or "").strip()

    @staticmethod
    def _clean(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""
