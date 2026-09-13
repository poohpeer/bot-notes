"""Shared extractor contract — see docs/architecture/03-ingest.md, "Шаг 3.
Экстракторы".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from notes_bot.db.models import Note


@dataclass(frozen=True)
class ExtractResult:
    text: str  # goes into extracted_text
    title: str | None  # technical name from the source, not an LLM title
    lang: str | None
    meta: dict = field(default_factory=dict)  # duration, author, has_subtitles, error, ...


class Extractor(Protocol):
    source_type: str
    queue: Literal["fast", "heavy"]

    async def extract(self, note: Note) -> ExtractResult: ...
