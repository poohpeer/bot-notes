"""`/events_table` — parse a screenshot of a table (exam schedules,
holidays, etc.) into individual events via a vision-capable LLM call. See
docs/architecture/03-ingest.md, "/events_table".

This is deliberately separate from enrich.py: enrich.py's functions all
take a note's own text and run over `LLMClient`'s plain text+json_schema
path; extraction here is the one place `images` (LLMClient's own
`ImageInput` list) matters, and it never touches an existing note - this
produces the *input* to several new notes, not an addition to one that
already exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from notes_bot.clients.llm import ImageInput, LLMClient
from notes_bot.clients.prompts import load_prompt

_EVENT_TYPES = {"exam", "holiday", "event"}

_EVENTS_TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_tags": {"type": "array", "items": {"type": "string"}},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date_start": {"type": ["string", "null"]},
                    "date_end": {"type": ["string", "null"]},
                    "type": {"type": "string", "enum": sorted(_EVENT_TYPES)},
                    "text": {"type": ["string", "null"]},
                    "subjects": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    "required": ["topic_tags", "events"],
}

_TYPE_TAG_RU = {"exam": "экзамен", "holiday": "праздник", "event": "мероприятие"}


@dataclass(frozen=True)
class ExtractedEvent:
    date_start: str
    date_end: str
    type: str
    text: str
    subjects: list[str] = field(default_factory=list)

    @property
    def tag(self) -> str:
        return _TYPE_TAG_RU[self.type]

    @property
    def note_text(self) -> str:
        date_range = (
            self.date_start
            if self.date_end == self.date_start
            else (f"{self.date_start}–{self.date_end}")
        )
        return f"{date_range}: {self.text}"


@dataclass(frozen=True)
class ExtractedTable:
    topic_tags: list[str]
    events: list[ExtractedEvent]


async def extract_events_table(
    llm: LLMClient, *, images: list[ImageInput], tab_name: str, timeout_s: float
) -> ExtractedTable:
    """Best-effort, same as every enrich.py function — a malformed or
    missing field is dropped, not raised; a garbled reply degrades to
    fewer events found, not a failed command.

    `tab_name` is optional (handlers.py's own docstring on why) — ai-proxy's
    `prompt` field has a `min_length` of 1, so an empty tab name still needs
    *some* non-empty instruction, not just "" or "Таб: "."""
    result = await llm.complete(
        system=load_prompt("events_table"),
        user=f"Таб: {tab_name}" if tab_name else "Разбери таблицу на скриншоте(ах).",
        json_schema=_EVENTS_TABLE_SCHEMA,
        images=images,
        timeout_s=timeout_s,
    )
    if result.parsed is None:
        return ExtractedTable(topic_tags=[], events=[])

    topic_tags = result.parsed.get("topic_tags")
    tags = (
        [t.strip().lower() for t in topic_tags if isinstance(t, str) and t.strip()]
        if isinstance(topic_tags, list)
        else []
    )

    raw_events = result.parsed.get("events")
    events: list[ExtractedEvent] = []
    if isinstance(raw_events, list):
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            date_start = item.get("date_start")
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(date_start, str) or not date_start.strip():
                continue
            date_end = item.get("date_end")
            event_type = item.get("type")
            raw_subjects = item.get("subjects")
            subjects = (
                [s.strip().lower() for s in raw_subjects if isinstance(s, str) and s.strip()]
                if isinstance(raw_subjects, list)
                else []
            )
            events.append(
                ExtractedEvent(
                    date_start=date_start.strip(),
                    date_end=(
                        date_end.strip()
                        if isinstance(date_end, str) and date_end.strip()
                        else date_start.strip()
                    ),
                    type=event_type if event_type in _EVENT_TYPES else "event",
                    text=text.strip(),
                    subjects=subjects,
                )
            )

    return ExtractedTable(topic_tags=tags, events=events)
