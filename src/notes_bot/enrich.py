"""Enrichment features over LLMClient — see docs/architecture/03-ingest.md,
"Шаг 6. Обогащение через ai-proxy", and 05-contracts.md's prompt
requirements (respond in the note's language, tags lowercase without '#',
return empty rather than invent).

Each function calls the LLM once with a fixed json_schema and degrades to
"nothing" (None / empty) on any failure or missing field. Enrichment never
blocks a note from being findable — that already happened in process_note;
this only adds polish on top, so `NullLLMClient` (LLM_ENABLED=false) and a
genuine ai-proxy failure take the exact same path through every function
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from notes_bot.clients.llm import LLMClient
from notes_bot.clients.prompts import load_prompt

# ~1500 tokens per 03-ingest.md's "Заголовок" row — a rough char budget so
# a long transcript doesn't blow past the model's context for tasks that
# don't need the whole thing anyway.
_HEAD_CHARS = 6000
_TITLE_MAX_CHARS = 60
_TAGS_MAX = 5

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": ["string", "null"]}},
    "required": ["summary"],
}
_ENRICH_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": ["string", "null"]},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "wrong": {"type": "string"},
                    "correct": {"type": "string"},
                },
                "required": ["wrong", "correct"],
            },
        },
        "place": {
            "type": ["object", "null"],
            "properties": {
                "name": {"type": ["string", "null"]},
                "district": {"type": ["string", "null"]},
                "cuisine": {"type": ["string", "null"]},
            },
        },
        "places": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "address": {"type": ["string", "null"]},
                    "location_hint": {"type": ["string", "null"]},
                },
            },
        },
    },
    "required": ["title", "tags", "summary"],
}
# Only these source_types plausibly carry ASR-transcribed speech — a typed
# text/page note has nothing for "corrections" to fix, so the field is
# dropped even if the model filled it anyway (defense in depth: the
# prompt's own "заполняй только применимое" instruction is the first line
# of defense, not the only one — see enrich()'s per-source_type gating
# below, same reasoning for `place`/`places`).
_ASR_SOURCE_TYPES = {"youtube", "instagram", "voice"}
_NOTE_TYPE_LABELS = {
    "map": "Тип: место на карте (ссылка Google Maps).",
    "youtube": "Тип: видео с YouTube — описание и/или расшифровка речи.",
    "instagram": "Тип: видео с Instagram — описание и/или расшифровка речи.",
    "voice": "Тип: голосовое сообщение — расшифровка речи (ASR).",
    "page": "Тип: заметка со страницы сайта.",
}
_DUPES_SCHEMA = {
    "type": "object",
    "properties": {
        "is_duplicate": {"type": "boolean"},
        "duplicate_note_id": {"type": ["integer", "null"]},
    },
    "required": ["is_duplicate"],
}


async def generate_summary(llm: LLMClient, text: str, *, timeout_s: float) -> str | None:
    """Also used standalone by /debug's own summary preview
    (queue/tasks.py's process_note_async) - kept separate from `enrich()`
    below rather than folded in, since that call happens synchronously
    inside process_note, before enrich_note (and its combined call) ever
    runs."""
    result = await llm.complete(
        system=load_prompt("summary"), user=text, json_schema=_SUMMARY_SCHEMA, timeout_s=timeout_s
    )
    if result.parsed is None:
        return None
    summary = result.parsed.get("summary")
    return summary.strip() if isinstance(summary, str) and summary.strip() else None


@dataclass(frozen=True)
class Correction:
    wrong: str
    correct: str


@dataclass(frozen=True)
class EnrichResult:
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    summary: str | None = None
    corrections: list[Correction] = field(default_factory=list)
    place: dict = field(default_factory=dict)
    places: list[dict] = field(default_factory=list)


async def enrich(llm: LLMClient, text: str, *, source_type: str, timeout_s: float) -> EnrichResult:
    """One combined call for title+tags+summary+corrections+place/places —
    see docs/architecture/03-ingest.md, "Обогащение": these used to be up
    to 4 separate ai-proxy calls (this codebase's earlier
    generate_title/generate_tags/generate_place/generate_places), each
    re-sending the same note text. Merging them cuts total enrich latency
    to roughly one call instead of the sum of several, and lets the model
    produce a *consistent* read of the note once instead of 4 independent
    ones. `find_duplicate` stays a separate call (enrich_note_async runs
    it in parallel via asyncio.gather) - it needs a fundamentally
    different input (candidate notes to compare against), not just this
    note's own text, so folding it in would mean stuffing other people's
    — well, the same owner's — note text into a "describe this note"
    prompt.

    `source_type` gates which of `corrections`/`place`/`places` are kept
    even if the model filled them anyway - defense in depth on top of the
    prompt's own "заполняй только применимое"."""
    label = _NOTE_TYPE_LABELS.get(source_type, "Тип: заметка.")
    result = await llm.complete(
        system=load_prompt("enrich"),
        user=f"{label}\n\n{text[:_HEAD_CHARS]}",
        json_schema=_ENRICH_SCHEMA,
        timeout_s=timeout_s,
    )
    if result.parsed is None:
        return EnrichResult()
    parsed = result.parsed

    title = parsed.get("title")
    title = title.strip()[:_TITLE_MAX_CHARS] if isinstance(title, str) and title.strip() else None

    tags_raw = parsed.get("tags")
    tags = (
        [t.strip().lstrip("#").lower() for t in tags_raw if isinstance(t, str) and t.strip()][
            :_TAGS_MAX
        ]
        if isinstance(tags_raw, list)
        else []
    )

    summary = parsed.get("summary")
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else None

    corrections: list[Correction] = []
    if source_type in _ASR_SOURCE_TYPES:
        for item in parsed.get("corrections") or []:
            if not isinstance(item, dict):
                continue
            wrong, correct = item.get("wrong"), item.get("correct")
            if (
                isinstance(wrong, str)
                and wrong.strip()
                and isinstance(correct, str)
                and correct.strip()
            ):
                corrections.append(Correction(wrong=wrong, correct=correct.strip()))

    place: dict = {}
    if source_type == "map":
        raw_place = parsed.get("place")
        if isinstance(raw_place, dict):
            place = {k: v for k, v in raw_place.items() if v}

    places: list[dict] = []
    if source_type in ("youtube", "instagram"):
        raw_places = parsed.get("places")
        if isinstance(raw_places, list):
            for item in raw_places:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                cleaned_place = {"name": name.strip()}
                address = item.get("address")
                if isinstance(address, str) and address.strip():
                    cleaned_place["address"] = address.strip()
                hint = item.get("location_hint")
                if isinstance(hint, str) and hint.strip():
                    cleaned_place["location_hint"] = hint.strip()
                places.append(cleaned_place)

    return EnrichResult(
        title=title, tags=tags, summary=summary, corrections=corrections, place=place, places=places
    )


def apply_corrections(text: str, corrections: list[Correction]) -> str:
    """Exact substring replace, one correction at a time - never a
    freeform rewrite: `enrich()`'s prompt asks the model for `wrong`
    exactly as written in the source text, precisely so this can be a
    plain `str.replace` instead of trusting the model to reproduce the
    whole text faithfully. A `wrong` that doesn't actually occur (the
    model misquoted it) is silently skipped, not an error - best-effort,
    same as everywhere else in this module."""
    for correction in corrections:
        if correction.wrong and correction.wrong in text:
            text = text.replace(correction.wrong, correction.correct)
    return text


async def find_duplicate(
    llm: LLMClient, *, new_text: str, candidates: list[tuple[int, str]], timeout_s: float
) -> int | None:
    """`candidates`: (note_id, chunk_text) pairs, already narrowed down by a
    cheap vector search among the same owner's notes — see
    03-ingest.md: "Прогонять через LLM все заметки нереально"."""
    if not candidates:
        return None
    candidate_ids = {note_id for note_id, _ in candidates}
    candidates_block = "\n\n".join(
        f"Кандидат #{note_id}:\n{snippet}" for note_id, snippet in candidates
    )
    user = f"Новая заметка:\n{new_text[:2000]}\n\n{candidates_block}"

    result = await llm.complete(
        system=load_prompt("dupes"), user=user, json_schema=_DUPES_SCHEMA, timeout_s=timeout_s
    )
    if result.parsed is None or not result.parsed.get("is_duplicate"):
        return None

    duplicate_id = result.parsed.get("duplicate_note_id")
    if not isinstance(duplicate_id, int) or duplicate_id not in candidate_ids:
        # Not offered as a candidate — the model invented an id rather than
        # picking one of the ones actually given. Treat as no match.
        return None
    return duplicate_id
