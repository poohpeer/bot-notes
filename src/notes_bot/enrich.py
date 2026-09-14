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

from notes_bot.clients.llm import LLMClient
from notes_bot.clients.prompts import load_prompt

# ~1500 tokens per 03-ingest.md's "Заголовок" row — a rough char budget so
# a long transcript doesn't blow past the model's context for tasks that
# don't need the whole thing anyway.
_HEAD_CHARS = 6000
_TITLE_MAX_CHARS = 60
_TAGS_MAX = 5

_TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": ["string", "null"]}},
    "required": ["title"],
}
_TAGS_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["tags"],
}
_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": ["string", "null"]}},
    "required": ["summary"],
}
_PLACE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": ["string", "null"]},
        "district": {"type": ["string", "null"]},
        "cuisine": {"type": ["string", "null"]},
    },
}
_PLACES_SCHEMA = {
    "type": "object",
    "properties": {
        "places": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "location_hint": {"type": ["string", "null"]},
                },
            },
        }
    },
    "required": ["places"],
}
_DUPES_SCHEMA = {
    "type": "object",
    "properties": {
        "is_duplicate": {"type": "boolean"},
        "duplicate_note_id": {"type": ["integer", "null"]},
    },
    "required": ["is_duplicate"],
}


async def generate_title(llm: LLMClient, text: str, *, timeout_s: float) -> str | None:
    result = await llm.complete(
        system=load_prompt("title"),
        user=text[:_HEAD_CHARS],
        json_schema=_TITLE_SCHEMA,
        timeout_s=timeout_s,
    )
    if result.parsed is None:
        return None
    title = result.parsed.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    return title.strip()[:_TITLE_MAX_CHARS]


async def generate_tags(llm: LLMClient, text: str, *, timeout_s: float) -> list[str]:
    result = await llm.complete(
        system=load_prompt("tags"),
        user=text[:_HEAD_CHARS],
        json_schema=_TAGS_SCHEMA,
        timeout_s=timeout_s,
    )
    if result.parsed is None:
        return []
    tags = result.parsed.get("tags")
    if not isinstance(tags, list):
        return []
    cleaned = [t.strip().lstrip("#").lower() for t in tags if isinstance(t, str) and t.strip()]
    return cleaned[:_TAGS_MAX]


async def generate_summary(llm: LLMClient, text: str, *, timeout_s: float) -> str | None:
    result = await llm.complete(
        system=load_prompt("summary"), user=text, json_schema=_SUMMARY_SCHEMA, timeout_s=timeout_s
    )
    if result.parsed is None:
        return None
    summary = result.parsed.get("summary")
    return summary.strip() if isinstance(summary, str) and summary.strip() else None


async def generate_place(llm: LLMClient, text: str, *, timeout_s: float) -> dict:
    result = await llm.complete(
        system=load_prompt("place"),
        user=text[:_HEAD_CHARS],
        json_schema=_PLACE_SCHEMA,
        timeout_s=timeout_s,
    )
    if result.parsed is None:
        return {}
    return {k: v for k, v in result.parsed.items() if v}


async def generate_places(llm: LLMClient, text: str, *, timeout_s: float) -> list[dict]:
    """For `youtube`/`instagram` notes: places mentioned in a video's
    caption/transcript, each `{"name": ..., "location_hint": ...}` — see
    04-search.md/03-ingest.md, "Места из видео". Unlike `generate_place`,
    a video can plausibly mention several places, not describe exactly one."""
    result = await llm.complete(
        system=load_prompt("places"),
        user=text[:_HEAD_CHARS],
        json_schema=_PLACES_SCHEMA,
        timeout_s=timeout_s,
    )
    if result.parsed is None:
        return []
    places = result.parsed.get("places")
    if not isinstance(places, list):
        return []
    cleaned = []
    for place in places:
        if not isinstance(place, dict):
            continue
        name = place.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        cleaned_place = {"name": name.strip()}
        hint = place.get("location_hint")
        if isinstance(hint, str) and hint.strip():
            cleaned_place["location_hint"] = hint.strip()
        cleaned.append(cleaned_place)
    return cleaned


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
