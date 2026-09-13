"""Rendering — pure text formatting, no aiogram types, no I/O. See
docs/architecture/04-search.md, "Рендер выдачи".
"""

from __future__ import annotations

from dataclasses import dataclass

_SOURCE_ICONS = {
    "text": "📝",
    "voice": "🎤",
    "page": "🔗",
    "youtube": "▶️",
    "instagram": "📷",
    "map": "📍",
}

_FRAGMENT_MAX = 200
_TITLE_FALLBACK_MAX = 60


@dataclass(frozen=True)
class RenderableHit:
    note_id: int
    title: str | None
    source_type: str
    source_url: str | None
    tags: list[str]
    chunk_text: str
    is_owner: bool


def render_privacy_toggle_confirmation(visibility: str) -> str:
    label = "приватная 🔒" if visibility == "private" else "публичная 🌐"
    return f"Заметка теперь {label}."


def render_search_card(hit: RenderableHit) -> str:
    icon = _SOURCE_ICONS.get(hit.source_type, "📝")
    heading = hit.title or _truncate(hit.chunk_text, _TITLE_FALLBACK_MAX)
    lines = [f"{icon} {heading}"]

    fragment = _truncate(hit.chunk_text, _FRAGMENT_MAX)
    if fragment and fragment != heading:
        lines.append(fragment)

    if hit.tags:
        lines.append(" ".join(f"#{t}" for t in hit.tags))

    if hit.source_url:
        lines.append(hit.source_url)

    return "\n".join(lines)


def render_no_more_results() -> str:
    return "Больше ничего не найдено."


def render_search_expired() -> str:
    return "Поиск устарел, повторите запрос."


def render_note_saved(*, visibility: str) -> str:
    label = "приватная 🔒" if visibility == "private" else "публичная 🌐"
    return f"Сохраняю... Заметка {label}."


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"
