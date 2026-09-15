"""Rendering — pure text formatting, no aiogram types, no I/O. See
docs/architecture/04-search.md, "Рендер выдачи".
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

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

# A link-type note (youtube/instagram/page/map) has raw_text == source_url
# verbatim right after saving — extracted_text/title only exist once
# process_note/enrich_note finish. Without this, an untitled, unextracted
# note showed the same link three times: truncated as the heading, in full
# as the "fragment", and again as the source_url line.
_PENDING_HEADING = {
    "youtube": "YouTube-видео",
    "instagram": "Instagram-видео",
    "page": "Страница",
    "map": "Место",
}


@dataclass(frozen=True)
class RenderableHit:
    note_id: int
    title: str | None
    source_type: str
    source_url: str | None
    tags: list[str]
    chunk_text: str
    is_owner: bool
    structured: dict


def render_privacy_toggle_confirmation(visibility: str) -> str:
    label = "приватная 🔒" if visibility == "private" else "публичная 🌐"
    return f"Заметка теперь {label}."


def render_debug_toggle_confirmation(*, enabled: bool) -> str:
    return "Debug-режим включён." if enabled else "Debug-режим выключен."


def render_debug_processing_done(*, elapsed_s: float) -> str:
    """Sent by process_note itself once a note is fully processed, when the
    owner has /debug on — see 03-ingest.md, "Debug: время обработки"."""
    return f"⏱ Обработка завершена. Заняло: {elapsed_s:.1f}с"


def render_processing_failed() -> str:
    """process_note's own except-block sends this to the note's own chat —
    previously only an ops-side alert (clients/alerts.py) went out on a
    failure, and the person who actually sent the note got silence, no
    different from a note that was still quietly processing. No error text
    here: that's for the ops alert, not the end user (03-ingest.md,
    "Обработка не удалась")."""
    return "❌ Не получилось обработать заметку. Попробуйте отправить ссылку ещё раз."


def _google_maps_search_url(query: str) -> str:
    # No geocoding step anywhere in this pipeline — a search URL (Google's
    # own documented fallback for "I have a name/address, not coordinates")
    # instead of a pin, since generate_places (enrich.py) only ever extracts
    # text out of a caption/ASR transcript, never a verified lat/lng.
    return f"https://www.google.com/maps/search/?api=1&query={quote(query)}"


def render_places(structured: dict) -> list[str]:
    """`structured["places"]` — see enrich.py's `generate_places`: a
    youtube/instagram note's places, extracted from its caption/transcript.
    One line per place, each a Google Maps search link built from whatever
    text the LLM found — no guarantee it resolves to exactly the right
    result, since the source text itself (often an ASR transcript) can
    mangle a name or address (03-ingest.md notes this under "Места из
    видео")."""
    places = structured.get("places") if isinstance(structured, dict) else None
    if not isinstance(places, list):
        return []
    lines = []
    for place in places:
        if not isinstance(place, dict):
            continue
        name = place.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        hint = place.get("location_hint")
        query = f"{name} {hint}" if isinstance(hint, str) and hint.strip() else name
        lines.append(f"📍 {name} — {_google_maps_search_url(query)}")
    return lines


def render_search_card(hit: RenderableHit) -> str:
    icon = _SOURCE_ICONS.get(hit.source_type, "📝")
    chunk = hit.chunk_text.strip()
    is_bare_url = bool(hit.source_url) and chunk == hit.source_url.strip()

    if hit.title:
        heading = hit.title
    elif is_bare_url:
        heading = _PENDING_HEADING.get(hit.source_type, "Заметка")
    else:
        heading = _truncate(chunk, _TITLE_FALLBACK_MAX)
    lines = [f"{icon} {heading}"]

    if not is_bare_url:
        fragment = _truncate(chunk, _FRAGMENT_MAX)
        if fragment and fragment != heading:
            lines.append(fragment)

    if hit.tags:
        lines.append(" ".join(f"#{t}" for t in hit.tags))

    if hit.source_url:
        lines.append(hit.source_url)

    lines.extend(render_places(hit.structured))

    return "\n".join(lines)


def render_no_more_results() -> str:
    return "Больше ничего не найдено."


def render_search_expired() -> str:
    return "Поиск устарел, повторите запрос."


def render_note_saved(*, visibility: str) -> str:
    label = "приватная 🔒" if visibility == "private" else "публичная 🌐"
    return f"Сохраняю... Заметка {label}."


def render_voice_note_saved(*, visibility: str) -> str:
    label = "приватная 🔒" if visibility == "private" else "публичная 🌐"
    return f"Распознаю голосовое... Заметка {label}."


def render_group_note_saved() -> str:
    """Group notes have no visibility to report — see ADR-5/ADR-10. R11 in
    07-decisions.md: an understandable reply on every save is what keeps
    group members from thinking a mention silently did nothing."""
    return "Сохранил в заметки комнаты."


def render_smart_answer_pending() -> str:
    """04-search.md's mermaid: shown in place of a synchronous typing
    indicator — the synthesis itself runs on a worker and may take minutes,
    so there's nothing to keep "typing" for from the bot process."""
    return "🧠 Готовлю умный ответ по вашим заметкам…"


def render_smart_answer_failed() -> str:
    """/smart_search no longer shows the raw cards alongside the pending
    marker (04-search.md) — a silent failure now leaves the pending marker
    as a dead end instead of a shrug next to results the user already has,
    so smart_answer_async sends this instead of just logging and returning
    on its failure paths."""
    return "Не получилось подготовить умный ответ. Попробуйте ещё раз."


def render_group_welcome() -> str:
    return (
        "Привет! Я сохраняю заметки с поиском по смыслу. В этой группе я "
        "запоминаю только то, что адресовано мне — упомяните меня (@) или "
        "ответьте на моё сообщение, и я сохраню текст. Обычный разговор "
        "участников я не трогаю."
    )


def render_capture_mode_changed(mode: str) -> str:
    if mode == "all":
        return (
            "Режим сохранения: всё. Буду сохранять каждое сообщение в этой "
            "группе — но для этого нужно выключить privacy mode у бота в "
            "BotFather (/setprivacy → Disable), иначе Telegram не покажет "
            "мне обычные сообщения без упоминания."
        )
    return "Режим сохранения: только упоминания и ответы мне."


def render_list_empty() -> str:
    return "Заметок пока нет."


def render_trash_empty() -> str:
    return "Корзина пуста."


def render_delete_confirmation_prompt() -> str:
    return "Удалить эту заметку?"


def render_note_deleted() -> str:
    return "Удалено. Можно восстановить из /trash."


def render_delete_refused() -> str:
    return "Не получилось удалить — заметка уже не ваша или уже удалена."


def render_note_restored() -> str:
    return "Восстановлено."


def render_edit_saved() -> str:
    return "Текст обновлён, переиндексирую."


def render_edit_refused() -> str:
    return "Не получилось изменить — заметка не ваша или её нельзя редактировать."


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"
