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


def render_voice_note_saved(*, visibility: str) -> str:
    label = "приватная 🔒" if visibility == "private" else "публичная 🌐"
    return f"Распознаю голосовое... Заметка {label}."


def render_group_note_saved() -> str:
    """Group notes have no visibility to report — see ADR-5/ADR-10. R11 in
    07-decisions.md: an understandable reply on every save is what keeps
    group members from thinking a mention silently did nothing."""
    return "Сохранил в заметки комнаты."


def render_smart_answer_pending() -> str:
    """04-search.md's mermaid: shown right after the ordinary results, in
    place of a synchronous typing indicator — the synthesis itself runs on
    a worker and may take minutes, so there's nothing to keep "typing" for
    from the bot process."""
    return "🧠 Готовлю умный ответ по вашим заметкам…"


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
