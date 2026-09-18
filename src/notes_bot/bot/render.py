"""Rendering — pure text formatting, no aiogram types, no I/O. See
docs/architecture/04-search.md, "Рендер выдачи".
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as _esc
from urllib.parse import quote

_SOURCE_ICONS = {
    "text": "📝",
    "voice": "🎤",
    "page": "🔗",
    "youtube": "▶️",
    "instagram": "📷",
    "map": "📍",
    "table_event": "📅",
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
    summary: str | None = None


def render_privacy_toggle_confirmation(visibility: str) -> str:
    label = "приватная 🔒" if visibility == "private" else "публичная 🌐"
    return f"Заметка теперь {label}."


def render_debug_toggle_confirmation(*, enabled: bool) -> str:
    return "Debug-режим включён." if enabled else "Debug-режим выключен."


_DEBUG_PREVIEW_MAX = 150


def render_stage_timings(stage_order: list[str], stages: dict[str, float]) -> list[str]:
    """One line per stage in `stage_order` that's actually present in
    `stages` (a stage skipped this run - e.g. translate when
    TRANSLATE_ENABLED=false - has no entry and is silently omitted, not
    shown as 0.0с, which would misleadingly claim it ran instantly)."""
    return [f"  {name}: {stages[name]:.2f}с" for name in stage_order if name in stages]


def render_token_usage(tokens_in: int | None, tokens_out: int | None) -> str | None:
    """`None` when ai-proxy's response didn't carry token counts at all -
    happens for `codex` (CLI-provider usage reporting differs from
    claude_code's SDK-style envelope; ProxyAILLMClient.complete's own
    extraction is best-effort, see clients/llm.py) - a debug line saying
    "вход 0, выход 0" would misleadingly claim a real zero-token call."""
    if tokens_in is None and tokens_out is None:
        return None
    tin = tokens_in if tokens_in is not None else "?"
    tout = tokens_out if tokens_out is not None else "?"
    return f"🔤 Токены: вход {tin}, выход {tout}"


def render_debug_processing_done(
    *,
    elapsed_s: float,
    summary: str | None,
    indexed_text: str,
    stage_timings: dict[str, float] | None = None,
) -> str:
    """Sent by process_note itself once a note is fully processed, when the
    owner has /debug on — see 03-ingest.md, "Debug: время обработки". The
    point is to tell at a glance what the note was actually about ("видео о
    том, как женщина ищет мужа"), not just that something got indexed — so
    a real LLM summary (process_note's own call to enrich.generate_summary,
    over the same `text` that got chunked and embedded) is what's shown
    when one's available. Falls back to a plain truncated excerpt of
    `indexed_text` when it isn't (LLM_ENABLED=false, or the summary call
    itself failed) — still enough to confirm real content got indexed
    rather than, say, a silently degraded extraction leaving only the URL.

    `stage_timings` breaks the total down by stage (extract/chunk/
    translate/embed/save) - added because "12.3с" alone doesn't say whether
    that was ASR transcription or a slow translate/embed call."""
    lines = [f"⏱ Обработка завершена. Заняло: {elapsed_s:.1f}с"]
    if stage_timings:
        lines.extend(
            render_stage_timings(["extract", "chunk", "translate", "embed", "save"], stage_timings)
        )
    preview = summary.strip() if summary else _truncate(indexed_text, _DEBUG_PREVIEW_MAX)
    if preview:
        lines.append(preview)
    return "\n".join(lines)


def render_smart_answer_debug(
    *,
    elapsed_s: float,
    stage_timings: dict[str, float],
    tokens_in: int | None,
    tokens_out: int | None,
) -> str:
    """Sent as a follow-up message after /smart_search's own answer, when
    the owner has /debug on (03-ingest.md's /debug flag, reused here rather
    than a second toggle) - see 04-search.md, "Debug: тайминги
    /smart_search". Answers "почему это заняло N секунд и что съело квоту"
    without grepping worker logs."""
    lines = [f"⏱ /smart_search: {elapsed_s:.1f}с"]
    lines.extend(render_stage_timings(["translate", "embed", "search", "llm"], stage_timings))
    usage_line = render_token_usage(tokens_in, tokens_out)
    if usage_line:
        lines.append(usage_line)
    return "\n".join(lines)


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
    One line per place, the place's own name rendered as an HTML link
    (`<a href="...">`) to a Google Maps search built from whatever text the
    LLM found — no guarantee it resolves to exactly the right result, since
    the source text itself (often an ASR transcript) can mangle a name or
    address (03-ingest.md notes this under "Места из видео").

    Callers MUST send the resulting text with parse_mode="HTML"
    (aiogram's Message.answer/TelegramSender.send) — plain text would show
    the raw `<a href=...>` markup instead of a link. `name` and the built
    URL are both HTML-escaped here since either can hold arbitrary text
    pulled from an ASR transcript or a caption."""
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
        url = _google_maps_search_url(query)
        lines.append(f'📍 <a href="{_esc(url)}">{_esc(name)}</a>')
    return lines


def _heading_for(hit: RenderableHit) -> tuple[str, bool]:
    """(heading, is_bare_url) — shared between the full card and the short
    summary card, so the two never disagree on what a note is called."""
    chunk = hit.chunk_text.strip()
    is_bare_url = bool(hit.source_url) and chunk == hit.source_url.strip()

    if hit.title:
        return hit.title, is_bare_url
    if is_bare_url:
        return _PENDING_HEADING.get(hit.source_type, "Заметка"), is_bare_url
    return _truncate(chunk, _TITLE_FALLBACK_MAX), is_bare_url


def render_search_card(hit: RenderableHit) -> str:
    """Sent with parse_mode="HTML" (see render_places) — every dynamic
    piece here is escaped for that reason, not just the places line."""
    icon = _SOURCE_ICONS.get(hit.source_type, "📝")
    heading, is_bare_url = _heading_for(hit)
    lines = [f"{icon} {_esc(heading)}"]

    if not is_bare_url:
        fragment = _truncate(hit.chunk_text.strip(), _FRAGMENT_MAX)
        if fragment and fragment != heading:
            lines.append(_esc(fragment))

    if hit.tags:
        lines.append(" ".join(f"#{_esc(t)}" for t in hit.tags))

    if hit.source_url:
        lines.append(_esc(hit.source_url))

    lines.extend(render_places(hit.structured))

    return "\n".join(lines)


_SUMMARY_CARD_FRAGMENT_MAX = 120


def render_search_summary_card(hit: RenderableHit) -> str:
    """/search's default view (04-search.md, "Краткая выдача") — one or two
    lines per hit instead of the full card, so a page of several results
    doesn't turn into a wall of text to scroll past. `summary` only exists
    for notes enrich_note actually summarized (long-enough text, LLM
    enabled) — falls back to the same truncated fragment the full card
    uses when there isn't one. "Подробнее" (keyboards.search_detail_keyboard)
    expands to render_search_card for this exact hit."""
    icon = _SOURCE_ICONS.get(hit.source_type, "📝")
    heading, is_bare_url = _heading_for(hit)
    lines = [f"{icon} {heading}"]

    if not is_bare_url:
        body = hit.summary or _truncate(hit.chunk_text.strip(), _SUMMARY_CARD_FRAGMENT_MAX)
        if body and body != heading:
            lines.append(body)

    return "\n".join(lines)


def render_no_more_results() -> str:
    return "Больше ничего не найдено."


def render_search_debug_empty(
    near_misses: list[tuple[str, float]], *, max_distance: float | None
) -> str:
    """/debug (03-ingest.md, "Debug: почему поиск ничего не нашёл") — sent
    right after "ничего не найдено", when the caller has debug on, so
    tuning SEARCH_MAX_DISTANCE doesn't require pulling distances out of the
    database by hand. `near_misses`: (label, distance) for the closest hits
    that existed but didn't clear the threshold — every note in the corpus
    could still be too far, which is a different, more useful thing to know
    than "the query returned nothing"."""
    if not near_misses:
        return "🔍 Debug: в базе вообще нет заметок, видимых для этого поиска."
    threshold_note = f" (порог {max_distance})" if max_distance is not None else ""
    lines = [f"🔍 Debug: ничего не прошло порог релевантности{threshold_note}. Ближе всего:"]
    lines += [f"  {label} — distance={distance:.3f}" for label, distance in near_misses]
    return "\n".join(lines)


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
    """07-decisions.md's ADR-10 update (2026-09-18): plain @-mentions do
    NOT reach the bot while Telegram's group privacy mode is on (only
    commands and replies do) - verified live, contradicting the ADR's
    original assumption. Without this line an admin who never touches
    BotFather sees mentions silently do nothing, with no way to tell
    whether the bot is broken or just never got the message."""
    return (
        "Привет! Я сохраняю заметки с поиском по смыслу. В этой группе я "
        "запоминаю только то, что адресовано мне — упомяните меня (@) или "
        "ответьте на моё сообщение, и я сохраню текст. Обычный разговор "
        "участников я не трогаю.\n\n"
        "Важно: чтобы упоминания вообще доходили до меня, отключите "
        "Group Privacy в настройках бота у @BotFather (Bot Settings → "
        "Group Privacy → Disable) — иначе Telegram будет пропускать "
        "команды и ответы мне, но не обычные сообщения с упоминанием."
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


def render_private_only() -> str:
    """/trash (and delete/restore) stay private-chat-only — unlike /list
    (which got a group-scoped variant, see logic.list_group_notes), a
    room's shared trash of soft-deleted notes isn't something every member
    should see. Previously this silently did nothing in a group; an
    explicit reply is cheaper than a confused "почему не работает" report."""
    return "Доступно только в личных сообщениях с ботом."


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


def render_events_table_usage() -> str:
    """A bare `/events_table`, no photos attached (or none downloadable) —
    see 03-ingest.md, "/events_table". Mirrors /search's own
    "Использование: ..." reply for an argument-less command, not silence.

    Naming a tab is optional — one screenshot is already a single specific
    view, unlike a multi-tab file (not supported yet) where naming one
    would actually disambiguate something."""
    return (
        "Использование: пришлите один или несколько скриншотов таблицы "
        "(альбомом, если их несколько) с подписью /events_table\n\n"
        "Можно добавить название - пригодится, если пришлёте несколько "
        "похожих таблиц подряд:\n"
        "/events_table שכבה י'"
    )


def _table_label(tab_name: str) -> str:
    """Tab names are optional (handlers.py's own docstring on why) — every
    render_events_table_* message below either says "таблицу «X»" or just
    "таблицу", never "таблицу «»"."""
    return f' "{tab_name}"' if tab_name else ""


def render_events_table_started(tab_name: str) -> str:
    """Sent immediately in the bot process, before the album is even
    downloaded (handlers.py's F.photo handler) — the actual LLM call runs
    async on the `llm` queue and can take up to events_table_timeout_s
    (240с by default), so this stands in for a "typing" indicator that
    can't span that long."""
    return f"🔎 Разбираю таблицу{_table_label(tab_name)}…"


def render_events_table_disabled() -> str:
    return "Разбор таблиц выключен (LLM_ENABLED=false)."


def render_events_table_empty(tab_name: str) -> str:
    label = _table_label(tab_name)
    return f"Не нашёл ни одного события в таблице{label}. Попробуйте более чёткий скриншот."


def render_events_table_preview(
    *, tab_name: str, topic_tags: list[str], type_counts: dict[str, int]
) -> str:
    """Sent by the worker (queue/tasks.py's parse_events_table) once
    extraction finishes, with `keyboards.events_table_confirm_keyboard`
    attached — see 03-ingest.md, "/events_table". `type_counts` keys are
    the Russian tag names (events_table.ExtractedEvent.tag), so this
    doesn't need to know the English exam/holiday/event vocabulary."""
    lines = [f"Нашёл {sum(type_counts.values())} событий в таблице{_table_label(tab_name)}:"]
    lines.extend(f"  {count} — {tag}" for tag, count in type_counts.items())
    if topic_tags:
        lines.append("Теги: " + ", ".join(f"#{t}" for t in topic_tags))
    return "\n".join(lines)


def render_events_table_saved(*, created: int, skipped_exact: int, conflicts: int) -> str:
    """03-ingest.md, "/events_table" - "уже существует": `skipped_exact`
    is a silent no-op elsewhere (a true duplicate — same date range, type,
    and text), so this is the only place the user ever learns it happened
    at all. `conflicts` are reported here but resolved in their own,
    separate messages (render_events_table_conflict) - this line just says
    how many are still pending."""
    lines = [f"Сохранил {created} заметок."]
    if skipped_exact:
        lines.append(f"Пропустил {skipped_exact} — уже есть, без изменений.")
    if conflicts:
        lines.append(f"{conflicts} событий отличаются от уже сохранённых — решите ниже.")
    return "\n".join(lines)


def render_events_table_conflict(*, old_text: str, new_text: str) -> str:
    """One message per dedup conflict (03-ingest.md) - same date range and
    event type as an existing table_event note, but different text (the
    source table changed between two screenshots of the same tab, or the
    new parse just read it differently). `keyboards.events_table_conflict_
    keyboard` is attached alongside this."""
    return (
        "Уже есть заметка на эту дату с другим текстом:\n\n"
        f"Было: {old_text}\n"
        f"Стало: {new_text}\n\n"
        "Что оставить?"
    )


def render_events_table_conflict_resolved(*, kept_new: bool) -> str:
    return "Заменил новым." if kept_new else "Оставил старое."


def render_events_table_cancelled() -> str:
    return "Отменено, ничего не сохранено."


def render_events_table_expired() -> str:
    return "Этот разбор таблицы устарел, пришлите скриншоты заново."


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"
