"""Ops alerts for a note's processing — separate from notes-bot's own
Telegram bot (TELEGRAM_BOT_TOKEN), which talks to end users, not operators.

Same shape as ai-proxy's own notify.py (ALERTS_TELEGRAM_* can point at the
same bot/chat as ai-proxy's AI_PROXY_TELEGRAM_*): sending never raises —
answering a user's note is the job here, and an unreachable Telegram is a
worse reason to fail that than whatever prompted the alert — and unset
credentials turn alerts off rather than failing startup.

Unlike ai-proxy's fixed set of provider slots, a note settles into 'done' or
'failed' for good, so the same alert for the same note cannot recur under
normal operation; the dedup window here guards only against two GC runs
racing each other, not against a stream of repeats.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Protocol

import httpx

log = logging.getLogger("notes_bot.alerts")

DEFAULT_MIN_INTERVAL = timedelta(minutes=5)
_TIMEOUT = 10.0
_ERROR_MAX = 400


class AlertNotifier(Protocol):
    async def reclaimed(
        self,
        *,
        note_id: int,
        source_type: str,
        source_url: str | None,
        attempt: int,
        max_attempts: int,
        stuck_minutes: int,
    ) -> None: ...

    async def abandoned(
        self,
        *,
        note_id: int,
        source_type: str,
        source_url: str | None,
        attempts: int,
        reason: str,
    ) -> None: ...

    async def failed(
        self,
        *,
        note_id: int,
        source_type: str,
        source_url: str | None,
        error: str,
    ) -> None: ...


def _identify(note_id: int, source_type: str, source_url: str | None) -> str:
    """The note named the way a human can act on it: id first (it is what
    every admin query and log line keys on), the link right after if there
    is one — a note with no URL (voice, plain text) just skips that part."""
    parts = [f"#{note_id} ({source_type})"]
    if source_url:
        parts.append(source_url)
    return " ".join(parts)


def _block(title: str, *fields, body: str | None = None) -> str:
    lines = [title]
    lines += [f"{name}: {value}" for name, value in fields if value]
    if body:
        lines += ["", body.strip()[:_ERROR_MAX]]
    return "\n".join(lines)


class Notifier:
    """Sends to a Telegram chat, at most once per key per interval."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        min_interval: timedelta = DEFAULT_MIN_INTERVAL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._min_interval = min_interval
        self._client = client
        self._last: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def reclaimed(
        self,
        *,
        note_id: int,
        source_type: str,
        source_url: str | None,
        attempt: int,
        max_attempts: int,
        stuck_minutes: int,
    ) -> None:
        """A worker went quiet on this note for `stuck_minutes` — died,
        was killed, or a signal-based timeout landed somewhere its own
        except-block could not catch (see queue/tasks.py's docstring on
        why RQ's job_timeout alone cannot be trusted to record a reason).
        There is no error to show: nothing here failed in a way that says
        why, only stopped answering."""
        await self._send(
            f"reclaimed:{note_id}:{attempt}",
            _block(
                f"🔄 Заметка {_identify(note_id, source_type, source_url)} зависла",
                ("Не отвечал", f"{stuck_minutes} мин"),
                ("Что сделали", f"поставили в очередь заново (попытка {attempt}/{max_attempts})"),
                (
                    "Чего ждать",
                    "обработается сама; продолжит виснуть — сообщим ещё раз"
                    if attempt < max_attempts
                    else "это последняя попытка — после неё, если не выйдет, "
                    "заметка будет помечена неудавшейся",
                ),
            ),
        )

    async def abandoned(
        self,
        *,
        note_id: int,
        source_type: str,
        source_url: str | None,
        attempts: int,
        reason: str,
    ) -> None:
        """GC_MAX_ATTEMPTS reclaims in a row and still no 'done' — retrying
        again would only repeat whatever this is, so this is the last word
        on the note rather than another silent loop."""
        await self._send(
            f"abandoned:{note_id}",
            _block(
                f"⛔ Заметка {_identify(note_id, source_type, source_url)} не обработалась",
                ("Попыток", str(attempts)),
                ("Что сделали", "пометили окончательно неудавшейся, больше не повторяем"),
                ("Чего ждать", "сама не обработается — нужно переслать заново"),
                body=reason,
            ),
        )

    async def failed(
        self,
        *,
        note_id: int,
        source_type: str,
        source_url: str | None,
        error: str,
    ) -> None:
        """process_note_async's own except-block: a real exception was
        caught and recorded, unlike `reclaimed`/`abandoned`, where the
        worker just went silent."""
        await self._send(
            f"failed:{note_id}",
            _block(
                f"❌ Заметка {_identify(note_id, source_type, source_url)} не обработалась",
                ("Что сделали", "пометили неудавшейся, автоматических попыток не будет"),
                ("Чего ждать", "нужно переслать заново"),
                body=error,
            ),
        )

    async def _send(self, key: str, text: str) -> None:
        now = datetime.now(UTC)
        async with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < self._min_interval:
                log.debug("alert %s suppressed, last sent %s", key, last)
                return
            # Recorded before the send, not after: a Telegram outage must
            # not turn every subsequent failure into another attempt.
            self._last[key] = now
        try:
            client = self._client or httpx.AsyncClient(timeout=_TIMEOUT)
            try:
                response = await client.post(
                    f"https://api.telegram.org/bot{self._token}/sendMessage",
                    data={"chat_id": self._chat_id, "text": text},
                )
                if response.status_code != 200:
                    log.warning(
                        "telegram refused %s: %s %s", key, response.status_code, response.text[:300]
                    )
                else:
                    log.info("alerted %s", key)
            finally:
                if self._client is None:
                    await client.aclose()
        except Exception:
            log.warning("could not send alert %s", key, exc_info=True)


class NullNotifier:
    """No token/chat id configured: every call is a no-op, quietly — every
    call site can alert unconditionally rather than guard each one."""

    async def reclaimed(self, **_: object) -> None:
        return None

    async def abandoned(self, **_: object) -> None:
        return None

    async def failed(self, **_: object) -> None:
        return None


def build_notifier(
    token: str | None, chat_id: str | None, *, min_interval_minutes: float
) -> AlertNotifier:
    if not token or not chat_id:
        log.info("alerts are off: no bot token or chat id configured")
        return NullNotifier()
    return Notifier(token, chat_id, min_interval=timedelta(minutes=min_interval_minutes))
