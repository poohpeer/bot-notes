"""LLM port and its three implementations — see docs/architecture/05-contracts.md,
"Порт LLMClient" and "ai-proxy".

**Security**: `ProxyAILLMClient` must stay unused (LLM_ENABLED=false) until
ADR-14's ai-proxy sandbox fix ships — see 05-contracts.md, "Безопасность:
блокер для LLM_ENABLED=true". This module doesn't enforce that itself; the
caller (queue/tasks.py's enrich_note) is what checks the flag before ever
constructing a real client.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    text: str
    parsed: dict | None
    model: str | None
    usage: dict | None


def extract_token_usage(usage: dict | None) -> tuple[int | None, int | None]:
    """(tokens_in, tokens_out), best-effort — ai-proxy's `metadata` shape
    (05-contracts.md, "Порт LLMClient") differs by provider. Verified live
    against `claude_code`: `metadata.raw_envelope.usage.input_tokens`/
    `.output_tokens` (plus separate cache_read/cache_creation counts this
    doesn't surface — "input_tokens" alone can undercount when most of the
    prompt was cache-served). `codex`'s CLI-based usage reporting wasn't
    reachable to verify (every codex account was quota-exhausted when this
    was written) and may not carry token counts at all — (None, None)
    rather than a guessed 0, which would claim a real zero-token call."""
    if not isinstance(usage, dict):
        return None, None
    envelope = usage.get("raw_envelope")
    if not isinstance(envelope, dict):
        return None, None
    inner = envelope.get("usage")
    if not isinstance(inner, dict):
        return None, None
    tokens_in = inner.get("input_tokens")
    tokens_out = inner.get("output_tokens")
    return (
        tokens_in if isinstance(tokens_in, int) else None,
        tokens_out if isinstance(tokens_out, int) else None,
    )


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        history: list[dict] | None = None,
        timeout_s: float = 60.0,
    ) -> LLMResult: ...


class LLMServiceError(Exception):
    """`error_type` is ai-proxy's own type string (05-contracts.md's error
    table) — callers branch on it because 503 quota_exhausted needs a much
    longer backoff than a plain timeout, not just "retry or don't".
    """

    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        super().__init__(f"ai-proxy returned {status_code} {error_type}: {message}")
        self.status_code = status_code
        self.error_type = error_type
        self.is_quota_exhausted = error_type == "quota_exhausted"
        # 4xx other than 429 is a request bug (missing_schema,
        # unavailable_provider, ...) — retrying it fails identically.
        self.retryable = status_code >= 500 or status_code == 429


class NullLLMClient:
    """LLM_ENABLED=false — see 05-contracts.md: "довести до рабочего
    состояния всё, кроме генеративных фич, не дожидаясь готовности
    интеграции". Always the same empty, uneventful result; every enrichment
    feature already has to treat "the LLM produced nothing usable" as a
    normal outcome (a real call can fail the same way), so this exercises
    that exact path rather than needing its own special case.
    """

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        history: list[dict] | None = None,
        timeout_s: float = 60.0,
    ) -> LLMResult:
        return LLMResult(text="", parsed=None, model=None, usage=None)


class FakeLLMClient:
    """Tests — see 05-contracts.md's "Три реализации" table: deterministic
    replies keyed by a substring of the prompt, so a test can script several
    different calls (title vs tags vs summary) against one client."""

    def __init__(self) -> None:
        self._replies: dict[str, LLMResult] = {}
        self.calls: list[dict] = []

    def add_reply(self, prompt_contains: str, result: LLMResult) -> None:
        self._replies[prompt_contains] = result

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        history: list[dict] | None = None,
        timeout_s: float = 60.0,
    ) -> LLMResult:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        for key, result in self._replies.items():
            if key in user or key in system:
                return result
        return LLMResult(text="", parsed=None, model="fake", usage=None)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Strips a ```json fence if present, then parses. Returns None (never
    raises) on anything that isn't a JSON object — the caller decides
    whether to retry."""
    candidate = text.strip()
    match = _JSON_FENCE_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class ProxyAILLMClient:
    """The real thing — POST /v1/complete, provider="codex" first, per
    05-contracts.md. codex never returns structured_output (see that doc's
    "Структурированный вывод по схеме недоступен"), so `json_schema`
    support is entirely this class's own doing: an instruction appended to
    `system`, then parsing `result` as JSON with one retry on failure —
    unaffected by which provider actually answered, since it works over
    plain text either way.

    Falls back to provider="claude_code" when codex reports every account
    quota_exhausted (05-contracts.md's error table) — that CLI login is a
    Claude subscription's rolling usage window, not per-token API billing
    (ai-proxy's claude_code.py runs `claude -p`, authenticated via
    `claudeAiOauth`/`subscriptionType`, not an API key), so retrying there
    costs no more than the enrichment already not happening. Only for
    quota_exhausted specifically — every other error (a real bug, a
    provider genuinely unreachable) is not made better by trying again
    against a different backend.
    """

    _PRIMARY_PROVIDER = "codex"
    _FALLBACK_PROVIDER = "claude_code"

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._transport = transport  # tests only

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        history: list[dict] | None = None,
        timeout_s: float = 60.0,
    ) -> LLMResult:
        effective_system = system
        if json_schema is not None:
            schema_text = json.dumps(json_schema, ensure_ascii=False)
            effective_system = (
                f"{system}\n\nВерни ТОЛЬКО JSON, соответствующий этой схеме, "
                f"без пояснений и без markdown-ограждений:\n{schema_text}"
            )

        response = await self._request(
            system=effective_system, user=user, history=history, timeout_s=timeout_s
        )
        text = response["result"] or ""
        model = response.get("model")
        usage = response.get("metadata")

        if json_schema is None:
            return LLMResult(text=text, parsed=None, model=model, usage=usage)

        parsed = _extract_json(text)
        if parsed is not None:
            return LLMResult(text=text, parsed=parsed, model=model, usage=usage)

        # One retry with the parse failure spelled out — see 05-contracts.md,
        # "Разбор JSON адаптером".
        retry_system = (
            f"{effective_system}\n\nПредыдущий ответ не был валидным JSON. "
            f"Верни только валидный JSON, без пояснений."
        )
        retry_response = await self._request(
            system=retry_system, user=user, history=history, timeout_s=timeout_s
        )
        retry_text = retry_response["result"] or ""
        parsed = _extract_json(retry_text)
        return LLMResult(
            text=retry_text, parsed=parsed, model=retry_response.get("model"), usage=usage
        )

    async def _request(
        self, *, system: str, user: str, history: list[dict] | None, timeout_s: float
    ) -> dict:
        try:
            return await self._request_provider(
                self._PRIMARY_PROVIDER,
                system=system,
                user=user,
                history=history,
                timeout_s=timeout_s,
            )
        except LLMServiceError as exc:
            if not exc.is_quota_exhausted:
                raise
            log.warning(
                "ai-proxy: %s quota_exhausted, falling back to %s",
                self._PRIMARY_PROVIDER,
                self._FALLBACK_PROVIDER,
            )
            return await self._request_provider(
                self._FALLBACK_PROVIDER,
                system=system,
                user=user,
                history=history,
                timeout_s=timeout_s,
            )

    async def _request_provider(
        self, provider: str, *, system: str, user: str, history: list[dict] | None, timeout_s: float
    ) -> dict:
        body = {
            "provider": provider,
            "prompt": user,
            "system": system,
            "output_format": "text",
            "json_schema": None,
            "timeout_s": timeout_s,
            "history": history or [],
        }
        async with httpx.AsyncClient(timeout=timeout_s + 5, transport=self._transport) as client:
            try:
                response = await client.post(f"{self._base_url}/v1/complete", json=body)
            except httpx.HTTPError as exc:
                raise LLMServiceError(502, "provider_unreachable", str(exc)) from exc

        if response.status_code != 200:
            error = _error_body(response)
            raise LLMServiceError(
                response.status_code, error.get("type", "unknown"), error.get("message", "")
            )
        return response.json()


def _error_body(response: httpx.Response) -> dict:
    try:
        return response.json().get("error", {})
    except ValueError:
        return {"message": response.text}
