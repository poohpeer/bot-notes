"""Client for notes-translate — see docs/architecture/05-contracts.md,
"Translate-сервис". Mirrors clients/embeddings.py's shape: callers get a
plain "translated text" back and never see the HTTP contract or the
service's own error codes.
"""

from __future__ import annotations

import httpx


class TranslateServiceError(RuntimeError):
    """Wraps a non-2xx response from notes-translate. Same
    retryable/not-retryable split as EmbeddingServiceError, but callers of
    this client only ever log-and-fall-back on it (see queue/tasks.py,
    bot/logic.py) — translation is a search-quality improvement, not a
    requirement for a note to be saved or a query to run."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"translate service returned {status_code}: {message}")
        self.status_code = status_code


class HttpTranslateClient:
    def __init__(
        self,
        base_url: str,
        *,
        target_lang: str = "eng_Latn",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._target_lang = target_lang
        self._timeout = timeout
        # None in production (real network); tests inject httpx.MockTransport.
        self._transport = transport

    async def translate_passages(self, texts: list[str]) -> list[str]:
        return await self._translate(texts)

    async def translate_query(self, text: str) -> str:
        results = await self._translate([text])
        return results[0]

    async def _translate(self, texts: list[str]) -> list[str]:
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/translate",
                    json={"texts": texts, "target_lang": self._target_lang},
                )
            except httpx.HTTPError as exc:
                raise TranslateServiceError(503, str(exc)) from exc

        if response.status_code != 200:
            raise TranslateServiceError(response.status_code, _error_detail(response))

        return [item["text"] for item in response.json()["items"]]


def _error_detail(response: httpx.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text
