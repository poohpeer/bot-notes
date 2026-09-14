"""Client for notes-embeddings — see docs/architecture/05-contracts.md.

Implements the `EmbeddingClient` interface from that doc: callers ask for a
vector "in the role of a query or a passage" and never see the HTTP
contract, batching limits, or the service's error codes directly.
"""

from __future__ import annotations

import time

import httpx

from notes_bot.metrics import EMBED_DURATION_SECONDS


class EmbeddingServiceError(RuntimeError):
    """Wraps a non-2xx response from notes-embeddings.

    `retryable` distinguishes a transient condition (429 saturated, 503 model
    not loaded, 5xx) from a request that is simply wrong for this text (400
    empty/oversized batch, 413 too long) — see 05-contracts.md's error table.
    A caller (process_note) uses this to decide whether re-queuing the job
    could possibly help.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"embeddings service returned {status_code}: {message}")
        self.status_code = status_code
        self.retryable = status_code in (429, 503) or status_code >= 500


class HttpEmbeddingClient:
    """model_name/dim are configuration, not queried per call: both come
    from Settings (EMBEDDING_MODEL_NAME/EMBEDDING_DIM), already verified
    against the service's own GET /model at process startup — see
    notes_bot.startup.validate_startup. Re-querying here on every property
    access would just be the same network round trip with extra steps.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model_name: str,
        dim: int,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dim = dim
        self._timeout = timeout
        # None in production (real network); tests inject httpx.MockTransport
        # instead of hitting a live service.
        self._transport = transport

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "passage")

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], "query")
        return vectors[0]

    async def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                try:
                    response = await client.post(
                        f"{self._base_url}/embed", json={"texts": texts, "kind": kind}
                    )
                except httpx.HTTPError as exc:
                    # Connection refused, DNS failure, timeout — treat like a
                    # transient service error rather than a distinct exception
                    # type every caller has to learn separately.
                    raise EmbeddingServiceError(503, str(exc)) from exc

            if response.status_code != 200:
                detail = _error_detail(response)
                raise EmbeddingServiceError(response.status_code, detail)

            return response.json()["vectors"]
        finally:
            # 06-deployment.md, "Латентность /embed" — узкое место общее
            # для приёма и поиска.
            EMBED_DURATION_SECONDS.labels(kind=kind).observe(time.perf_counter() - started)


def _error_detail(response: httpx.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text
