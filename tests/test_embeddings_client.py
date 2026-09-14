from __future__ import annotations

import json

import httpx
import pytest
from prometheus_client import REGISTRY

from notes_bot.clients.embeddings import EmbeddingServiceError, HttpEmbeddingClient


def _observed(kind: str) -> float:
    return REGISTRY.get_sample_value("notes_embed_duration_seconds_count", {"kind": kind}) or 0.0


# asyncio_mode = "auto" (pyproject.toml) picks up async def tests on its
# own — no per-test marker needed, unlike test_acl.py's `pytestmark` which
# predates that setting being relied on here.


def _client(handler) -> HttpEmbeddingClient:
    return HttpEmbeddingClient(
        "http://notes-embeddings:8000",
        model_name="test-model",
        dim=4,
        transport=httpx.MockTransport(handler),
    )


def test_model_name_and_dim_are_configuration_not_network_calls():
    client = HttpEmbeddingClient("http://x:8000", model_name="e5", dim=768)
    assert client.model_name == "e5"
    assert client.dim == 768


async def test_embed_passages_sends_kind_passage_and_returns_vectors():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        body = json.loads(request.read())
        assert body == {"texts": ["a", "b"], "kind": "passage"}
        return httpx.Response(200, json={"vectors": [[0.1, 0.2], [0.3, 0.4]]})

    vectors = await _client(handler).embed_passages(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_query_sends_kind_query_and_returns_a_single_vector():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert body == {"texts": ["q"], "kind": "query"}
        return httpx.Response(200, json={"vectors": [[0.5, 0.6]]})

    vector = await _client(handler).embed_query("q")
    assert vector == [0.5, 0.6]


async def test_429_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "at capacity"})

    with pytest.raises(EmbeddingServiceError) as exc_info:
        await _client(handler).embed_query("q")
    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 429


async def test_400_is_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "batch too large"})

    with pytest.raises(EmbeddingServiceError) as exc_info:
        await _client(handler).embed_passages(["a"])
    assert exc_info.value.retryable is False


async def test_503_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "model not loaded"})

    with pytest.raises(EmbeddingServiceError) as exc_info:
        await _client(handler).embed_query("q")
    assert exc_info.value.retryable is True


async def test_5xx_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(EmbeddingServiceError) as exc_info:
        await _client(handler).embed_query("q")
    assert exc_info.value.retryable is True


async def test_non_json_error_body_falls_back_to_raw_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="not json")

    with pytest.raises(EmbeddingServiceError, match="not json"):
        await _client(handler).embed_query("q")


async def test_records_duration_by_kind_on_success_and_failure():
    """06-deployment.md, "Латентность /embed" — recorded for both a
    successful call and one that raises, see clients/embeddings.py's
    `_embed` finally block."""
    passage_before = _observed("passage")
    query_before = _observed("query")

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vectors": [[0.1]]})

    await _client(ok).embed_passages(["a"])
    assert _observed("passage") == passage_before + 1

    def fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(EmbeddingServiceError):
        await _client(fail).embed_query("q")
    assert _observed("query") == query_before + 1
