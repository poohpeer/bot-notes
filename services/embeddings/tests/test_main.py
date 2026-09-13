from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from embeddings_service.config import Settings
from embeddings_service.main import create_app


def _client(settings, fake_model) -> TestClient:
    app = create_app(settings=settings, model_loader=lambda: fake_model)
    return TestClient(app)


def test_healthz_ok(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_ok_after_startup(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        response = client.get("/readyz")
        assert response.status_code == 200


def test_model_endpoint(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        body = client.get("/model").json()
        assert body == {
            "name": "fake-e5-model",
            "dim": 4,
            "max_seq_length": 512,
            "tokenizer": "FakeTokenizer",
            "normalized": True,
            "requires_prefix": True,
        }


def test_embed_returns_vectors_for_each_text(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        response = client.post("/embed", json={"texts": ["a", "bb"], "kind": "passage"})
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "fake-e5-model"
        assert body["normalized"] is True
        assert len(body["vectors"]) == 2


def test_embed_passes_kind_through_to_the_model(test_settings, fake_model):
    """Query vs passage must reach the model layer — see fake_model.calls,
    which records exactly what embed() was called with. The first call is
    always the startup warmup run (see main.py's lifespan), so this checks
    the last one."""
    with _client(test_settings, fake_model) as client:
        client.post("/embed", json={"texts": ["hi"], "kind": "query"})
        assert fake_model.calls[-1] == (["hi"], "query")


def test_embed_rejects_empty_texts_list(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        response = client.post("/embed", json={"texts": [], "kind": "passage"})
        assert response.status_code == 422


def test_embed_rejects_empty_string_in_texts(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        response = client.post("/embed", json={"texts": ["ok", ""], "kind": "passage"})
        assert response.status_code == 400


def test_embed_rejects_unknown_kind(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        response = client.post("/embed", json={"texts": ["hi"], "kind": "sideways"})
        assert response.status_code == 422


def test_embed_rejects_batch_over_limit(test_settings, fake_model):
    # test_settings.max_batch_size == 3
    with _client(test_settings, fake_model) as client:
        response = client.post("/embed", json={"texts": ["a", "b", "c", "d"], "kind": "passage"})
        assert response.status_code == 400


def test_embed_rejects_total_length_over_limit(test_settings, fake_model):
    # test_settings.max_total_chars == 50
    with _client(test_settings, fake_model) as client:
        response = client.post("/embed", json={"texts": ["x" * 30, "y" * 30], "kind": "passage"})
        assert response.status_code == 413


def test_model_not_loaded_returns_503():
    """A loader that fails leaves app.state.model None — routes must not
    crash on a missing model, they answer 503 (see /model and /embed)."""
    from embeddings_service.config import Settings

    def failing_loader():
        raise RuntimeError("weights not found")

    app = create_app(settings=Settings(), model_loader=failing_loader)
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass  # lifespan startup re-raises — the process should crash, not degrade


class _GatedFakeModel:
    """Blocks inside embed() until the test releases it — deterministic
    contention instead of racing real sleep durations against a timeout."""

    name = "gated-fake"
    dim = 2
    max_seq_length = 8
    tokenizer_name = "Gated"
    requires_prefix = False

    def __init__(self) -> None:
        self.holding = threading.Event()
        self.release = threading.Event()

    def embed(self, texts, kind):
        if texts == ["warmup"]:
            # The lifespan startup warmup call (see main.py) must not gate,
            # or every test using this model would pay the full wait
            # timeout during app startup before the test body even runs.
            return [[0.0, 0.0]]
        self.holding.set()
        self.release.wait(timeout=5)
        return [[0.0, 0.0] for _ in texts]


async def test_concurrency_over_limit_returns_429():
    """max_concurrency=1: while request A holds the only slot, request B
    must queue and time out into 429 rather than wait forever — see
    "Внутреннее устройство". A's own request shares the same timeout budget
    and is never released in time to complete either (by design: it is
    parked in embed() for the entire test), so this only asserts B's
    outcome and cleans A up afterwards.
    """
    model = _GatedFakeModel()
    settings = Settings(EMBEDDINGS_MAX_CONCURRENCY=1, EMBEDDINGS_QUEUE_TIMEOUT_S=0.2)
    app = create_app(settings=settings, model_loader=lambda: model)
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            task_a = asyncio.create_task(
                client.post("/embed", json={"texts": ["a"], "kind": "passage"})
            )
            # Wait until A is actually inside embed() and holding the slot,
            # not just "started" — otherwise B might win the race for it.
            await asyncio.to_thread(model.holding.wait, 5)

            response_b = await client.post("/embed", json={"texts": ["b"], "kind": "passage"})
            assert response_b.status_code == 429

            # Cleanup only — task_a's own outcome (it shares the same 0.2s
            # budget as B and is still parked in embed() at this point) is
            # not part of what this test asserts.
            model.release.set()
            task_a.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task_a
