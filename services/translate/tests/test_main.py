from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from translate_service.config import Settings
from translate_service.main import create_app
from translate_service.model import TranslationResult


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
        assert body == {"name": "fake-nllb-model", "default_target_lang": "eng_Latn"}


def test_translate_returns_an_item_per_text(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        response = client.post(
            "/translate", json={"texts": ["привет", "мир"], "target_lang": "eng_Latn"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "fake-nllb-model"
        assert [item["text"] for item in body["items"]] == [
            "[eng_Latn] привет",
            "[eng_Latn] мир",
        ]
        assert all(item["source_lang"] == "ru" and item["translated"] for item in body["items"])


def test_translate_defaults_target_lang_when_omitted(test_settings, fake_model):
    """The startup warmup call already used the default once — this checks
    the last real call, not the first."""
    with _client(test_settings, fake_model) as client:
        client.post("/translate", json={"texts": ["hi"]})
        assert fake_model.calls[-1] == (["hi"], "eng_Latn")


def test_translate_rejects_empty_texts_list(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        response = client.post("/translate", json={"texts": []})
        assert response.status_code == 422


def test_translate_rejects_empty_string_in_texts(test_settings, fake_model):
    with _client(test_settings, fake_model) as client:
        response = client.post("/translate", json={"texts": ["ok", ""]})
        assert response.status_code == 400


def test_translate_rejects_batch_over_limit(test_settings, fake_model):
    # test_settings.max_batch_size == 3
    with _client(test_settings, fake_model) as client:
        response = client.post("/translate", json={"texts": ["a", "b", "c", "d"]})
        assert response.status_code == 400


def test_translate_rejects_total_length_over_limit(test_settings, fake_model):
    # test_settings.max_total_chars == 50
    with _client(test_settings, fake_model) as client:
        response = client.post("/translate", json={"texts": ["x" * 30, "y" * 30]})
        assert response.status_code == 413


def test_model_not_loaded_returns_503():
    """A loader that fails leaves app.state.model None — routes must not
    crash on a missing model, they answer 503 (see /model and /translate)."""

    def failing_loader():
        raise RuntimeError("weights not found")

    app = create_app(settings=Settings(), model_loader=failing_loader)
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass  # lifespan startup re-raises — the process should crash, not degrade


class _GatedFakeModel:
    """Blocks inside translate() until the test releases it — deterministic
    contention instead of racing real sleep durations against a timeout."""

    name = "gated-fake"

    def __init__(self) -> None:
        self.holding = threading.Event()
        self.release = threading.Event()

    def translate(self, texts, target_lang):
        if texts == ["warmup"]:
            # The lifespan startup warmup call (see main.py) must not gate,
            # or every test using this model would pay the full wait
            # timeout during app startup before the test body even runs.
            return [TranslationResult(text="warmup", source_lang=None, translated=False)]
        self.holding.set()
        self.release.wait(timeout=5)
        return [TranslationResult(text=t, source_lang="ru", translated=True) for t in texts]


async def test_concurrency_over_limit_returns_429():
    """max_concurrency=1: while request A holds the only slot, request B
    must queue and time out into 429 rather than wait forever — see
    embeddings_service's own equivalent test."""
    model = _GatedFakeModel()
    settings = Settings(TRANSLATE_MAX_CONCURRENCY=1, TRANSLATE_QUEUE_TIMEOUT_S=0.2)
    app = create_app(settings=settings, model_loader=lambda: model)
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            task_a = asyncio.create_task(client.post("/translate", json={"texts": ["a"]}))
            await asyncio.to_thread(model.holding.wait, 5)

            response_b = await client.post("/translate", json={"texts": ["b"]})
            assert response_b.status_code == 429

            model.release.set()
            task_a.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task_a
