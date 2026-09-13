from __future__ import annotations

import pytest

from embeddings_service.config import Settings
from embeddings_service.model import Kind


class FakeEmbeddingModel:
    """Deterministic stand-in for SentenceTransformerModel.

    Real weights are ~1GB and slow to load; nothing under test needs actual
    semantics, only that the service wires kind/prefix/dim/errors correctly.
    """

    name = "fake-e5-model"
    dim = 4
    max_seq_length = 512
    tokenizer_name = "FakeTokenizer"
    requires_prefix = True

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Kind]] = []

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        self.calls.append((list(texts), kind))
        # Encodes (kind, length) into the vector so tests can assert the
        # prefix/kind actually reached here without inspecting internals.
        marker = 1.0 if kind == "query" else -1.0
        return [[marker, float(len(t)), 0.0, 0.0] for t in texts]


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        EMBEDDINGS_MAX_BATCH_SIZE=3,
        EMBEDDINGS_MAX_TOTAL_CHARS=50,
        EMBEDDINGS_MAX_CONCURRENCY=2,
        EMBEDDINGS_QUEUE_TIMEOUT_S=1.0,
    )
