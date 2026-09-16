from __future__ import annotations

import pytest

from translate_service.config import Settings
from translate_service.model import TranslationResult


class FakeTranslationModel:
    """Deterministic stand-in for NllbTranslationModel.

    Real weights are ~2.5GB and slow to load; nothing under test needs
    actual translation, only that the service wires target_lang/errors
    correctly. Mimics the real model's own "skip English" rule so route
    tests can exercise both branches without touching torch.
    """

    name = "fake-nllb-model"

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def translate(self, texts: list[str], target_lang: str) -> list[TranslationResult]:
        self.calls.append((list(texts), target_lang))
        return [
            TranslationResult(text=f"[{target_lang}] {t}", source_lang="ru", translated=True)
            for t in texts
        ]


@pytest.fixture
def fake_model() -> FakeTranslationModel:
    return FakeTranslationModel()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        TRANSLATE_MAX_BATCH_SIZE=3,
        TRANSLATE_MAX_TOTAL_CHARS=50,
        TRANSLATE_MAX_CONCURRENCY=2,
        TRANSLATE_QUEUE_TIMEOUT_S=1.0,
    )
