import pytest
from pydantic import ValidationError

from notes_bot.config import Settings


def _required_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/3")
    monkeypatch.setenv("EMBEDDINGS_URL", "http://notes-embeddings:8000")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")


def test_settings_loads_with_required_env(monkeypatch):
    _required_env(monkeypatch)
    settings = Settings()  # type: ignore[call-arg]
    assert settings.telegram_bot_token == "123:ABC"
    assert settings.embedding_dim == 768
    assert settings.llm_enabled is False
    assert settings.proxy_ai_url == "http://ai-proxy:8787"


def test_settings_fails_fast_without_required_env(monkeypatch):
    """Missing a required variable must fail at construction, not at first
    use — see docs/architecture/05-contracts.md, "Валидация на старте"."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_llm_enabled_can_be_turned_on(monkeypatch):
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_ENABLED", "true")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.llm_enabled is True
