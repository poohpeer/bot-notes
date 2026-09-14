from __future__ import annotations

import json

import httpx
import pytest

from notes_bot.clients.llm import (
    FakeLLMClient,
    LLMResult,
    LLMServiceError,
    NullLLMClient,
    ProxyAILLMClient,
)


def _client(handler) -> ProxyAILLMClient:
    return ProxyAILLMClient("http://ai-proxy:8787", transport=httpx.MockTransport(handler))


def _ok(result: str, model: str = "gpt-5-codex") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "provider": "codex",
            "model": model,
            "output_format": "text",
            "result": result,
            "structured_output": None,
            "tool_calls": [],
            "metadata": {"cost_usd": None, "duration_ms": 100, "exit_code": 0, "raw_envelope": {}},
        },
    )


async def test_null_client_always_returns_an_empty_result():
    result = await NullLLMClient().complete(system="s", user="u")
    assert result == LLMResult(text="", parsed=None, model=None, usage=None)


async def test_fake_client_replies_by_prompt_substring():
    client = FakeLLMClient()
    client.add_reply(
        "give me a title", LLMResult(text="A Title", parsed=None, model="fake", usage=None)
    )
    result = await client.complete(system="s", user="please give me a title now")
    assert result.text == "A Title"


async def test_fake_client_records_calls():
    client = FakeLLMClient()
    await client.complete(system="sys", user="usr", json_schema={"type": "object"})
    assert client.calls == [{"system": "sys", "user": "usr", "json_schema": {"type": "object"}}]


async def test_proxy_client_sends_provider_codex_and_text_format():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return _ok("hello")

    result = await _client(handler).complete(system="sys", user="usr")
    assert seen["body"]["provider"] == "codex"
    assert seen["body"]["output_format"] == "text"
    assert seen["body"]["prompt"] == "usr"
    assert "model" not in seen["body"]  # never selected — codex rejects it
    assert result.text == "hello"
    assert result.model == "gpt-5-codex"


async def test_proxy_client_without_schema_leaves_parsed_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok("just prose, not json")

    result = await _client(handler).complete(system="s", user="u")
    assert result.parsed is None
    assert result.text == "just prose, not json"


async def test_proxy_client_parses_json_from_a_fenced_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok('```json\n{"title": "Хинкали"}\n```')

    result = await _client(handler).complete(system="s", user="u", json_schema={"type": "object"})
    assert result.parsed == {"title": "Хинкали"}


async def test_proxy_client_parses_bare_json_without_fence():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok('{"tags": ["food", "tbilisi"]}')

    result = await _client(handler).complete(system="s", user="u", json_schema={"type": "object"})
    assert result.parsed == {"tags": ["food", "tbilisi"]}


async def test_proxy_client_retries_once_on_bad_json_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return _ok("not json at all")
        return _ok('{"title": "fixed"}')

    result = await _client(handler).complete(system="s", user="u", json_schema={"type": "object"})
    assert len(calls) == 2
    assert result.parsed == {"title": "fixed"}
    body2 = json.loads(calls[1].read())
    assert "JSON" in body2["system"]


async def test_proxy_client_gives_up_after_one_retry():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _ok("still not json")

    result = await _client(handler).complete(system="s", user="u", json_schema={"type": "object"})
    assert len(calls) == 2
    assert result.parsed is None


async def test_json_schema_instruction_goes_in_system_not_user():
    """Notes text lives in `user`/data sections, never mixed with
    instructions — see 05-contracts.md's prompt requirements."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return _ok('{"a": 1}')

    await _client(handler).complete(
        system="base instructions", user="the note text itself", json_schema={"type": "object"}
    )
    assert seen["body"]["prompt"] == "the note text itself"
    assert "JSON" in seen["body"]["system"]


async def test_quota_exhausted_is_flagged_separately_from_other_retryable_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"error": {"type": "quota_exhausted", "message": "no accounts left"}}
        )

    with pytest.raises(LLMServiceError) as exc_info:
        await _client(handler).complete(system="s", user="u")
    assert exc_info.value.is_quota_exhausted is True
    assert exc_info.value.retryable is True


async def test_codex_quota_exhausted_falls_back_to_claude_code():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        calls.append(body["provider"])
        if body["provider"] == "codex":
            return httpx.Response(
                503, json={"error": {"type": "quota_exhausted", "message": "no accounts left"}}
            )
        return _ok("from claude", model="claude-sonnet-5")

    result = await _client(handler).complete(system="s", user="u")
    assert calls == ["codex", "claude_code"]
    assert result.text == "from claude"
    assert result.model == "claude-sonnet-5"


async def test_fallback_is_not_used_for_non_quota_errors():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        calls.append(body["provider"])
        return httpx.Response(504, json={"error": {"type": "timeout", "message": "too slow"}})

    with pytest.raises(LLMServiceError):
        await _client(handler).complete(system="s", user="u")
    assert calls == ["codex"]  # never tried claude_code — timeout isn't quota_exhausted


async def test_when_both_providers_are_exhausted_the_fallbacks_error_surfaces():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"error": {"type": "quota_exhausted", "message": "no accounts left"}}
        )

    with pytest.raises(LLMServiceError) as exc_info:
        await _client(handler).complete(system="s", user="u")
    assert exc_info.value.is_quota_exhausted is True


async def test_timeout_error_is_retryable_but_not_quota_exhausted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(504, json={"error": {"type": "timeout", "message": "too slow"}})

    with pytest.raises(LLMServiceError) as exc_info:
        await _client(handler).complete(system="s", user="u")
    assert exc_info.value.is_quota_exhausted is False
    assert exc_info.value.retryable is True


async def test_missing_schema_error_is_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"type": "missing_schema", "message": "need json_schema"}}
        )

    with pytest.raises(LLMServiceError) as exc_info:
        await _client(handler).complete(system="s", user="u")
    assert exc_info.value.retryable is False


async def test_connection_failure_raises_llm_service_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMServiceError) as exc_info:
        await _client(handler).complete(system="s", user="u")
    assert exc_info.value.retryable is True
