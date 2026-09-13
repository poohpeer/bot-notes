from __future__ import annotations

from notes_bot.clients.llm import FakeLLMClient, LLMResult, NullLLMClient
from notes_bot.enrich import (
    find_duplicate,
    generate_place,
    generate_summary,
    generate_tags,
    generate_title,
)


class ScriptedLLMClient:
    """Replies with a fixed parsed payload regardless of prompt — simpler
    than FakeLLMClient's substring matching when a test only makes one call."""

    def __init__(self, parsed):
        self._parsed = parsed

    async def complete(self, *, system, user, json_schema=None, history=None, timeout_s=60.0):
        return LLMResult(text="", parsed=self._parsed, model="scripted", usage=None)


async def test_generate_title_returns_the_parsed_title():
    llm = ScriptedLLMClient({"title": "Хинкальная на Руставели"})
    title = await generate_title(llm, "some note text", timeout_s=10)
    assert title == "Хинкальная на Руставели"


async def test_generate_title_truncates_to_60_chars():
    llm = ScriptedLLMClient({"title": "x" * 100})
    title = await generate_title(llm, "text", timeout_s=10)
    assert len(title) == 60


async def test_generate_title_returns_none_when_llm_returns_null():
    llm = ScriptedLLMClient({"title": None})
    assert await generate_title(llm, "text", timeout_s=10) is None


async def test_generate_title_returns_none_on_unparseable_response():
    assert await generate_title(NullLLMClient(), "text", timeout_s=10) is None


async def test_generate_tags_lowercases_and_strips_hash():
    llm = ScriptedLLMClient({"tags": ["Food", "#Tbilisi", "  spaced  "]})
    tags = await generate_tags(llm, "text", timeout_s=10)
    assert tags == ["food", "tbilisi", "spaced"]


async def test_generate_tags_caps_at_five():
    llm = ScriptedLLMClient({"tags": [f"tag{i}" for i in range(10)]})
    tags = await generate_tags(llm, "text", timeout_s=10)
    assert len(tags) == 5


async def test_generate_tags_returns_empty_list_when_nothing_parsed():
    assert await generate_tags(NullLLMClient(), "text", timeout_s=10) == []


async def test_generate_summary_returns_stripped_text():
    llm = ScriptedLLMClient({"summary": "  a short summary  "})
    assert await generate_summary(llm, "long text", timeout_s=10) == "a short summary"


async def test_generate_summary_returns_none_for_empty_string():
    llm = ScriptedLLMClient({"summary": "   "})
    assert await generate_summary(llm, "text", timeout_s=10) is None


async def test_generate_place_drops_null_and_empty_fields():
    llm = ScriptedLLMClient({"name": "Хинкальная", "district": None, "cuisine": ""})
    place = await generate_place(llm, "text", timeout_s=10)
    assert place == {"name": "Хинкальная"}


async def test_generate_place_returns_empty_dict_when_nothing_parsed():
    assert await generate_place(NullLLMClient(), "text", timeout_s=10) == {}


async def test_find_duplicate_returns_none_with_no_candidates():
    llm = FakeLLMClient()
    result = await find_duplicate(llm, new_text="x", candidates=[], timeout_s=10)
    assert result is None
    assert llm.calls == []  # never even asked — nothing to compare against


async def test_find_duplicate_returns_the_matched_id():
    llm = ScriptedLLMClient({"is_duplicate": True, "duplicate_note_id": 42})
    result = await find_duplicate(llm, new_text="x", candidates=[(42, "a"), (7, "b")], timeout_s=10)
    assert result == 42


async def test_find_duplicate_returns_none_when_not_a_duplicate():
    llm = ScriptedLLMClient({"is_duplicate": False, "duplicate_note_id": None})
    result = await find_duplicate(llm, new_text="x", candidates=[(42, "a")], timeout_s=10)
    assert result is None


async def test_find_duplicate_ignores_a_hallucinated_id_not_in_candidates():
    llm = ScriptedLLMClient({"is_duplicate": True, "duplicate_note_id": 999})
    result = await find_duplicate(llm, new_text="x", candidates=[(42, "a")], timeout_s=10)
    assert result is None


async def test_find_duplicate_returns_none_when_llm_produces_nothing():
    result = await find_duplicate(
        NullLLMClient(), new_text="x", candidates=[(42, "a")], timeout_s=10
    )
    assert result is None
