from __future__ import annotations

from notes_bot.clients.llm import FakeLLMClient, LLMResult, NullLLMClient
from notes_bot.enrich import (
    find_duplicate,
    generate_place,
    generate_places,
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


async def test_generate_places_keeps_only_entries_with_a_name():
    llm = ScriptedLLMClient(
        {
            "places": [
                {"name": "Кахелеби", "location_hint": "Кахетинское шоссе"},
                {"name": None, "location_hint": "no name, dropped"},
                {"name": "  ", "location_hint": "blank name, dropped"},
            ]
        }
    )
    places = await generate_places(llm, "text", timeout_s=10)
    assert places == [{"name": "Кахелеби", "location_hint": "Кахетинское шоссе"}]


async def test_generate_places_strips_empty_location_hint():
    llm = ScriptedLLMClient({"places": [{"name": "Ботанический сад", "location_hint": "  "}]})
    places = await generate_places(llm, "text", timeout_s=10)
    assert places == [{"name": "Ботанический сад"}]


async def test_generate_places_keeps_address_separate_from_location_hint():
    """render.render_places builds its Maps query from `address` alone
    when present - the two must never collapse into one field again."""
    llm = ScriptedLLMClient(
        {
            "places": [
                {
                    "name": "Лавка у Лены",
                    "address": "39 Mikheili Tsinamdzghvrishvili St, Tbilisi 0102",
                    "location_hint": "напротив Wine Gallery",
                }
            ]
        }
    )
    places = await generate_places(llm, "text", timeout_s=10)
    assert places == [
        {
            "name": "Лавка у Лены",
            "address": "39 Mikheili Tsinamdzghvrishvili St, Tbilisi 0102",
            "location_hint": "напротив Wine Gallery",
        }
    ]


async def test_generate_places_strips_empty_address():
    llm = ScriptedLLMClient({"places": [{"name": "Fabrika", "address": "  "}]})
    places = await generate_places(llm, "text", timeout_s=10)
    assert places == [{"name": "Fabrika"}]


async def test_generate_places_returns_every_place_no_cap():
    llm = ScriptedLLMClient({"places": [{"name": f"place{i}"} for i in range(10)]})
    places = await generate_places(llm, "text", timeout_s=10)
    assert len(places) == 10


async def test_generate_places_returns_empty_list_when_nothing_parsed():
    assert await generate_places(NullLLMClient(), "text", timeout_s=10) == []


async def test_generate_places_returns_empty_list_when_places_is_not_a_list():
    llm = ScriptedLLMClient({"places": "not a list"})
    assert await generate_places(llm, "text", timeout_s=10) == []


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
