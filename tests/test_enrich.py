from __future__ import annotations

from notes_bot.clients.llm import FakeLLMClient, LLMResult, NullLLMClient
from notes_bot.enrich import (
    Correction,
    apply_corrections,
    enrich,
    find_duplicate,
    generate_summary,
)


class ScriptedLLMClient:
    """Replies with a fixed parsed payload regardless of prompt — simpler
    than FakeLLMClient's substring matching when a test only makes one call."""

    def __init__(self, parsed):
        self._parsed = parsed
        self.calls: list[dict] = []

    async def complete(self, *, system, user, json_schema=None, history=None, timeout_s=60.0):
        self.calls.append({"system": system, "user": user})
        return LLMResult(text="", parsed=self._parsed, model="scripted", usage=None)


def _enrich_payload(**overrides) -> dict:
    payload = {"title": None, "tags": [], "summary": None}
    payload.update(overrides)
    return payload


async def test_generate_summary_returns_stripped_text():
    llm = ScriptedLLMClient({"summary": "  a short summary  "})
    assert await generate_summary(llm, "long text", timeout_s=10) == "a short summary"


async def test_generate_summary_returns_none_for_empty_string():
    llm = ScriptedLLMClient({"summary": "   "})
    assert await generate_summary(llm, "text", timeout_s=10) is None


async def test_enrich_returns_title_tags_summary():
    llm = ScriptedLLMClient(
        _enrich_payload(title="x" * 100, tags=["Food", "#Tbilisi", "  spaced  "], summary="  s  ")
    )
    result = await enrich(llm, "text", source_type="text", timeout_s=10)
    assert len(result.title) == 60
    assert result.tags == ["food", "tbilisi", "spaced"]
    assert result.summary == "s"


async def test_enrich_caps_tags_at_five():
    llm = ScriptedLLMClient(_enrich_payload(tags=[f"tag{i}" for i in range(10)]))
    result = await enrich(llm, "text", source_type="text", timeout_s=10)
    assert len(result.tags) == 5


async def test_enrich_includes_the_note_type_in_the_prompt():
    llm = ScriptedLLMClient(_enrich_payload())
    await enrich(llm, "text", source_type="instagram", timeout_s=10)
    assert "Instagram" in llm.calls[0]["user"]


async def test_enrich_returns_empty_result_on_unparseable_response():
    result = await enrich(NullLLMClient(), "text", source_type="text", timeout_s=10)
    assert result.title is None
    assert result.tags == []
    assert result.summary is None
    assert result.corrections == []
    assert result.place == {}
    assert result.places == []


async def test_enrich_keeps_corrections_only_for_asr_source_types():
    llm = ScriptedLLMClient(
        _enrich_payload(corrections=[{"wrong": "Белиси", "correct": "Тбилиси"}])
    )
    for source_type in ("youtube", "instagram", "voice"):
        result = await enrich(llm, "text", source_type=source_type, timeout_s=10)
        assert result.corrections == [Correction(wrong="Белиси", correct="Тбилиси")]

    for source_type in ("text", "page", "map"):
        result = await enrich(llm, "text", source_type=source_type, timeout_s=10)
        assert result.corrections == []


async def test_enrich_drops_malformed_correction_entries():
    llm = ScriptedLLMClient(
        _enrich_payload(
            corrections=[
                {"wrong": "a", "correct": "b"},
                {"wrong": "", "correct": "c"},
                {"wrong": "d"},
                "not a dict",
            ]
        )
    )
    result = await enrich(llm, "text", source_type="voice", timeout_s=10)
    assert result.corrections == [Correction(wrong="a", correct="b")]


async def test_enrich_place_only_for_map_source_type():
    llm = ScriptedLLMClient(
        _enrich_payload(place={"name": "Хинкальная", "district": None, "cuisine": ""})
    )
    assert (await enrich(llm, "text", source_type="map", timeout_s=10)).place == {
        "name": "Хинкальная"
    }
    assert (await enrich(llm, "text", source_type="instagram", timeout_s=10)).place == {}


async def test_enrich_places_only_for_youtube_and_instagram():
    llm = ScriptedLLMClient(
        _enrich_payload(
            places=[
                {"name": "Кахелеби", "location_hint": "Кахетинское шоссе"},
                {"name": None, "location_hint": "no name, dropped"},
                {"name": "  ", "location_hint": "blank name, dropped"},
            ]
        )
    )
    for source_type in ("youtube", "instagram"):
        result = await enrich(llm, "text", source_type=source_type, timeout_s=10)
        assert result.places == [{"name": "Кахелеби", "location_hint": "Кахетинское шоссе"}]

    assert (await enrich(llm, "text", source_type="voice", timeout_s=10)).places == []


async def test_enrich_places_keeps_address_separate_from_location_hint():
    """render.render_places builds its Maps query from `address` alone
    when present - the two must never collapse into one field again."""
    llm = ScriptedLLMClient(
        _enrich_payload(
            places=[
                {
                    "name": "Лавка у Лены",
                    "address": "39 Mikheili Tsinamdzghvrishvili St, Tbilisi 0102",
                    "location_hint": "напротив Wine Gallery",
                }
            ]
        )
    )
    result = await enrich(llm, "text", source_type="instagram", timeout_s=10)
    assert result.places == [
        {
            "name": "Лавка у Лены",
            "address": "39 Mikheili Tsinamdzghvrishvili St, Tbilisi 0102",
            "location_hint": "напротив Wine Gallery",
        }
    ]


async def test_enrich_places_strips_empty_address_and_hint():
    llm = ScriptedLLMClient(_enrich_payload(places=[{"name": "Fabrika", "address": "  "}]))
    result = await enrich(llm, "text", source_type="youtube", timeout_s=10)
    assert result.places == [{"name": "Fabrika"}]


def test_apply_corrections_replaces_exact_substrings():
    text = "Этим летом я купила квартиру в Белиси. Белиси прекрасен."
    corrected = apply_corrections(text, [Correction(wrong="Белиси", correct="Тбилиси")])
    assert corrected == "Этим летом я купила квартиру в Тбилиси. Тбилиси прекрасен."


def test_apply_corrections_skips_a_wrong_that_does_not_occur():
    text = "some text"
    corrected = apply_corrections(text, [Correction(wrong="nope", correct="whatever")])
    assert corrected == text


def test_apply_corrections_with_no_corrections_is_a_no_op():
    assert apply_corrections("text", []) == "text"


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
