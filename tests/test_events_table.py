from __future__ import annotations

from notes_bot.clients.llm import ImageInput, LLMResult, NullLLMClient
from notes_bot.events_table import ExtractedEvent, extract_events_table


class ScriptedLLMClient:
    """Replies with a fixed parsed payload regardless of prompt — same
    pattern as test_enrich.py's own, but records `images` too since that's
    what this module's tests actually care about."""

    def __init__(self, parsed):
        self._parsed = parsed
        self.calls: list[dict] = []

    async def complete(
        self, *, system, user, json_schema=None, history=None, timeout_s=60.0, images=None
    ):
        self.calls.append({"system": system, "user": user, "images": images})
        return LLMResult(text="", parsed=self._parsed, model="scripted", usage=None)


def _image() -> ImageInput:
    return ImageInput(media_type="image/jpeg", data_base64="Zm9v")


async def test_extract_events_table_passes_images_and_tab_name_through():
    llm = ScriptedLLMClient({"topic_tags": [], "events": []})
    images = [_image(), _image()]
    await extract_events_table(llm, images=images, tab_name="שכבה י'", timeout_s=10)
    assert llm.calls[0]["images"] == images
    assert "שכבה י'" in llm.calls[0]["user"]


async def test_extract_events_table_parses_events_and_topic_tags():
    llm = ScriptedLLMClient(
        {
            "topic_tags": ["Школа", "  "],
            "events": [
                {
                    "date_start": "11/09/2026",
                    "date_end": "13/09/2026",
                    "type": "holiday",
                    "text": "ראש השנה",
                },
                {
                    "date_start": "15/09/2026",
                    "date_end": None,
                    "type": "exam",
                    "text": "מבחן מתמטיקה",
                },
            ],
        }
    )
    table = await extract_events_table(llm, images=[_image()], tab_name="t", timeout_s=10)
    assert table.topic_tags == ["школа"]
    assert table.events == [
        ExtractedEvent(
            date_start="11/09/2026", date_end="13/09/2026", type="holiday", text="ראש השנה"
        ),
        ExtractedEvent(
            date_start="15/09/2026", date_end="15/09/2026", type="exam", text="מבחן מתמטיקה"
        ),
    ]


async def test_extract_events_table_drops_events_missing_text_or_date():
    llm = ScriptedLLMClient(
        {
            "topic_tags": [],
            "events": [
                {"date_start": "01/09/2026", "type": "exam", "text": ""},
                {"date_start": None, "type": "exam", "text": "מבחן"},
                {"date_start": "02/09/2026", "type": "exam", "text": "ok"},
            ],
        }
    )
    table = await extract_events_table(llm, images=[_image()], tab_name="t", timeout_s=10)
    assert len(table.events) == 1
    assert table.events[0].text == "ok"


async def test_extract_events_table_defaults_an_unknown_type_to_event():
    llm = ScriptedLLMClient(
        {
            "topic_tags": [],
            "events": [{"date_start": "01/09/2026", "type": "bogus", "text": "ok"}],
        }
    )
    table = await extract_events_table(llm, images=[_image()], tab_name="t", timeout_s=10)
    assert table.events[0].type == "event"


async def test_extract_events_table_empty_on_unparseable_response():
    table = await extract_events_table(
        NullLLMClient(), images=[_image()], tab_name="t", timeout_s=10
    )
    assert table.topic_tags == []
    assert table.events == []


def test_extracted_event_note_text_single_day():
    event = ExtractedEvent(date_start="01/09/2026", date_end="01/09/2026", type="exam", text="ok")
    assert event.note_text == "01/09/2026: ok"


def test_extracted_event_note_text_date_range():
    event = ExtractedEvent(
        date_start="11/09/2026", date_end="13/09/2026", type="holiday", text="ראש השנה"
    )
    assert event.note_text == "11/09/2026–13/09/2026: ראש השנה"


def test_extracted_event_tag_maps_type_to_russian():
    assert ExtractedEvent("d", "d", "exam", "t").tag == "экзамен"
    assert ExtractedEvent("d", "d", "holiday", "t").tag == "праздник"
    assert ExtractedEvent("d", "d", "event", "t").tag == "мероприятие"
