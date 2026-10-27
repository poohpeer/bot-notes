from __future__ import annotations

import base64

from notes_bot.clients.llm import ImageInput, LLMResult, LLMServiceError, NullLLMClient
from notes_bot.clients.telegram_files import TelegramFileError
from notes_bot.db.models import Note
from notes_bot.extractors.image import ImageExtractor, join_file_ids, split_file_ids


def _note(source_url: str | None, *, raw_text: str | None = None) -> Note:
    return Note(
        id=1,
        user_id=1,
        chat_id=1,
        is_group=False,
        visibility="private",
        source_type="image",
        source_url=source_url,
        raw_text=raw_text,
    )


class FakeFileClient:
    def __init__(self, *, failing: set[str] | None = None):
        self._failing = failing or set()
        self.downloaded: list[str] = []

    async def download(self, file_id):
        if file_id in self._failing:
            raise TelegramFileError(f"cannot download {file_id}")
        self.downloaded.append(file_id)
        return f"bytes-of-{file_id}".encode()


class ScriptedVisionClient:
    def __init__(self, parsed=None, fail_with=None):
        self._parsed = parsed
        self._fail_with = fail_with
        self.calls: list[dict] = []

    async def complete(
        self, *, system, user, json_schema=None, history=None, timeout_s=60.0, images=None
    ):
        self.calls.append(
            {"system": system, "user": user, "timeout_s": timeout_s, "images": images}
        )
        if self._fail_with:
            raise self._fail_with
        return LLMResult(text="", parsed=self._parsed, model="scripted", usage=None)


def _extractor(llm, file_client=None, timeout_s=77.0):
    return ImageExtractor(
        file_client=file_client or FakeFileClient(), llm_client=llm, timeout_s=timeout_s
    )


def test_file_ids_round_trip_through_source_url():
    assert split_file_ids(join_file_ids(["a", "b", "c"])) == ["a", "b", "c"]
    assert split_file_ids(None) == []
    assert split_file_ids("") == []
    assert split_file_ids(" a \n\n b \n") == ["a", "b"]


async def test_combines_caption_description_and_visible_text_in_that_order():
    llm = ScriptedVisionClient({"description": "Страница ресторана", "text": "Mamacita\nTacos 12€"})
    result = await _extractor(llm).extract(_note("f1", raw_text="хочу сюда"))
    assert result.text == "хочу сюда\n\nСтраница ресторана\n\nMamacita\nTacos 12€"
    assert result.meta == {}


async def test_an_album_is_one_vision_call_with_every_image():
    files = FakeFileClient()
    llm = ScriptedVisionClient({"description": "d", "text": "t"})
    await _extractor(llm, files, timeout_s=77.0).extract(_note(join_file_ids(["f1", "f2", "f3"])))

    assert files.downloaded == ["f1", "f2", "f3"]
    [call] = llm.calls
    assert call["timeout_s"] == 77.0
    assert [i.data_base64 for i in call["images"]] == [
        base64.b64encode(f"bytes-of-{f}".encode()).decode() for f in ("f1", "f2", "f3")
    ]
    assert all(isinstance(i, ImageInput) and i.media_type == "image/jpeg" for i in call["images"])
    assert call["user"].strip()  # ai-proxy rejects an empty prompt


async def test_one_lost_frame_does_not_drop_the_rest_of_the_album():
    files = FakeFileClient(failing={"f2"})
    llm = ScriptedVisionClient({"description": "d", "text": "t"})
    result = await _extractor(llm, files).extract(_note(join_file_ids(["f1", "f2", "f3"])))

    assert result.text == "d\n\nt"
    [call] = llm.calls
    assert len(call["images"]) == 2


async def test_nothing_downloadable_degrades_to_the_caption_with_an_error():
    llm = ScriptedVisionClient({"description": "d", "text": "t"})
    result = await _extractor(llm, FakeFileClient(failing={"f1"})).extract(
        _note("f1", raw_text="a caption")
    )
    assert result.text == "a caption"
    assert result.meta["error"] == "download failed"
    assert llm.calls == []


async def test_a_failed_vision_call_degrades_to_the_caption_with_an_error():
    llm = ScriptedVisionClient(fail_with=LLMServiceError(503, "quota_exhausted", "no accounts"))
    result = await _extractor(llm).extract(_note("f1", raw_text="a caption"))
    assert result.text == "a caption"
    assert "quota_exhausted" in result.meta["error"]


async def test_a_disabled_llm_degrades_rather_than_pretending_to_have_read_the_image():
    result = await _extractor(NullLLMClient()).extract(_note("f1", raw_text="a caption"))
    assert result.text == "a caption"
    assert result.meta["error"] == "no vision reply"


async def test_an_unreadable_image_without_a_caption_yields_an_error_and_no_text():
    llm = ScriptedVisionClient({"description": "  ", "text": ""})
    result = await _extractor(llm).extract(_note("f1"))
    assert result.text == ""
    assert result.meta["error"] == "nothing recognised on the image"


async def test_non_string_fields_in_the_reply_are_ignored_not_crashed_on():
    llm = ScriptedVisionClient({"description": 5, "text": ["x"]})
    result = await _extractor(llm).extract(_note("f1", raw_text="cap"))
    assert result.text == "cap"


async def test_a_note_without_a_file_id_is_an_error_not_a_crash():
    result = await _extractor(ScriptedVisionClient({})).extract(_note(None))
    assert result.text == ""
    assert result.meta["error"] == "no file_id"
