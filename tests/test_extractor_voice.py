from __future__ import annotations

from notes_bot.clients.telegram_files import TelegramFileError
from notes_bot.db.models import Note
from notes_bot.extractors.voice import VoiceExtractor


def _note(file_id: str | None, *, raw_text: str | None = None) -> Note:
    return Note(
        id=1,
        user_id=1,
        chat_id=1,
        is_group=False,
        visibility="private",
        source_type="voice",
        source_url=file_id,
        raw_text=raw_text,
    )


class FakeFileClient:
    def __init__(self, data=b"audio-bytes", error=None):
        self._data = data
        self._error = error

    async def download(self, file_id):
        if self._error:
            raise self._error
        return self._data


class FakeTranscriptionClient:
    def __init__(self, text="", error=None):
        self._text = text
        self._error = error
        self.paths_seen: list[str] = []

    def transcribe(self, audio_path):
        self.paths_seen.append(audio_path)
        if self._error:
            raise self._error
        return self._text


async def test_transcribes_and_returns_text():
    extractor = VoiceExtractor(
        file_client=FakeFileClient(),
        transcription_client=FakeTranscriptionClient(text="привет как дела"),
    )
    result = await extractor.extract(_note("file123"))
    assert result.text == "привет как дела"


async def test_combines_caption_and_transcript():
    extractor = VoiceExtractor(
        file_client=FakeFileClient(),
        transcription_client=FakeTranscriptionClient(text="the transcript"),
    )
    result = await extractor.extract(_note("file123", raw_text="a caption"))
    assert result.text == "a caption\n\nthe transcript"


async def test_caption_alone_is_enough_when_transcript_is_empty():
    extractor = VoiceExtractor(
        file_client=FakeFileClient(),
        transcription_client=FakeTranscriptionClient(text=""),
    )
    result = await extractor.extract(_note("file123", raw_text="just a caption"))
    assert result.text == "just a caption"


async def test_writes_and_cleans_up_a_temp_file():
    transcription = FakeTranscriptionClient(text="ok")
    extractor = VoiceExtractor(file_client=FakeFileClient(), transcription_client=transcription)
    await extractor.extract(_note("file123"))

    import os

    assert len(transcription.paths_seen) == 1
    path = transcription.paths_seen[0]
    assert not os.path.exists(path)


async def test_missing_file_id_degrades_to_empty_result():
    extractor = VoiceExtractor(
        file_client=FakeFileClient(), transcription_client=FakeTranscriptionClient()
    )
    result = await extractor.extract(_note(None))
    assert result.text == ""
    assert result.meta["error"] == "no file_id"


async def test_download_failure_degrades_to_empty_result():
    extractor = VoiceExtractor(
        file_client=FakeFileClient(error=TelegramFileError("too large")),
        transcription_client=FakeTranscriptionClient(),
    )
    result = await extractor.extract(_note("file123"))
    assert result.text == ""
    assert "too large" in result.meta["error"]


async def test_transcription_failure_degrades_to_empty_result():
    extractor = VoiceExtractor(
        file_client=FakeFileClient(),
        transcription_client=FakeTranscriptionClient(error=RuntimeError("bad audio")),
    )
    result = await extractor.extract(_note("file123"))
    assert result.text == ""
    assert "bad audio" in result.meta["error"]


async def test_empty_transcript_and_no_caption_degrades_to_empty_result():
    extractor = VoiceExtractor(
        file_client=FakeFileClient(), transcription_client=FakeTranscriptionClient(text="")
    )
    result = await extractor.extract(_note("file123"))
    assert result.text == ""
    assert "error" in result.meta
