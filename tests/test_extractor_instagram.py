from __future__ import annotations

import os
import tempfile

from notes_bot.db.models import Note
from notes_bot.extractors.instagram import DownloadedMedia, InstagramExtractor


def _note(url: str | None) -> Note:
    return Note(
        id=1,
        user_id=1,
        chat_id=1,
        is_group=False,
        visibility="private",
        source_type="instagram",
        source_url=url,
    )


def _temp_audio_file() -> str:
    # A dedicated subdirectory, not a file loose in the system temp dir —
    # matches _YtDlpDownloader's real tempfile.mkdtemp() layout, which the
    # extractor's cleanup (rmtree of the containing directory) assumes.
    # Using mkstemp() directly in the system temp dir here would make that
    # cleanup call shutil.rmtree() on the whole shared temp directory.
    scratch_dir = tempfile.mkdtemp()
    path = os.path.join(scratch_dir, "audio.wav")
    with open(path, "wb"):
        pass
    return path


class FakeDownloader:
    def __init__(self, media=None, error=None):
        self._media = media
        self._error = error

    def download(self, url, *, max_bytes):
        if self._error:
            raise self._error
        return self._media


class FakeTranscriptionClient:
    def __init__(self, text="", error=None):
        self._text = text
        self._error = error

    def transcribe(self, audio_path):
        if self._error:
            raise self._error
        return self._text


async def test_combines_caption_and_transcript():
    audio_path = _temp_audio_file()
    extractor = InstagramExtractor(
        transcription_client=FakeTranscriptionClient(text="the transcript"),
        downloader=FakeDownloader(
            media=DownloadedMedia(caption="a caption", audio_path=audio_path, duration=30)
        ),
    )
    result = await extractor.extract(_note("https://www.instagram.com/p/x/"))
    assert result.text == "a caption\n\nthe transcript"


async def test_cleans_up_the_downloaded_file_and_its_scratch_directory():
    audio_path = _temp_audio_file()
    scratch_dir = os.path.dirname(audio_path)
    extractor = InstagramExtractor(
        transcription_client=FakeTranscriptionClient(text="t"),
        downloader=FakeDownloader(
            media=DownloadedMedia(caption="c", audio_path=audio_path, duration=10)
        ),
    )
    await extractor.extract(_note("https://www.instagram.com/p/x/"))
    assert not os.path.exists(audio_path)
    assert not os.path.exists(scratch_dir)


async def test_caption_alone_when_no_audio():
    extractor = InstagramExtractor(
        transcription_client=FakeTranscriptionClient(text="unused"),
        downloader=FakeDownloader(
            media=DownloadedMedia(caption="just a caption", audio_path=None, duration=None)
        ),
    )
    result = await extractor.extract(_note("https://www.instagram.com/p/x/"))
    assert result.text == "just a caption"


async def test_skips_transcription_for_audio_over_max_duration():
    audio_path = _temp_audio_file()
    transcription = FakeTranscriptionClient(text="should not appear")
    extractor = InstagramExtractor(
        transcription_client=transcription,
        downloader=FakeDownloader(
            media=DownloadedMedia(caption="a long post", audio_path=audio_path, duration=9999)
        ),
        max_audio_seconds=1200,
    )
    result = await extractor.extract(_note("https://www.instagram.com/p/x/"))
    assert result.text == "a long post"
    assert "should not appear" not in result.text
    assert result.meta["transcription_skipped"]


async def test_download_failure_degrades_to_empty_result():
    extractor = InstagramExtractor(
        transcription_client=FakeTranscriptionClient(),
        downloader=FakeDownloader(error=RuntimeError("post is private")),
    )
    result = await extractor.extract(_note("https://www.instagram.com/p/x/"))
    assert result.text == ""
    assert "post is private" in result.meta["error"]


async def test_transcription_failure_still_keeps_the_caption():
    audio_path = _temp_audio_file()
    extractor = InstagramExtractor(
        transcription_client=FakeTranscriptionClient(error=RuntimeError("bad audio")),
        downloader=FakeDownloader(
            media=DownloadedMedia(caption="the caption survives", audio_path=audio_path, duration=5)
        ),
    )
    result = await extractor.extract(_note("https://www.instagram.com/p/x/"))
    assert result.text == "the caption survives"


async def test_missing_source_url_degrades_to_empty_result():
    extractor = InstagramExtractor(transcription_client=FakeTranscriptionClient())
    result = await extractor.extract(_note(None))
    assert result.text == ""
    assert result.meta["error"] == "no source_url"


async def test_no_caption_and_no_transcript_degrades_to_empty_result():
    extractor = InstagramExtractor(
        transcription_client=FakeTranscriptionClient(text=""),
        downloader=FakeDownloader(
            media=DownloadedMedia(caption=None, audio_path=None, duration=None)
        ),
    )
    result = await extractor.extract(_note("https://www.instagram.com/p/x/"))
    assert result.text == ""
    assert "error" in result.meta
