from __future__ import annotations

from notes_bot.db.models import Note
from notes_bot.extractors.youtube import YoutubeExtractor, extract_video_id


def _note(url: str | None) -> Note:
    return Note(
        id=1,
        user_id=1,
        chat_id=1,
        is_group=False,
        visibility="private",
        source_type="youtube",
        source_url=url,
    )


class FakeMetadataClient:
    def __init__(self, metadata=None, error=None):
        self._metadata = metadata or {}
        self._error = error

    def get_metadata(self, url):
        if self._error:
            raise self._error
        return self._metadata


class FakeTranscriptClient:
    def __init__(self, segments=None, error=None):
        self._segments = segments
        self._error = error

    def get_transcript(self, video_id):
        if self._error:
            raise self._error
        return self._segments


def test_extract_video_id_from_watch_url():
    assert extract_video_id("https://www.youtube.com/watch?v=abc123XYZ") == "abc123XYZ"


def test_extract_video_id_from_short_url():
    assert extract_video_id("https://youtu.be/abc123XYZ") == "abc123XYZ"


def test_extract_video_id_from_short_url_with_trailing_path():
    assert extract_video_id("https://youtu.be/abc123XYZ?t=30") == "abc123XYZ"


def test_extract_video_id_from_shorts_url():
    assert extract_video_id("https://www.youtube.com/shorts/abc123XYZ") == "abc123XYZ"


def test_extract_video_id_returns_none_for_a_non_video_url():
    assert extract_video_id("https://www.youtube.com/") is None


async def test_combines_title_description_and_transcript():
    extractor = YoutubeExtractor(
        metadata_client=FakeMetadataClient(
            metadata={"title": "Хинкали рецепт", "description": "Как готовить хинкали дома"}
        ),
        transcript_client=FakeTranscriptClient(
            segments=[{"text": "Берём тесто"}, {"text": "и мясо"}]
        ),
    )
    result = await extractor.extract(_note("https://youtu.be/abc123"))
    assert "Хинкали рецепт" in result.text
    assert "Как готовить хинкали дома" in result.text
    assert "Берём тесто и мясо" in result.text
    assert result.title == "Хинкали рецепт"
    assert result.meta["has_subtitles"] is True


async def test_works_without_a_transcript():
    extractor = YoutubeExtractor(
        metadata_client=FakeMetadataClient(metadata={"title": "No captions here"}),
        transcript_client=FakeTranscriptClient(error=RuntimeError("no transcript available")),
    )
    result = await extractor.extract(_note("https://youtu.be/abc123"))
    assert result.text == "No captions here"
    assert result.meta["has_subtitles"] is False


async def test_metadata_failure_degrades_to_empty_result():
    extractor = YoutubeExtractor(
        metadata_client=FakeMetadataClient(error=RuntimeError("video unavailable")),
        transcript_client=FakeTranscriptClient(),
    )
    result = await extractor.extract(_note("https://youtu.be/abc123"))
    assert result.text == ""
    assert "error" in result.meta


async def test_missing_source_url_degrades_to_empty_result():
    extractor = YoutubeExtractor(
        metadata_client=FakeMetadataClient(), transcript_client=FakeTranscriptClient()
    )
    result = await extractor.extract(_note(None))
    assert result.text == ""
    assert result.meta["error"] == "no source_url"


async def test_unparseable_url_degrades_to_empty_result():
    extractor = YoutubeExtractor(
        metadata_client=FakeMetadataClient(), transcript_client=FakeTranscriptClient()
    )
    result = await extractor.extract(_note("https://www.youtube.com/"))
    assert result.text == ""
    assert "video id" in result.meta["error"]


async def test_no_title_description_or_transcript_degrades_to_empty_result():
    extractor = YoutubeExtractor(
        metadata_client=FakeMetadataClient(metadata={}),
        transcript_client=FakeTranscriptClient(segments=None),
    )
    result = await extractor.extract(_note("https://youtu.be/abc123"))
    assert result.text == ""
    assert "error" in result.meta
