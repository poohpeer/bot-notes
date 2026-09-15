from notes_bot.domain.classify import classify_text_message


def test_plain_text_with_no_url():
    result = classify_text_message("just a thought, no links here")
    assert result.source_type == "text"
    assert result.source_url is None


def test_youtube_full_url():
    result = classify_text_message("check this out https://www.youtube.com/watch?v=abc123")
    assert result.source_type == "youtube"
    assert result.source_url == "https://www.youtube.com/watch?v=abc123"


def test_youtube_short_url():
    result = classify_text_message("https://youtu.be/abc123")
    assert result.source_type == "youtube"


def test_youtube_mobile_host():
    result = classify_text_message("https://m.youtube.com/watch?v=abc123")
    assert result.source_type == "youtube"


def test_instagram_url():
    result = classify_text_message("https://www.instagram.com/p/abc123/")
    assert result.source_type == "instagram"


def test_map_short_link():
    result = classify_text_message("https://maps.app.goo.gl/xyz123")
    assert result.source_type == "map"


def test_map_goo_gl_maps_path():
    result = classify_text_message("https://goo.gl/maps/xyz123")
    assert result.source_type == "map"


def test_goo_gl_without_maps_path_is_not_a_map_link():
    result = classify_text_message("https://goo.gl/somethingelse")
    assert result.source_type == "page"


def test_maps_google_domain():
    result = classify_text_message("https://maps.google.com/?q=restaurant")
    assert result.source_type == "map"


def test_maps_google_other_tld():
    result = classify_text_message("https://maps.google.co.il/?q=restaurant")
    assert result.source_type == "map"


def test_generic_url_is_a_page():
    result = classify_text_message("https://example.com/some/article")
    assert result.source_type == "page"


def test_url_is_extracted_alongside_surrounding_text():
    """03-ingest.md: both survive — the URL as source_url, the whole
    message as raw_text (done by the caller, not this function) — this only
    confirms the URL is found even with text around it."""
    result = classify_text_message("рекомендую! https://example.com/place отличное место")
    assert result.source_url == "https://example.com/place"


def test_only_the_first_url_is_used():
    result = classify_text_message("https://example.com/a and also https://example.com/b")
    assert result.source_url == "https://example.com/a"


def test_curly_closing_quote_is_stripped():
    """Real production bug: a curly quote (iOS/macOS autocorrect turning "
    into ”) glued onto the end of a pasted link made it past classification
    and broke DNS resolution outright ("Invalid IDNA hostname") deep inside
    the page extractor instead."""
    result = classify_text_message("check this out https://google.com”")
    assert result.source_url == "https://google.com"


def test_straight_quotes_around_the_url_are_stripped():
    result = classify_text_message('recommend "https://example.com/place" here')
    assert result.source_url == "https://example.com/place"


def test_trailing_sentence_punctuation_is_stripped():
    result = classify_text_message("check out https://example.com/place.")
    assert result.source_url == "https://example.com/place"


def test_unbalanced_trailing_paren_is_stripped():
    result = classify_text_message("see this (https://example.com/place)")
    assert result.source_url == "https://example.com/place"


def test_balanced_paren_in_the_url_itself_is_kept():
    """A wiki-style URL can legitimately end in ')' — only strip one that
    isn't balanced by a '(' earlier in the URL."""
    result = classify_text_message("https://en.wikipedia.org/wiki/Foo_(bar)")
    assert result.source_url == "https://en.wikipedia.org/wiki/Foo_(bar)"
