from urllib.parse import unquote

from notes_bot.bot.render import (
    RenderableHit,
    render_capture_mode_changed,
    render_debug_processing_done,
    render_debug_toggle_confirmation,
    render_delete_confirmation_prompt,
    render_delete_refused,
    render_edit_refused,
    render_edit_saved,
    render_group_note_saved,
    render_group_welcome,
    render_list_empty,
    render_no_more_results,
    render_note_deleted,
    render_note_restored,
    render_places,
    render_privacy_toggle_confirmation,
    render_processing_failed,
    render_search_card,
    render_search_debug_empty,
    render_search_expired,
    render_search_summary_card,
    render_trash_empty,
)


def _hit(**kw) -> RenderableHit:
    defaults = dict(
        note_id=1,
        title=None,
        source_type="text",
        source_url=None,
        tags=[],
        chunk_text="some note text",
        is_owner=True,
        structured={},
    )
    defaults.update(kw)
    return RenderableHit(**defaults)


def test_privacy_confirmation_private():
    assert "приватная" in render_privacy_toggle_confirmation("private")


def test_privacy_confirmation_public():
    assert "публичная" in render_privacy_toggle_confirmation("public")


def test_debug_toggle_confirmation_on():
    assert "включён" in render_debug_toggle_confirmation(enabled=True)


def test_debug_toggle_confirmation_off():
    assert "выключен" in render_debug_toggle_confirmation(enabled=False)


def test_debug_processing_done_includes_elapsed_seconds():
    text = render_debug_processing_done(elapsed_s=12.34, summary=None, indexed_text="text")
    assert "12.3" in text


def test_debug_processing_done_prefers_the_llm_summary_over_the_raw_text():
    text = render_debug_processing_done(
        elapsed_s=1.0,
        summary="Видео о том, как женщина ищет мужа",
        indexed_text="a" * 500,  # would dominate the message if it were used instead
    )
    assert "Видео о том, как женщина ищет мужа" in text
    assert "a" * 150 not in text


def test_debug_processing_done_falls_back_to_a_text_preview_without_a_summary():
    text = render_debug_processing_done(
        elapsed_s=1.0, summary=None, indexed_text="Обзор уличной еды в Тбилиси, хинкали и хачапури"
    )
    assert "Обзор уличной еды в Тбилиси" in text


def test_debug_processing_done_truncates_a_long_fallback_preview():
    text = render_debug_processing_done(elapsed_s=1.0, summary=None, indexed_text="a" * 500)
    lines = text.splitlines()
    assert len(lines[1]) <= 150


def test_debug_processing_done_omits_the_preview_line_when_both_are_empty():
    text = render_debug_processing_done(elapsed_s=1.0, summary=None, indexed_text="")
    assert text.splitlines() == ["⏱ Обработка завершена. Заняло: 1.0с"]


def test_processing_failed_message():
    assert render_processing_failed()


def test_search_debug_empty_lists_near_misses_with_distances():
    text = render_search_debug_empty(
        [("Хинкальная на Руставели", 0.234), ("Велодорожка в парке", 0.256)], max_distance=0.18
    )
    assert "distance=0.234" in text
    assert "distance=0.256" in text
    assert "0.18" in text


def test_search_debug_empty_with_no_near_misses_says_the_corpus_is_empty():
    text = render_search_debug_empty([], max_distance=0.18)
    assert "нет заметок" in text


def test_card_uses_title_when_present():
    card = render_search_card(_hit(title="Хинкальная на Руставели"))
    assert card.startswith("📝 Хинкальная на Руставели")


def test_card_falls_back_to_chunk_text_when_no_title():
    card = render_search_card(_hit(title=None, chunk_text="a short note"))
    assert "a short note" in card.splitlines()[0]


def test_card_uses_source_icon():
    card = render_search_card(_hit(source_type="youtube", title="A video"))
    assert card.startswith("▶️")


def test_card_includes_tags_with_hash_prefix():
    card = render_search_card(_hit(title="x", tags=["georgia", "food"]))
    assert "#georgia #food" in card


def test_card_omits_tags_line_when_empty():
    card = render_search_card(_hit(title="x", tags=[]))
    assert "#" not in card


def test_card_includes_source_url_when_present():
    card = render_search_card(_hit(title="x", source_url="https://example.com/a"))
    assert "https://example.com/a" in card


def test_card_fragment_is_truncated():
    long_text = "w" * 500
    card = render_search_card(_hit(title="x", chunk_text=long_text))
    lines = card.splitlines()
    assert len(lines[1]) <= 200
    assert lines[1].endswith("…")


def test_card_for_a_not_yet_extracted_link_note_shows_the_url_once():
    """Before process_note/enrich_note finish, chunk_text (raw_text) is the
    pasted link itself, verbatim — without special-casing this, the card
    showed it three times: truncated as the heading, in full as the
    "fragment", and again as the source_url line."""
    url = "https://www.instagram.com/reel/DdOdsrnIeBX/?stkn=MTB3YWhyZGUwUwerRxaQ=="
    card = render_search_card(
        _hit(title=None, source_type="instagram", chunk_text=url, source_url=url)
    )
    assert card.count(url) == 1
    assert card.startswith("📷 Instagram-видео")


def test_card_for_a_titled_link_note_still_shows_the_url_once():
    url = "https://www.instagram.com/reel/DdOdsrnIeBX/"
    card = render_search_card(
        _hit(title="A Reel", source_type="instagram", chunk_text=url, source_url=url)
    )
    assert card.count(url) == 1
    assert card.startswith("📷 A Reel")


def test_card_includes_places_as_maps_links():
    """The place name is the clickable text (HTML <a> — see render_places'
    own docstring on why callers must send with parse_mode="HTML"), not a
    raw URL printed next to it."""
    structured = {"places": [{"name": "Кахелеби", "location_hint": "Кахетинское шоссе"}]}
    card = render_search_card(_hit(title="x", structured=structured))
    assert '📍 <a href="https://www.google.com/maps/search/?api=1&amp;query=' in card
    assert ">Кахелеби</a>" in card
    query_param = card.split("query=")[1].split('"')[0]
    assert "Кахелеби Кахетинское шоссе" in unquote(query_param)


def test_card_omits_places_line_when_none():
    card = render_search_card(_hit(title="x", structured={}))
    assert "📍" not in card


def test_summary_card_uses_summary_when_present():
    card = render_search_summary_card(
        _hit(title="x", summary="Короткое содержание", chunk_text="a" * 500)
    )
    assert "Короткое содержание" in card
    assert "a" * 200 not in card  # the long fragment never appears alongside a real summary


def test_summary_card_falls_back_to_a_short_fragment_without_a_summary():
    card = render_search_summary_card(_hit(title="x", summary=None, chunk_text="a" * 500))
    lines = card.splitlines()
    assert len(lines[1]) <= 120


def test_summary_card_omits_tags_source_url_and_places():
    """The short card is deliberately short — those stay in the full card
    behind "Подробнее"."""
    card = render_search_summary_card(
        _hit(
            title="x",
            tags=["food"],
            source_url="https://example.com/a",
            structured={"places": [{"name": "Кахелеби"}]},
        )
    )
    assert "#food" not in card
    assert "https://example.com/a" not in card
    assert "📍" not in card


def test_summary_card_for_a_not_yet_extracted_link_note_shows_only_the_heading():
    url = "https://www.instagram.com/reel/abc/"
    card = render_search_summary_card(
        _hit(title=None, source_type="instagram", chunk_text=url, source_url=url)
    )
    assert card == "📷 Instagram-видео"


def test_render_places_skips_entries_with_no_name():
    lines = render_places({"places": [{"location_hint": "no name here"}]})
    assert lines == []


def test_render_places_works_without_location_hint():
    lines = render_places({"places": [{"name": "Ботанический сад"}]})
    assert len(lines) == 1
    assert lines[0].startswith('📍 <a href="https://www.google.com/maps/search/')
    assert lines[0].endswith(">Ботанический сад</a>")


def test_render_places_on_malformed_structured_is_empty():
    assert render_places({}) == []
    assert render_places({"places": "not a list"}) == []
    assert render_places({"places": ["not a dict"]}) == []


def test_no_more_results_message():
    assert render_no_more_results()


def test_search_expired_message():
    assert render_search_expired()


def test_capture_mode_all_mentions_botfather_privacy():
    text = render_capture_mode_changed("all")
    assert "privacy" in text.lower() or "BotFather" in text


def test_capture_mode_mentions_and_replies_says_so():
    text = render_capture_mode_changed("mentions_and_replies")
    assert "упоминани" in text.lower()


def test_group_welcome_message():
    assert render_group_welcome()


def test_group_note_saved_message():
    assert render_group_note_saved()


def test_list_and_trash_empty_messages():
    assert render_list_empty()
    assert render_trash_empty()


def test_delete_flow_messages():
    assert render_delete_confirmation_prompt()
    assert render_note_deleted()
    assert render_delete_refused()
    assert render_note_restored()


def test_edit_flow_messages():
    assert render_edit_saved()
    assert render_edit_refused()
