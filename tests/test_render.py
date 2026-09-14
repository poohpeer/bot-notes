from notes_bot.bot.render import (
    RenderableHit,
    render_no_more_results,
    render_privacy_toggle_confirmation,
    render_search_card,
    render_search_expired,
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
    )
    defaults.update(kw)
    return RenderableHit(**defaults)


def test_privacy_confirmation_private():
    assert "приватная" in render_privacy_toggle_confirmation("private")


def test_privacy_confirmation_public():
    assert "публичная" in render_privacy_toggle_confirmation("public")


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


def test_no_more_results_message():
    assert render_no_more_results()


def test_search_expired_message():
    assert render_search_expired()
