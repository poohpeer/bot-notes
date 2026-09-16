"""Pure tests for model.py's language-code table and detection helper —
deliberately not instantiating NllbTranslationModel (that pulls in torch +
downloads ~2.5GB of weights); see conftest.py's FakeTranslationModel for
what the route layer needs instead."""

from __future__ import annotations

from translate_service.model import _FLORES_BY_ISO639_1, _ISO_BY_FLORES, _detect_iso639_1


def test_flores_table_is_a_bijection():
    """Every entry must round-trip — main.py's target_iso lookup
    (_ISO_BY_FLORES.get(target_lang)) silently returns None on a collision,
    which would wrongly disable the "skip already-target-language" check."""
    assert len(_FLORES_BY_ISO639_1) == len(_ISO_BY_FLORES)
    for iso, flores in _FLORES_BY_ISO639_1.items():
        assert _ISO_BY_FLORES[flores] == iso


def test_flores_table_covers_the_languages_this_bot_actually_sees():
    for iso in ("en", "ru", "he"):
        assert iso in _FLORES_BY_ISO639_1


def test_detect_iso639_1_on_empty_text_returns_none():
    assert _detect_iso639_1("") is None
    assert _detect_iso639_1("   ") is None


def test_detect_iso639_1_identifies_russian():
    assert _detect_iso639_1("где покататься на велике в тель авиве") == "ru"


def test_detect_iso639_1_identifies_english():
    assert _detect_iso639_1("where can I ride a bike in Tel Aviv") == "en"
