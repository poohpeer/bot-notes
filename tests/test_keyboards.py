from notes_bot.bot.keyboards import privacy_keyboard, search_more_keyboard


def test_privacy_keyboard_offers_the_opposite_of_current_state():
    private_kb = privacy_keyboard(42, "private")
    button = private_kb.inline_keyboard[0][0]
    assert button.callback_data == "vis:42"
    assert "публичной" in button.text

    public_kb = privacy_keyboard(42, "public")
    assert "приватной" in public_kb.inline_keyboard[0][0].text


def test_search_more_keyboard_encodes_session_id():
    kb = search_more_keyboard("abc123")
    assert kb.inline_keyboard[0][0].callback_data == "more:abc123"
