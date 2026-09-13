from notes_bot.bot.keyboards import (
    delete_confirm_keyboard,
    list_item_keyboard,
    list_more_keyboard,
    privacy_keyboard,
    search_more_keyboard,
    trash_item_keyboard,
    trash_more_keyboard,
)


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


def test_list_item_keyboard_encodes_note_id():
    kb = list_item_keyboard(42)
    assert kb.inline_keyboard[0][0].callback_data == "del:42"


def test_delete_confirm_keyboard_has_yes_and_no():
    kb = delete_confirm_keyboard(42)
    callbacks = {b.callback_data for row in kb.inline_keyboard for b in row}
    assert callbacks == {"delyes:42", "delno:42"}


def test_trash_item_keyboard_encodes_note_id():
    kb = trash_item_keyboard(42)
    assert kb.inline_keyboard[0][0].callback_data == "restore:42"


def test_list_more_keyboard_encodes_offset():
    kb = list_more_keyboard(10)
    assert kb.inline_keyboard[0][0].callback_data == "listmore:10"


def test_trash_more_keyboard_encodes_offset():
    kb = trash_more_keyboard(10)
    assert kb.inline_keyboard[0][0].callback_data == "trashmore:10"
