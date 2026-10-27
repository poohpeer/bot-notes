from notes_bot.domain.group_capture import (
    Entity,
    mentions_bot,
    should_capture_group_message,
    should_save_photo_note,
)


def test_mentions_bot_true_for_matching_mention_entity():
    text = "hey @notesbot save this"
    entities = [Entity(type="mention", offset=4, length=9)]
    assert mentions_bot(text, entities, "notesbot") is True


def test_mentions_bot_false_for_a_different_mention():
    text = "hey @someoneelse look"
    entities = [Entity(type="mention", offset=4, length=13)]
    assert mentions_bot(text, entities, "notesbot") is False


def test_mentions_bot_is_case_insensitive():
    text = "@NotesBot save this"
    entities = [Entity(type="mention", offset=0, length=9)]
    assert mentions_bot(text, entities, "notesbot") is True


def test_mentions_bot_false_with_no_entities():
    assert mentions_bot("just text, no mentions", [], "notesbot") is False


def test_mentions_bot_ignores_non_mention_entities():
    text = "check https://example.com"
    entities = [Entity(type="url", offset=6, length=19)]
    assert mentions_bot(text, entities, "notesbot") is False


def test_capture_all_mode_captures_everything():
    assert should_capture_group_message(
        capture_mode="all", mentions_bot=False, is_reply_to_bot=False
    )


def test_default_mode_requires_mention_or_reply():
    assert not should_capture_group_message(
        capture_mode="mentions_and_replies", mentions_bot=False, is_reply_to_bot=False
    )


def test_default_mode_captures_a_mention():
    assert should_capture_group_message(
        capture_mode="mentions_and_replies", mentions_bot=True, is_reply_to_bot=False
    )


def test_default_mode_captures_a_reply_to_the_bot():
    assert should_capture_group_message(
        capture_mode="mentions_and_replies", mentions_bot=False, is_reply_to_bot=True
    )


def test_mentions_bot_survives_an_emoji_earlier_in_the_text():
    """Telegram counts entity offsets in UTF-16 code units: the emoji is two
    of them, one Python character."""
    text = "🎉 @notesbot запомни"
    entities = [Entity(type="mention", offset=3, length=9)]
    assert mentions_bot(text, entities, "notesbot") is True


def test_mentions_bot_survives_several_emoji_and_still_rejects_other_names():
    text = "🎉🎂🥳 @someoneelse @notesbot"
    entities = [
        Entity(type="mention", offset=7, length=12),
        Entity(type="mention", offset=20, length=9),
    ]
    assert mentions_bot(text, entities, "notesbot") is True
    assert mentions_bot(text, entities[:1], "notesbot") is False


def test_a_private_photo_is_always_a_note():
    assert should_save_photo_note(is_group=False, mentions_bot=False)


def test_a_group_photo_needs_the_mention():
    assert not should_save_photo_note(is_group=True, mentions_bot=False)
    assert should_save_photo_note(is_group=True, mentions_bot=True)
