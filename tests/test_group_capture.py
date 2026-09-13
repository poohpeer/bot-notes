from notes_bot.domain.group_capture import Entity, mentions_bot, should_capture_group_message


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
