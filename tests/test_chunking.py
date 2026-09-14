import pytest

from notes_bot.domain.chunking import chunk


def test_empty_text_yields_no_chunks():
    assert chunk("") == []
    assert chunk("   ") == []


def test_short_text_yields_one_chunk():
    text = "one two three"
    chunks = chunk(text, target=10, overlap=2, min_size=1)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == text
    assert chunks[0].token_count == 3


def test_long_text_splits_into_multiple_chunks():
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)
    chunks = chunk(text, target=10, overlap=2, min_size=1)
    assert len(chunks) > 1
    # Every word appears somewhere in the output.
    covered = " ".join(c.text for c in chunks).split()
    assert set(words) <= set(covered)


def test_consecutive_chunks_overlap_by_the_configured_amount():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk(text, target=10, overlap=3, min_size=1)
    assert len(chunks) >= 2
    first_tail = chunks[0].text.split()[-3:]
    second_head = chunks[1].text.split()[:3]
    assert first_tail == second_head


def test_trailing_remainder_smaller_than_min_size_is_merged_into_previous():
    # target=10, overlap=2 -> step=8. 18 words: window0=[0:10], window1
    # starts at 8, giving [8:18] = 10 words — no remainder here. Use 13
    # words so the second window is short: window1=[8:13] = 5 words.
    words = [f"w{i}" for i in range(13)]
    text = " ".join(words)
    chunks = chunk(text, target=10, overlap=2, min_size=6)
    assert len(chunks) == 1
    assert chunks[0].text.split() == words


def test_trailing_remainder_at_or_above_min_size_stays_its_own_chunk():
    words = [f"w{i}" for i in range(13)]
    text = " ".join(words)
    chunks = chunk(text, target=10, overlap=2, min_size=5)
    assert len(chunks) == 2


def test_max_chunks_caps_output():
    words = [f"w{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk(text, target=10, overlap=0, min_size=1, max_chunks=5)
    assert len(chunks) == 5


def test_overlap_must_be_smaller_than_target():
    with pytest.raises(ValueError):
        chunk("a b c", target=5, overlap=5)


def test_chunk_indices_are_sequential():
    words = [f"w{i}" for i in range(30)]
    text = " ".join(words)
    chunks = chunk(text, target=10, overlap=2, min_size=1)
    assert [c.index for c in chunks] == list(range(len(chunks)))
