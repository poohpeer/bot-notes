"""Structural checks on the ORM models — the real invariants (CHECK
constraints, indexes) are exercised against a live Postgres in later stages;
this only guards against typos that would silently drift from
docs/architecture/02-data-model.md."""

from notes_bot.db.models import Base, ChatSettings, Note, NoteChunk, UserSettings


def test_all_tables_registered():
    assert set(Base.metadata.tables) == {
        "notes",
        "note_chunks",
        "user_settings",
        "chat_settings",
    }


def test_note_chunks_fk_cascades_on_delete():
    fk = next(iter(NoteChunk.__table__.c.note_id.foreign_keys))
    assert fk.column.table.name == "notes"
    assert fk.ondelete == "CASCADE"


def test_note_visibility_check_present():
    names = {c.name for c in Note.__table__.constraints if hasattr(c, "name")}
    assert "notes_visibility_ck" in names


def test_embedding_dimension_is_768():
    assert NoteChunk.__table__.c.embedding.type.dim == 768


def test_default_search_mode_and_visibility():
    assert UserSettings.__table__.c.search_mode.server_default.arg == "all"
    assert UserSettings.__table__.c.default_visibility.server_default.arg == "private"


def test_default_capture_mode_is_mentions_and_replies():
    """ADR-10: the safer default is only what's addressed to the bot."""
    assert ChatSettings.__table__.c.capture_mode.server_default.arg == "mentions_and_replies"
