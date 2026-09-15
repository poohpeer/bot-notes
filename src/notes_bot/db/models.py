"""SQLAlchemy models — mirrors the DDL in docs/architecture/02-data-model.md.

Kept in lockstep with that document: any change here needs a matching Alembic
migration and, if it changes an invariant, a matching update to the doc.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Telegram identity and idempotency
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    is_group: Mapped[bool] = mapped_column(Boolean, nullable=False)
    tg_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    visibility: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Content
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    lang: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Enrichment via ai-proxy
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    structured: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    # Processing state
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    enrich_status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    chunks: Mapped[list[NoteChunk]] = relationship(
        "NoteChunk", back_populates="note", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','processing','done','failed')", name="notes_status_ck"
        ),
        CheckConstraint(
            "enrich_status IN ('pending','processing','done','failed','skipped')",
            name="notes_enrich_status_ck",
        ),
        CheckConstraint(
            "source_type IN ('text','voice','page','youtube','instagram','map')",
            name="notes_source_type_ck",
        ),
        # Key privacy invariant: a group note has no visibility, a private
        # chat note must have one. Enforced by the database, not only code.
        CheckConstraint(
            "(is_group AND visibility IS NULL) OR "
            "(NOT is_group AND visibility IN ('private','public'))",
            name="notes_visibility_ck",
        ),
        Index(
            "uq_notes_tg_message",
            "chat_id",
            "tg_message_id",
            unique=True,
            postgresql_where=(text("tg_message_id IS NOT NULL")),
        ),
        Index("idx_notes_tags", "tags", postgresql_using="gin"),
        Index(
            "idx_notes_owner_live",
            "user_id",
            "created_at",
            postgresql_where=(text("deleted_at IS NULL")),
        ),
        Index(
            "idx_notes_chat_live",
            "chat_id",
            "created_at",
            postgresql_where=(text("deleted_at IS NULL")),
        ),
        Index(
            "idx_notes_trash",
            "user_id",
            "deleted_at",
            postgresql_where=(text("deleted_at IS NOT NULL")),
        ),
        Index("idx_notes_gc", "deleted_at", postgresql_where=(text("deleted_at IS NOT NULL"))),
        Index(
            "idx_notes_unfinished",
            "status",
            "updated_at",
            postgresql_where=(text("status IN ('pending','processing')")),
        ),
    )


class NoteChunk(Base):
    __tablename__ = "note_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("notes.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Dimension fixed at 768 per ADR-3 — both candidate embedding models agree.
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    note: Mapped[Note] = relationship("Note", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("note_id", "chunk_index", name="uq_note_chunks_note_chunk"),
        Index("idx_note_chunks_note", "note_id"),
        # HNSW ANN index — created here for completeness; the migration that
        # actually builds it uses CREATE INDEX CONCURRENTLY once there is data
        # (see docs/architecture/02-data-model.md, "Миграции").
        Index(
            "idx_note_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    search_mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="all")
    default_visibility: Mapped[str] = mapped_column(Text, nullable=False, server_default="private")
    # /debug — a temporary diagnostic toggle (see 03-ingest.md, "Debug:
    # время обработки"): when on, process_note sends a follow-up message
    # with how long a note's processing actually took. Off by default so
    # nobody gets it without asking.
    debug_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("search_mode IN ('all','mine_only')", name="user_settings_search_mode_ck"),
        CheckConstraint(
            "default_visibility IN ('private','public')",
            name="user_settings_default_visibility_ck",
        ),
    )


class ChatSettings(Base):
    __tablename__ = "chat_settings"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    capture_mode: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="mentions_and_replies"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "capture_mode IN ('all','mentions_and_replies')", name="chat_settings_capture_mode_ck"
        ),
    )
