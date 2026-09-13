"""Initial schema: notes, note_chunks, user_settings, chat_settings

Mirrors docs/architecture/02-data-model.md. The HNSW index is created
directly (not CONCURRENTLY) because the table is empty at this point in a
fresh deploy; CONCURRENTLY only matters once notes exist and the table can't
tolerate a long lock — see the doc's "Миграции" section for later index
changes.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-13
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("is_group", sa.Boolean(), nullable=False),
        sa.Column("tg_message_id", sa.BigInteger(), nullable=True),
        sa.Column("visibility", sa.Text(), nullable=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("lang", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "structured",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("enrich_status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "status IN ('pending','processing','done','failed')", name="notes_status_ck"
        ),
        sa.CheckConstraint(
            "enrich_status IN ('pending','processing','done','failed','skipped')",
            name="notes_enrich_status_ck",
        ),
        sa.CheckConstraint(
            "source_type IN ('text','voice','page','youtube','instagram','map')",
            name="notes_source_type_ck",
        ),
        sa.CheckConstraint(
            "(is_group AND visibility IS NULL) OR "
            "(NOT is_group AND visibility IN ('private','public'))",
            name="notes_visibility_ck",
        ),
    )
    op.create_index(
        "uq_notes_tg_message",
        "notes",
        ["chat_id", "tg_message_id"],
        unique=True,
        postgresql_where=sa.text("tg_message_id IS NOT NULL"),
    )
    op.create_index("idx_notes_tags", "notes", ["tags"], postgresql_using="gin")
    op.create_index(
        "idx_notes_owner_live",
        "notes",
        ["user_id", sa.text("created_at DESC")],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_notes_chat_live",
        "notes",
        ["chat_id", sa.text("created_at DESC")],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_notes_trash",
        "notes",
        ["user_id", sa.text("deleted_at DESC")],
        postgresql_where=sa.text("deleted_at IS NOT NULL"),
    )
    op.create_index(
        "idx_notes_gc",
        "notes",
        ["deleted_at"],
        postgresql_where=sa.text("deleted_at IS NOT NULL"),
    )
    op.create_index(
        "idx_notes_unfinished",
        "notes",
        ["status", "updated_at"],
        postgresql_where=sa.text("status IN ('pending','processing')"),
    )

    op.create_table(
        "note_chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "note_id",
            sa.BigInteger(),
            sa.ForeignKey("notes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("note_id", "chunk_index", name="uq_note_chunks_note_chunk"),
    )
    op.create_index("idx_note_chunks_note", "note_chunks", ["note_id"])
    op.execute(
        "CREATE INDEX idx_note_chunks_embedding ON note_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "user_settings",
        sa.Column("user_id", sa.BigInteger(), primary_key=True),
        sa.Column("search_mode", sa.Text(), nullable=False, server_default="all"),
        sa.Column("default_visibility", sa.Text(), nullable=False, server_default="private"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "search_mode IN ('all','mine_only')", name="user_settings_search_mode_ck"
        ),
        sa.CheckConstraint(
            "default_visibility IN ('private','public')",
            name="user_settings_default_visibility_ck",
        ),
    )

    op.create_table(
        "chat_settings",
        sa.Column("chat_id", sa.BigInteger(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("capture_mode", sa.Text(), nullable=False, server_default="mentions_and_replies"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "capture_mode IN ('all','mentions_and_replies')", name="chat_settings_capture_mode_ck"
        ),
    )


def downgrade() -> None:
    op.drop_table("chat_settings")
    op.drop_table("user_settings")
    op.drop_index("idx_note_chunks_embedding", table_name="note_chunks")
    op.drop_index("idx_note_chunks_note", table_name="note_chunks")
    op.drop_table("note_chunks")
    op.drop_index("idx_notes_unfinished", table_name="notes")
    op.drop_index("idx_notes_gc", table_name="notes")
    op.drop_index("idx_notes_trash", table_name="notes")
    op.drop_index("idx_notes_chat_live", table_name="notes")
    op.drop_index("idx_notes_owner_live", table_name="notes")
    op.drop_index("idx_notes_tags", table_name="notes")
    op.drop_index("uq_notes_tg_message", table_name="notes")
    op.drop_table("notes")
