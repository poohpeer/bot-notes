"""Repositories — the only layer that writes to notes/note_chunks/user_settings.

Encodes the invariants from docs/architecture/02-data-model.md that are not
already CHECK constraints: idempotent insert (ADR-8), atomic chunk
replacement (invariant 5), and ownership checks living in the WHERE clause
of the mutation itself rather than a separate SELECT (04-search.md, "Удаление
и восстановление" — no race between the check and the change).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import ColumnElement, case, delete, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from notes_bot.db.models import ChatSettings, Note, NoteChunk, UserSettings
from notes_bot.domain.chunking import Chunk


@dataclass(frozen=True)
class NewChunk:
    text: str
    token_count: int
    embedding: list[float]


class NoteRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_text_note(
        self,
        *,
        user_id: int,
        chat_id: int,
        is_group: bool,
        tg_message_id: int | None,
        raw_text: str,
        visibility: str | None,
    ) -> Note | None:
        """Insert a pending text note. Returns None on a duplicate delivery
        (same chat_id + tg_message_id) — ADR-8: idempotency is a database
        constraint (`uq_notes_tg_message`), not a pre-check, so a retried
        Telegram update or a bot restart mid-processing never double-saves.
        """
        stmt = (
            pg_insert(Note)
            .values(
                user_id=user_id,
                chat_id=chat_id,
                is_group=is_group,
                tg_message_id=tg_message_id,
                visibility=visibility,
                source_type="text",
                raw_text=raw_text,
                status="pending",
            )
            .on_conflict_do_nothing(
                # `uq_notes_tg_message` is a partial unique index (WHERE
                # tg_message_id IS NOT NULL) — Postgres only matches ON
                # CONFLICT against a partial index when the same predicate
                # is repeated here.
                index_elements=["chat_id", "tg_message_id"],
                index_where=text("tg_message_id IS NOT NULL"),
            )
            .returning(Note)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return row

    async def get(self, note_id: int) -> Note | None:
        return await self._session.get(Note, note_id)

    async def get_visible_by_ids(
        self, note_ids: list[int], *, acl_predicate: ColumnElement[bool]
    ) -> dict[int, Note]:
        """Re-checks `deleted_at`/ACL for a cached page of search results —
        see 04-search.md, "Пагинация": a note deleted between pages must not
        surface just because its id was cached. Returns a mapping so the
        caller re-orders by its own ranking rather than trusting SQL order.
        """
        if not note_ids:
            return {}
        result = await self._session.execute(
            select(Note).where(Note.id.in_(note_ids), Note.deleted_at.is_(None), acl_predicate)
        )
        return {n.id: n for n in result.scalars()}

    async def mark_processing(self, note_id: int) -> None:
        await self._session.execute(
            update(Note).where(Note.id == note_id).values(status="processing")
        )

    async def mark_done(self, note_id: int) -> None:
        await self._session.execute(update(Note).where(Note.id == note_id).values(status="done"))

    async def mark_failed(self, note_id: int, error: str) -> None:
        await self._session.execute(
            update(Note)
            .where(Note.id == note_id)
            .values(status="failed", error=error, attempts=Note.attempts + 1)
        )

    async def toggle_visibility(self, note_id: int, user_id: int) -> str | None:
        """Flips private<->public. Ownership is enforced in the WHERE clause
        of the UPDATE itself — zero rows affected means "not yours or
        doesn't exist", not a separate permission error to race against.

        Returns the new visibility, or None if nothing was updated (wrong
        owner, missing note, or a group note — group notes have no
        visibility to toggle, per the CHECK constraint).
        """
        result = await self._session.execute(
            update(Note)
            .where(Note.id == note_id, Note.user_id == user_id, Note.visibility.is_not(None))
            .values(visibility=case((Note.visibility == "private", "public"), else_="private"))
            .returning(Note.visibility)
        )
        return result.scalar_one_or_none()


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_chunks(
        self, note_id: int, chunks: list[NewChunk], *, embedding_model: str
    ) -> None:
        """DELETE + INSERT in one transaction — invariant 5 in
        02-data-model.md: there is no intermediate state where some chunks
        are old and some are new. The caller commits (or rolls back) the
        session as one unit; this method does not commit on its own.
        """
        await self._session.execute(delete(NoteChunk).where(NoteChunk.note_id == note_id))
        if not chunks:
            return
        await self._session.execute(
            insert(NoteChunk),
            [
                {
                    "note_id": note_id,
                    "chunk_index": i,
                    "chunk_text": c.text,
                    "token_count": c.token_count,
                    "embedding": c.embedding,
                    "embedding_model": embedding_model,
                }
                for i, c in enumerate(chunks)
            ],
        )

    @staticmethod
    def from_domain_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> list[NewChunk]:
        return [
            NewChunk(text=c.text, token_count=c.token_count, embedding=e)
            for c, e in zip(chunks, embeddings, strict=True)
        ]


class UserSettingsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, user_id: int) -> UserSettings:
        existing = await self._session.get(UserSettings, user_id)
        if existing is not None:
            return existing

        stmt = (
            pg_insert(UserSettings)
            .values(user_id=user_id)
            .on_conflict_do_nothing(index_elements=["user_id"])
            .returning(UserSettings)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            return row
        # Lost the race to a concurrent insert (two updates from the same
        # user arriving together) — the other one committed first.
        return await self._session.get(UserSettings, user_id)

    async def set_search_mode(self, user_id: int, mode: str) -> None:
        await self.get_or_create(user_id)
        await self._session.execute(
            update(UserSettings).where(UserSettings.user_id == user_id).values(search_mode=mode)
        )


class ChatSettingsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_capture_mode(self, chat_id: int) -> str:
        result = await self._session.execute(
            select(ChatSettings.capture_mode).where(ChatSettings.chat_id == chat_id)
        )
        row = result.scalar_one_or_none()
        return row or "mentions_and_replies"
