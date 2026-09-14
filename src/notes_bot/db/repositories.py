"""Repositories — the only layer that writes to notes/note_chunks/user_settings.

Encodes the invariants from docs/architecture/02-data-model.md that are not
already CHECK constraints: idempotent insert (ADR-8), atomic chunk
replacement (invariant 5), and ownership checks living in the WHERE clause
of the mutation itself rather than a separate SELECT (04-search.md, "Удаление
и восстановление" — no race between the check and the change).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import ColumnElement, case, delete, func, insert, select, text, update
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

    async def create_note(
        self,
        *,
        user_id: int,
        chat_id: int,
        is_group: bool,
        tg_message_id: int | None,
        source_type: str,
        source_url: str | None = None,
        raw_text: str | None,
        visibility: str | None,
    ) -> Note | None:
        """Insert a pending note of any source_type. Returns None on a
        duplicate delivery (same chat_id + tg_message_id) — ADR-8:
        idempotency is a database constraint (`uq_notes_tg_message`), not a
        pre-check, so a retried Telegram update or a bot restart
        mid-processing never double-saves.
        """
        stmt = (
            pg_insert(Note)
            .values(
                user_id=user_id,
                chat_id=chat_id,
                is_group=is_group,
                tg_message_id=tg_message_id,
                visibility=visibility,
                source_type=source_type,
                source_url=source_url,
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
        """Thin wrapper kept for the many callers (and tests) that only ever
        made plain text notes before create_note grew source_type/source_url
        for M3's links."""
        return await self.create_note(
            user_id=user_id,
            chat_id=chat_id,
            is_group=is_group,
            tg_message_id=tg_message_id,
            source_type="text",
            raw_text=raw_text,
            visibility=visibility,
        )

    async def get_by_import_id(self, *, chat_id: int, import_id: str) -> Note | None:
        """tools/import_telegram_export.py's idempotency key — see
        03-ingest.md, "Импорт экспорта Telegram". `tg_message_id` is
        deliberately NULL for every imported note (export ids don't match
        what the live bot will later see for the same chat), so
        `uq_notes_tg_message` can't dedupe imports; `structured.import_id`
        does instead, checked here before insert rather than as a DB
        constraint, per the design doc."""
        result = await self._session.execute(
            select(Note).where(
                Note.chat_id == chat_id, Note.structured["import_id"].astext == import_id
            )
        )
        return result.scalar_one_or_none()

    async def create_imported_note(
        self,
        *,
        user_id: int,
        chat_id: int,
        source_type: str,
        source_url: str | None,
        raw_text: str,
        created_at: datetime,
        import_id: str,
    ) -> Note:
        """Always a group note (`is_group=True`, `visibility=NULL`) and
        always `tg_message_id=NULL` — see 03-ingest.md, "Импорт экспорта
        Telegram". `created_at` is the export's own date, not import time,
        so imported notes sort correctly alongside live ones."""
        note = Note(
            user_id=user_id,
            chat_id=chat_id,
            is_group=True,
            tg_message_id=None,
            visibility=None,
            source_type=source_type,
            source_url=source_url,
            raw_text=raw_text,
            status="pending",
            structured={"import_id": import_id},
            created_at=created_at,
            updated_at=created_at,
        )
        self._session.add(note)
        await self._session.flush()
        return note

    async def record_extraction(
        self, note_id: int, *, extracted_text: str | None, lang: str | None, error: str | None
    ) -> None:
        """Persists what an extractor found, before chunking runs. `error`
        is set even on a degraded-but-not-failed note (03-ingest.md,
        "Деградация") — it explains *why* extracted_text is empty without
        making the note unfindable, which only `status='failed'` does."""
        await self._session.execute(
            update(Note)
            .where(Note.id == note_id)
            .values(extracted_text=extracted_text, lang=lang, error=error)
        )

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

    async def mark_enrich_processing(self, note_id: int) -> None:
        await self._session.execute(
            update(Note).where(Note.id == note_id).values(enrich_status="processing")
        )

    async def set_enrichment(
        self,
        note_id: int,
        *,
        title: str | None,
        summary: str | None,
        tags: list[str],
        structured: dict,
    ) -> None:
        """Never touches `status`/`extracted_text`/chunks — enrichment is a
        layer on top of an already-findable note (03-ingest.md, "Шаг 6"),
        never a gate in front of it."""
        await self._session.execute(
            update(Note)
            .where(Note.id == note_id)
            .values(
                title=title,
                summary=summary,
                tags=tags,
                structured=structured,
                enrich_status="done",
            )
        )

    async def mark_enrich_failed(self, note_id: int, error: str) -> None:
        await self._session.execute(
            update(Note).where(Note.id == note_id).values(enrich_status="failed", error=error)
        )

    async def mark_enrich_skipped(self, note_id: int) -> None:
        """LLM_ENABLED=false — see 05-contracts.md's NullLLMClient. Distinct
        from 'failed': nothing went wrong, enrichment just isn't turned on."""
        await self._session.execute(
            update(Note).where(Note.id == note_id).values(enrich_status="skipped")
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

    async def list_own(self, user_id: int, *, limit: int, offset: int) -> list[Note]:
        """`/list` — see 04-search.md: "Свои заметки по дате, без
        векторов". Plain SQL pagination, not the search cache: there is no
        ANN re-ranking to go stale between pages here."""
        result = await self._session.execute(
            select(Note)
            .where(Note.user_id == user_id, Note.deleted_at.is_(None))
            .order_by(Note.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars())

    async def list_own_deleted(self, user_id: int, *, limit: int, offset: int) -> list[Note]:
        result = await self._session.execute(
            select(Note)
            .where(Note.user_id == user_id, Note.deleted_at.is_not(None))
            .order_by(Note.deleted_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars())

    async def soft_delete(self, note_id: int, user_id: int) -> bool:
        """`user_id = :me` in the WHERE clause of the mutation itself, per
        04-search.md, "Удаление и восстановление" — check and change are
        one operation, no race between them. Returns whether a row was
        actually touched (False: wrong owner, already deleted, or gone)."""
        result = await self._session.execute(
            update(Note)
            .where(Note.id == note_id, Note.user_id == user_id, Note.deleted_at.is_(None))
            .values(deleted_at=func.now())
        )
        return result.rowcount > 0

    async def restore(self, note_id: int, user_id: int) -> bool:
        result = await self._session.execute(
            update(Note)
            .where(Note.id == note_id, Note.user_id == user_id, Note.deleted_at.is_not(None))
            .values(deleted_at=None)
        )
        return result.rowcount > 0

    async def hard_delete_expired(self, *, before: datetime, limit: int) -> int:
        """notes-gc, see 08-roadmap.md M8 and 02-data-model.md's `idx_notes_gc`.
        Batched via a subquery + LIMIT so one GC pass never holds a single
        multi-million-row transaction open (07-decisions.md R7: no long
        transactions on the shared Postgres). `note_chunks` cascades via
        `ON DELETE CASCADE`."""
        subq = (
            select(Note.id)
            .where(Note.deleted_at.is_not(None), Note.deleted_at < before)
            .limit(limit)
        )
        result = await self._session.execute(delete(Note).where(Note.id.in_(subq)))
        return result.rowcount

    async def reclaim_stuck(self, *, before: datetime, limit: int) -> list[Note]:
        """notes-gc's other job: a worker that crashed or was killed mid-job
        leaves a note in `status='processing'` forever — RQ's own job never
        retries because the job itself is gone, not failed. `idx_notes_unfinished`
        exists for exactly this scan. Returns the reclaimed rows (id +
        source_type) so the caller knows which queue to re-enqueue each one
        into."""
        subq = (
            select(Note.id)
            .where(Note.status.in_(("pending", "processing")), Note.updated_at < before)
            .limit(limit)
        )
        result = await self._session.execute(
            update(Note)
            .where(Note.id.in_(subq))
            .values(status="pending", attempts=Note.attempts + 1)
            .returning(Note)
        )
        return list(result.scalars())

    async def edit_text(self, note_id: int, user_id: int, new_text: str) -> bool:
        """Only for `source_type='text'` — a voice transcript or a page's
        extracted_text isn't user-authored text to begin with (03-ingest.md,
        "Редактирование"). Resets to `status='pending'`: the caller must
        re-enqueue process_note, since the old chunks/vectors no longer
        match `raw_text`."""
        result = await self._session.execute(
            update(Note)
            .where(
                Note.id == note_id,
                Note.user_id == user_id,
                Note.source_type == "text",
            )
            .values(raw_text=new_text, status="pending")
        )
        return result.rowcount > 0

    async def edit_text_by_message(
        self, *, chat_id: int, tg_message_id: int, user_id: int, new_text: str
    ) -> int | None:
        """Same as `edit_text`, but for the edited_message path: the user
        edited their original Telegram message rather than issuing an
        explicit edit command, so all the bot has to go on is
        (chat_id, tg_message_id) — the same pair ADR-8's idempotent insert
        keys on. Returns the note_id on success, so the caller can
        re-enqueue process_note without a second lookup."""
        result = await self._session.execute(
            update(Note)
            .where(
                Note.chat_id == chat_id,
                Note.tg_message_id == tg_message_id,
                Note.user_id == user_id,
                Note.source_type == "text",
            )
            .values(raw_text=new_text, status="pending")
            .returning(Note.id)
        )
        return result.scalar_one_or_none()


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_first_chunk_embedding(self, note_id: int) -> list[float] | None:
        """Stands in for "this note's vector" in duplicate detection — one
        chunk's embedding, not an average across all of them. Good enough
        to find near-duplicate notes; a proper per-note centroid isn't
        worth the complexity for a candidate search that an LLM call
        confirms afterward anyway (03-ingest.md, "Шаг 6")."""
        result = await self._session.execute(
            select(NoteChunk.embedding)
            .where(NoteChunk.note_id == note_id)
            .order_by(NoteChunk.chunk_index)
            .limit(1)
        )
        row = result.first()
        return list(row[0]) if row else None

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

    async def set_capture_mode(self, chat_id: int, mode: str, *, title: str | None = None) -> None:
        stmt = (
            pg_insert(ChatSettings)
            .values(chat_id=chat_id, capture_mode=mode, title=title)
            .on_conflict_do_update(
                index_elements=["chat_id"], set_={"capture_mode": mode, "updated_at": func.now()}
            )
        )
        await self._session.execute(stmt)
