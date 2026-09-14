"""Queue definitions — three RQ queues, see docs/architecture/06-deployment.md
and ADR-6: `fast` (text/pages, seconds), `heavy` (media downloads + Whisper,
minutes), `llm` (ai-proxy enrichment, not urgent).
"""

from __future__ import annotations

from redis import Redis
from rq import Queue

from notes_bot.queue.tasks import process_note, smart_answer


def get_queues(redis_conn: Redis) -> dict[str, Queue]:
    return {name: Queue(name, connection=redis_conn) for name in ("fast", "heavy", "llm")}


def enqueue_process_note(
    fast_queue: Queue, note_id: int, *, job_timeout: int | None = None
) -> None:
    """Deterministic job_id (ADR-8). A dash, not a colon (docs/architecture/
    03-ingest.md's snippet used `note:{id}`): RQ 2.x's `validate_job_id`
    rejects any character outside `[A-Za-z0-9_-]`.

    RQ does not dedupe the queue's own pending list by job_id — a duplicate
    enqueue (a retried Telegram update, a bot restart mid-processing, or
    notes-gc reclaiming an already-reclaimed note) can still push a second
    list entry sharing the same job_id. The safety this buys is elsewhere:
    both entries resolve to the same `Job` (its stored args/status are
    shared, last enqueue wins), and process_note itself no-ops a second run
    once the note is no longer 'pending'/'processing' (see queue/tasks.py,
    "already settled, skipping") — so a duplicate is wasted worker time, not
    a duplicate write.

    `job_timeout` is None for the fast queue (RQ's own default, 180s, is
    plenty for text/pages) and `settings.heavy_job_timeout_s` for the heavy
    one — see Settings.heavy_job_timeout_s for why 180s is too tight there."""
    fast_queue.enqueue(
        process_note, note_id, job_id=f"process_note-{note_id}", job_timeout=job_timeout
    )


def enqueue_smart_answer(
    llm_queue: Queue, *, user_id: int, chat_id: int, is_group_chat: bool, query_text: str
) -> None:
    """No deterministic job_id here, unlike enqueue_process_note — this
    isn't persisted data with a natural dedup key (ADR-8's idempotency is
    about a note existing once; a `/smart_search` call is a one-off action,
    and two of them in a row from an impatient user should both answer)."""
    llm_queue.enqueue(smart_answer, user_id, chat_id, is_group_chat, query_text)
