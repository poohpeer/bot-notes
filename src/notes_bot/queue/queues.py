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


def enqueue_process_note(fast_queue: Queue, note_id: int) -> None:
    """Deterministic job_id (ADR-8): a duplicate enqueue for the same note —
    a retried Telegram update, or a bot restart mid-processing — collapses
    into the same job instead of stacking a second run."""
    fast_queue.enqueue(process_note, note_id, job_id=f"process_note:{note_id}")


def enqueue_smart_answer(
    llm_queue: Queue, *, user_id: int, chat_id: int, is_group_chat: bool, query_text: str
) -> None:
    """No deterministic job_id here, unlike enqueue_process_note — this
    isn't persisted data with a natural dedup key (ADR-8's idempotency is
    about a note existing once; a `/smart_search` call is a one-off action,
    and two of them in a row from an impatient user should both answer)."""
    llm_queue.enqueue(smart_answer, user_id, chat_id, is_group_chat, query_text)
