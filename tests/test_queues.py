from __future__ import annotations

from notes_bot.queue.queues import enqueue_process_note


class FakeQueue:
    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def enqueue(self, func, *args, **kw):
        self.kwargs.append(kw)


def test_job_timeout_is_passed_through():
    queue = FakeQueue()
    enqueue_process_note(queue, 1, job_timeout=900)
    assert queue.kwargs[0]["job_timeout"] == 900


def test_job_timeout_defaults_to_none():
    """None, not omitted — RQ reads None the same as "use my own default"
    (180s), which is what the fast queue wants."""
    queue = FakeQueue()
    enqueue_process_note(queue, 1)
    assert queue.kwargs[0]["job_timeout"] is None


def test_the_job_id_stays_deterministic_regardless_of_timeout():
    queue = FakeQueue()
    enqueue_process_note(queue, 42, job_timeout=900)
    assert queue.kwargs[0]["job_id"] == "process_note-42"
