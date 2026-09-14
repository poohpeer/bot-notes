from __future__ import annotations

import os

import pytest
import redis as sync_redis
from rq import Queue

from notes_bot.metrics import QueueLengthCollector


@pytest.fixture
def redis_conn():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
    conn = sync_redis.from_url(url)
    for name in ("fast", "heavy", "llm"):
        Queue(name, connection=conn).empty()
    yield conn
    for name in ("fast", "heavy", "llm"):
        Queue(name, connection=conn).empty()
    conn.close()


def _noop(*args):
    pass


def test_reports_zero_for_empty_queues(redis_conn):
    families = list(QueueLengthCollector(redis_conn).collect())
    assert len(families) == 1
    samples = {s.labels["queue"]: s.value for s in families[0].samples}
    assert samples == {"fast": 0, "heavy": 0, "llm": 0}


def test_reports_pending_job_count_per_queue(redis_conn):
    fast = Queue("fast", connection=redis_conn)
    heavy = Queue("heavy", connection=redis_conn)
    fast.enqueue(_noop)
    fast.enqueue(_noop)
    heavy.enqueue(_noop)

    families = list(QueueLengthCollector(redis_conn).collect())
    samples = {s.labels["queue"]: s.value for s in families[0].samples}
    assert samples == {"fast": 2, "heavy": 1, "llm": 0}
