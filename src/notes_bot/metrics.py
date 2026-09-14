"""Prometheus metrics — see docs/architecture/06-deployment.md, "Наблюдаемость".

Every metric here maps to a row of that section's table. Module-level
Counters/Histograms auto-register to `prometheus_client`'s default global
`REGISTRY`; `register_queue_length_collector` adds the one metric that needs
a live Redis connection to compute (queue length isn't something a process
increments itself, it's read from RQ on every scrape).
"""

from __future__ import annotations

from collections.abc import Iterable

from prometheus_client import REGISTRY, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from redis import Redis
from rq import Queue

# "Время обработки по source_type" — отдельно покажет реальную стоимость Whisper.
PROCESS_NOTE_DURATION_SECONDS = Histogram(
    "notes_process_note_duration_seconds",
    "process_note wall time, by source_type.",
    ["source_type"],
)

# "Доля status='failed' и enrich_status='failed'" — здоровье экстракторов и ai-proxy.
NOTES_STATUS_FAILED_TOTAL = Counter(
    "notes_status_failed_total",
    "Notes whose status ended 'failed', by source_type.",
    ["source_type"],
)
NOTES_ENRICH_FAILED_TOTAL = Counter(
    "notes_enrich_status_failed_total",
    "Notes whose enrich_status ended 'failed'.",
)

# "Латентность /search p50/p95".
SEARCH_DURATION_SECONDS = Histogram(
    "notes_search_duration_seconds",
    "search_notes() wall time.",
)

# "Латентность /embed" — узкое место общее для приёма и поиска.
EMBED_DURATION_SECONDS = Histogram(
    "notes_embed_duration_seconds",
    "Embedding service HTTP request duration, by kind.",
    ["kind"],
)

# "Доля ошибок yt-dlp по источникам" — "заслуживает алерта": блокировки
# внешних площадок — самый вероятный вид поломки в этой системе.
EXTRACTOR_FAILURES_TOTAL = Counter(
    "notes_extractor_failures_total",
    "Extraction failures by source_type.",
    ["source_type"],
)


class QueueLengthCollector(Collector):
    """ "Длина очередей fast, heavy, llm" — "раньше всего показывает, что
    воркер не справляется". A `Collector`, not a `Gauge` set by hand:
    queue length is state RQ/Redis already holds, not something any one
    process increments — reading it fresh on every scrape is both simpler
    and can't drift out of sync the way a manually-updated gauge could."""

    def __init__(self, redis_conn: Redis) -> None:
        self._redis_conn = redis_conn

    def collect(self) -> Iterable:
        family = GaugeMetricFamily(
            "notes_queue_length", "Pending job count per RQ queue.", labels=["queue"]
        )
        for name in ("fast", "heavy", "llm"):
            family.add_metric([name], Queue(name, connection=self._redis_conn).count)
        yield family


def register_queue_length_collector(redis_conn: Redis) -> None:
    """Call once per process, at startup (cli/bot.py, cli/worker.py) —
    registering the same collector into the global registry twice raises.
    Tests exercise `QueueLengthCollector.collect()` directly instead of
    going through the global registry."""
    REGISTRY.register(QueueLengthCollector(redis_conn))
