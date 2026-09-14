# Notes Bot

A Telegram bot for saving notes (links and free-form text) with semantic
search via embeddings. Supports multiple users and shared "rooms" based on
Telegram groups.

## Status

M0 (scaffold) + M1 (embedding service) + M2 (text notes and search) + M3
(links) + M4 (transcription) + M5 (LLM enrichment): you can send the bot
text, a link (a regular page, YouTube, a short maps link), or a voice
message in a private chat, toggle privacy with a button, find a note with
`/search`. Voice notes and Instagram posts are transcribed via
`faster-whisper` on a separate heavy worker (`notes-worker-heavy`), without
blocking a plain text save. A failed extraction doesn't fail the note — the
raw text is indexed instead (degradation, see `03-ingest.md`).

After saving, a note is asynchronously enriched via ai-proxy (title, tags,
summary, structured place fields for maps, duplicate detection), and
`/smart_search` synthesizes a sourced answer the same way. `LLM_ENABLED=true`
in `deploy/k8s/configmap.yaml` — the codex sandbox fix this was blocked on
(ADR-14, risk R5 in `07-decisions.md`) has shipped (`ai-proxy#15`, merged and
deployed). With it disabled, everything still works without titles and
tags — `NullLLMClient` is substituted automatically.

M6 (groups and note management): the bot works in group "rooms" — saves
mentions and replies to itself (`/capture_all` turns on "all" mode, warning
about the bot's privacy mode in BotFather), greets on being added to a
group. `/list` and `/trash` (private chat only — private notes are never
broadcast into a group), delete with confirmation, restore. Editing a text
note works through a regular Telegram message edit.

M7 (`/smart_search`): the same results as `/search` (never worse), but
followed by a synthesized, source-linked answer via ai-proxy
(`rag_answer.md`) computed on the `llm` queue and delivered as a separate
message — the worker sends it directly via the Bot API (`TelegramSender`),
without a shared Dispatcher. No step rewrites the query (see
`04-search.md`). Degradation: with no matches, no LLM (`LLM_ENABLED=false`),
an ai-proxy error, or an empty response — nothing arrives, and the
`/search` part has already run.

M8 (garbage collection, importing old groups): `notes-gc` (daily CronJob) —
hard-deletes notes soft-deleted more than `GC_RETENTION_DAYS` ago (30 days
by default), and reclaims notes stuck in `pending`/`processing` (worker
crashed or killed) back into the queue via a deterministic `job_id` — see
`notes_bot/cli/gc.py`. `tools/import_telegram_export.py` parses `result.json`
from a Telegram Desktop export, classifies each message the same way a live
`/save` does, and is idempotent via `structured.import_id` (`tg_message_id`
on imported notes is always NULL — export ids don't match what the bot will
see live for the same chat).

M9 (operations/observability): Prometheus metrics from the `06-deployment.md`
table (`notes_bot/metrics.py`) — `fast`/`heavy`/`llm` queue length, processing
time and failure rate by `source_type`, `/search` and `/embed` latency,
extractor error rate. `notes-bot` serves `/metrics` on `HEALTH_PORT`,
`notes-worker-*` on a separate `METRICS_PORT` (its own HTTP server, since the
worker is synchronous). Logs actually became JSON — `logging_setup.py` used
to configure structlog, but no call in the codebase went through
`structlog.get_logger()`, so plain text was rendered instead; fixed via
`structlog.stdlib.ProcessorFormatter` on top of standard `logging`. An
example alert on the extractor error rate is in `06-deployment.md`.

**Not done** (needs real operation, a synthetic run doesn't substitute for
it): reviewing pod resources against a week of real data, a load test
measuring p95, reviewing the search query plan at real data volume. The
M0-M9 roadmap is now fully implemented at the code level; these three items
are, by definition, not code but production data.

The embedding model choice is still open — needs a benchmark on real notes,
see `services/embeddings/README.md`.

- [Design document](docs/design/notes-bot-design.md) — original requirements
- [Architecture](docs/architecture/README.md) — target system design
- [Implementation stages](docs/architecture/08-roadmap.md) — M0-M9 work plan

## Stack

Python, Postgres + pgvector, Redis + RQ, FastAPI + sentence-transformers for
embeddings, faster-whisper for transcription, Kubernetes.

## Development

```bash
uv sync --extra heavy   # --extra heavy pulls in yt-dlp/faster-whisper
uv run pytest -q
uv run ruff check .
uv run ruff format .
```

Migrations (needs a `DATABASE_URL` pointing at a pgvector instance, e.g.
`pgvector/pgvector:pg16` in Docker):

```bash
uv run alembic upgrade head
uv run alembic downgrade base   # roll back to an empty database, to check downgrade()
```

Most tests (ACL, repositories, search, bot business logic) are integration
tests against real Postgres+pgvector and Redis, not mocks:

```bash
docker run -d --name notes-pg -e POSTGRES_PASSWORD=test -p 5432:5432 pgvector/pgvector:pg16
docker run -d --name notes-redis -p 6379:6379 redis:7-alpine
DATABASE_URL=postgresql+psycopg://postgres:test@localhost:5432/postgres \
  REDIS_URL=redis://localhost:6379/3 \
  TELEGRAM_BOT_TOKEN=x EMBEDDINGS_URL=http://localhost:9 EMBEDDING_MODEL_NAME=x \
  uv run pytest -q
```

Required environment variables (see `notes_bot/config.py` and
`docs/architecture/05-contracts.md`): `TELEGRAM_BOT_TOKEN`, `DATABASE_URL`,
`REDIS_URL`, `EMBEDDINGS_URL`, `EMBEDDING_MODEL_NAME`. The rest are optional,
with defaults in `config.py`.

## Docker

One `Dockerfile`, two targets — see `docs/architecture/06-deployment.md`:

```bash
docker build --target app -t notes-bot-app .
docker build --target app-heavy -t notes-bot-app-heavy .
```

## Kubernetes

`deploy/k8s/` — ConfigMap, the migration Job, `notes-embeddings`, `notes-bot`,
`notes-worker-fast`, `notes-worker-heavy`, `notes-gc` (CronJob, `0 3 * * *`).

```bash
kubectl apply -f deploy/k8s/secret.yaml   # copy from secret.example.yaml, never commit it
kubectl apply -k deploy/k8s/
```

Real transcription time on the cluster node's CPU hasn't been measured —
this environment has no network access, and therefore no way to run
`faster-whisper` against real audio (see risk R2 in `07-decisions.md`).
Measure it on the first real deployment, as the roadmap expects.
