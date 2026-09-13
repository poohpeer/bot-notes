# Notes Bot

A Telegram bot for saving notes (links and free-form text) with semantic
search via embeddings. Supports multiple users and shared "rooms" based on
Telegram groups.

## Status

M0 (scaffold) + M1 (embedding service): configuration, data model,
migrations, CI, image builds, `notes-embeddings` (`POST /embed`, `GET
/model`). The embedding model choice is still open — needs a benchmark on
real notes, see `services/embeddings/README.md`. The bot and workers don't
do anything yet — see `docs/architecture/08-roadmap.md`.

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

`deploy/k8s/` — ConfigMap and the migration Job (M0). Deployment manifests
for the bot and workers land with M2, once those processes actually do
something.

```bash
kubectl apply -f deploy/k8s/secret.yaml   # copy from secret.example.yaml, never commit it
kubectl apply -k deploy/k8s/
```
