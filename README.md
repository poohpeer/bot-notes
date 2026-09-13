# Notes Bot

Telegram-бот для сохранения заметок (ссылки и свободный текст) с
семантическим поиском через embeddings. Поддержка нескольких пользователей и
общих «комнат» на базе Telegram-групп.

## Состояние

M0 (каркас): конфигурация, модель данных, миграции, CI, сборка образов.
Бот и воркеры пока ничего не делают - см. `docs/architecture/08-roadmap.md`.

- [Дизайн-документ](docs/design/notes-bot-design.md) - исходные требования
- [Архитектура](docs/architecture/README.md) - целевое устройство системы
- [Этапы имплементации](docs/architecture/08-roadmap.md) - план работ M0-M9

## Стек

Python, Postgres + pgvector, Redis + RQ, FastAPI + sentence-transformers для
эмбеддингов, faster-whisper для транскрипции, Kubernetes.

## Разработка

```bash
uv sync --extra heavy   # --extra heavy подтягивает yt-dlp/faster-whisper
uv run pytest -q
uv run ruff check .
uv run ruff format .
```

Миграции (нужен `DATABASE_URL` с pgvector-инстансом, например
`pgvector/pgvector:pg16` в Docker):

```bash
uv run alembic upgrade head
uv run alembic downgrade base   # откат до пустой базы, для проверки downgrade()
```

Обязательные переменные окружения (см. `notes_bot/config.py` и
`docs/architecture/05-contracts.md`): `TELEGRAM_BOT_TOKEN`, `DATABASE_URL`,
`REDIS_URL`, `EMBEDDINGS_URL`, `EMBEDDING_MODEL_NAME`. Остальные - опциональны,
дефолты в `config.py`.

## Docker

Один `Dockerfile`, два таргета - см. `docs/architecture/06-deployment.md`:

```bash
docker build --target app -t notes-bot-app .
docker build --target app-heavy -t notes-bot-app-heavy .
```

## Kubernetes

`deploy/k8s/` - ConfigMap и Job миграций (M0). Deployment-манифесты бота и
воркеров появятся вместе с M2, когда эти процессы начнут что-то делать.

```bash
kubectl apply -f deploy/k8s/secret.yaml   # скопировать из secret.example.yaml, не коммитить
kubectl apply -k deploy/k8s/
```
