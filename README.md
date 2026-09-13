# Notes Bot

Telegram-бот для сохранения заметок (ссылки и свободный текст) с
семантическим поиском через embeddings. Поддержка нескольких пользователей и
общих «комнат» на базе Telegram-групп.

## Состояние

M0 (каркас) + M1 (embedding-сервис) + M2 (текстовые заметки и поиск -
минимальная полезная система): можно отправить боту текст в личном чате,
переключить приватность кнопкой, найти заметку `/search`, переключить режим
`/search_mine` / `/search_all`. Ссылки, транскрипция, LLM-обогащение и
группы - следующие этапы, см. `docs/architecture/08-roadmap.md`.

Выбор embedding-модели ещё не закрыт - нужен бенчмарк на реальных заметках,
см. `services/embeddings/README.md`.

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

Большая часть тестов (ACL, репозитории, поиск, бизнес-логика бота) - это
интеграционные тесты против настоящих Postgres+pgvector и Redis, не моки:

```bash
docker run -d --name notes-pg -e POSTGRES_PASSWORD=test -p 5432:5432 pgvector/pgvector:pg16
docker run -d --name notes-redis -p 6379:6379 redis:7-alpine
DATABASE_URL=postgresql+psycopg://postgres:test@localhost:5432/postgres \
  REDIS_URL=redis://localhost:6379/3 \
  TELEGRAM_BOT_TOKEN=x EMBEDDINGS_URL=http://localhost:9 EMBEDDING_MODEL_NAME=x \
  uv run pytest -q
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
