# Notes Bot

Telegram-бот для сохранения заметок (ссылки и свободный текст) с
семантическим поиском через embeddings. Поддержка нескольких пользователей и
общих «комнат» на базе Telegram-групп.

## Состояние

M0 (каркас) + M1 (embedding-сервис) + M2 (текстовые заметки и поиск) + M3
(ссылки) + M4 (транскрипция) + M5 (LLM-обогащение): можно отправить боту
текст, ссылку (обычную, YouTube, короткую ссылку карт) или голосовое
сообщение в личном чате, переключить приватность кнопкой, найти заметку
`/search`. Голосовые заметки и посты Instagram транскрибируются через
`faster-whisper` на отдельном тяжёлом воркере (`notes-worker-heavy`), не
блокируя сохранение обычного текста. Неудачное извлечение не роняет
заметку, индексируется исходный текст (деградация, см. `03-ingest.md`).

После сохранения заметка асинхронно обогащается через ai-proxy (заголовок,
теги, суммаризация, для карт - структурные поля места, детекция дублей) -
но `LLM_ENABLED=false` по умолчанию и **должен оставаться `false` в проде**
до фикса песочницы codex в `poohpeer/ai-proxy` (ADR-14, риск R5 в
`07-decisions.md`; фикс - `ai-proxy#15`, не смержен). Без него всё работает
без заголовков и тегов - `NullLLMClient` подставляется автоматически.

M6 (группы и управление заметками): бот работает в группах-«комнатах» -
сохраняет упоминания и ответы себе (`/capture_all` включает режим "всё",
предупреждая про privacy mode у BotFather), здоровается при добавлении в
группу. `/list` и `/trash` (только в личном чате - не транслируют личные
заметки в группу), удаление с подтверждением, восстановление. Редактирование
текстовой заметки - через обычное редактирование сообщения в Telegram.

M7 (`/smart_search`): те же результаты, что и у `/search` (никогда не хуже),
но следом в очереди `llm` считается синтезированный, со ссылками на заметки
ответ через ai-proxy (`rag_answer.md`) и приходит отдельным сообщением -
воркер шлёт его напрямую через Bot API (`TelegramSender`), без общего
Dispatcher. Ни один этап не переписывает запрос (см. `04-search.md`).
Деградация: без совпадений, без LLM (`LLM_ENABLED=false`), при ошибке
ai-proxy или пустом ответе - просто ничего не приходит, `/search`-часть уже
отработала.

M8 (сборка мусора, импорт старых групп): `notes-gc` (ежедневный CronJob) -
hard-delete заметок, удалённых больше `GC_RETENTION_DAYS` (30 дней по
умолчанию) назад, и возврат зависших в `pending`/`processing` заметок
(воркер упал/убит) обратно в очередь по детерминированному `job_id` - см.
`notes_bot/cli/gc.py`. `tools/import_telegram_export.py` разбирает
`result.json` из экспорта Telegram Desktop, классифицирует каждое
сообщение так же, как живой `/save`, и идемпотентен по `structured.import_id`
(`tg_message_id` у импортированных заметок всегда NULL - id из экспорта не
совпадают с тем, что бот увидит вживую для того же чата).

M9 (эксплуатация/observability): Prometheus-метрики из таблицы
`06-deployment.md` (`notes_bot/metrics.py`) - длина очередей `fast`/`heavy`/`llm`,
время обработки и доля failed по `source_type`, латентность `/search` и
`/embed`, доля ошибок экстракторов. `notes-bot` отдаёт `/metrics` на
`HEALTH_PORT`, `notes-worker-*` - на отдельном `METRICS_PORT` (свой
HTTP-сервер, т.к. воркер синхронный). Логи по-настоящему стали JSON -
`logging_setup.py` раньше настраивал structlog, но ни один вызов в кодовой
базе не шёл через `structlog.get_logger()`, так что рендерился обычный
текст; исправлено через `structlog.stdlib.ProcessorFormatter` поверх
стандартного `logging`. Пример алерта на долю ошибок экстракторов - в
`06-deployment.md`.

**Не сделано** (требует реальной эксплуатации, синтетический прогон не
заменяет): ревизия ресурсов подов по фактическим данным за неделю, прогон
нагрузки с измерением p95, ревизия плана запроса поиска на реальном объёме
данных. Roadmap M0-M9 на этом реализован полностью на уровне кода; эти три
пункта - по определению не код, а данные с прода.

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

`deploy/k8s/` - ConfigMap, Job миграций, `notes-embeddings`, `notes-bot`,
`notes-worker-fast`, `notes-worker-heavy`, `notes-gc` (CronJob, `0 3 * * *`).

```bash
kubectl apply -f deploy/k8s/secret.yaml   # скопировать из secret.example.yaml, не коммитить
kubectl apply -k deploy/k8s/
```

Реальное время транскрибации на CPU узла кластера не измерялось - в этом
окружении нет сети и, соответственно, нет способа прогнать `faster-whisper`
на реальном аудио (см. риск R2 в `07-decisions.md`). Замерить на первом же
проде, как и предполагает roadmap.
