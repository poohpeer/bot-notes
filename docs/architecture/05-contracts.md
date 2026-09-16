# 05. Контракты

## Embedding-сервис

Единственный владелец модели. Всё, что специфично для конкретной
модели - префиксы, нормализация, токенизация, батчинг, - живёт здесь.
Приложение оперирует понятием «роль текста», а не деталями модели.

### POST /embed

```jsonc
// Запрос
{
  "texts": ["Хинкальная на Руставели", "..."],
  "kind": "passage"          // "passage" при индексации, "query" при поиске
}

// Ответ 200
{
  "model": "intfloat/multilingual-e5-base",
  "dim": 768,
  "normalized": true,
  "vectors": [[0.013, -0.041, ...], [...]]
}
```

Разделение `passage` / `query` обязательно. Модели семейства e5 требуют
префиксов `passage: ` и `query: `, без них качество заметно падает. Модель
семейства paraphrase-mpnet префиксов не требует и игнорирует параметр.
Именно поэтому решение принимается внутри сервиса: смена модели не требует
изменений в боте и воркерах.

Векторы возвращаются L2-нормализованными. При нормализованных векторах
косинусное расстояние `<=>` согласовано со скалярным произведением, и
`ORDER BY embedding <=> :q` корректен без дополнительных преобразований.

Ошибки:

| Код | Когда |
|---|---|
| 400 | Пустой `texts`, неизвестный `kind`, превышен лимит батча |
| 413 | Суммарная длина текстов больше лимита |
| 429 | Превышена конкурентность (семафор) |
| 503 | Модель ещё не загружена |

### GET /model

```jsonc
{
  "name": "intfloat/multilingual-e5-base",
  "dim": 768,
  "max_seq_length": 512,
  "tokenizer": "xlm-roberta-base",
  "normalized": true,
  "requires_prefix": true
}
```

Бот и воркеры вызывают этот эндпоинт на старте и сверяют `dim` с
константой из конфига. Несовпадение - фатальная ошибка запуска, а не тихая
запись мусора в `VECTOR(768)`. `tokenizer` используется чанкером для счёта
токенов.

### GET /healthz, GET /readyz

`/healthz` отвечает сразу после старта процесса. `/readyz` отвечает 200
только после того, как модель загружена в память и прогрета одним пробным
прогоном. Kubernetes не должен слать трафик в под, где первый запрос будет
ждать загрузки модели.

### Внутреннее устройство

- Один процесс uvicorn, `workers=1`. Несколько воркеров означают несколько
  копий модели в памяти одного пода.
- Масштабирование - репликами пода, не процессами.
- `torch.set_num_threads()` выставляется по CPU-лимиту пода, иначе torch
  берёт число ядер узла и деградирует на троттлинге cgroup.
- Семафор на число одновременных прогонов, за ним - очередь с таймаутом,
  далее 429.
- Внутренний микробатчинг: запросы, пришедшие в пределах нескольких
  десятков миллисекунд, объединяются в один прогон.
- Веса модели запечены в образ, не скачиваются при старте. Скачивание при
  старте означает зависимость холодного старта от внешней сети и риск
  упереться в лимиты хаба.

## Translate-сервис

Отдельный сервис, отдельная модель (NLLB-200-distilled-600M) - та же
причина разделения, что и у embedding-сервиса: всё специфичное для
конкретной модели перевода (коды языков FLORES-200, детекция языка,
батчинг по языку) живёт здесь, вызывающий код оперирует только «текст +
целевой язык».

Появился из-за измеренной проблемы: multilingual-e5 сравнивает эмбеддинги
кросс-язычно заметно хуже, чем внутри одного языка - один и тот же запрос
на иврите к заметке на иврите давал distance 0.137, а к заметке на русском
(после перевода обеих сторон на общий язык для сравнения) - 0.247, то есть
без перевода релевантная, но иноязычная заметка нередко не проходит
`SEARCH_MAX_DISTANCE`. См. `04-search.md`, "Перевод перед эмбеддингом".

### POST /translate

```jsonc
// Запрос
{
  "texts": ["где покататься на велике в тель авиве", "..."],
  "target_lang": "eng_Latn"   // опционально, иначе TRANSLATE_DEFAULT_TARGET_LANG
}

// Ответ 200
{
  "model": "facebook/nllb-200-distilled-600M",
  "items": [
    {"text": "where to ride a bike in tel aviv", "source_lang": "ru", "translated": true}
  ]
}
```

Каждый элемент `texts` детектируется независимо (py3langid, ISO 639-1) и
переводится только если его язык не совпадает с целевым и попадает в
таблицу известных языков сервиса - иначе `translated: false` и текст
возвращается как есть. Не ошибка: тот же принцип деградации, что у
экстракторов в `03-ingest.md` - лучше вернуть исходный текст, чем упасть
или угадать код языка неверно.

Ошибки: тот же набор кодов, что у embedding-сервиса (400/413/429/503) - см.
таблицу выше, смысл идентичен.

### GET /model

```jsonc
{"name": "facebook/nllb-200-distilled-600M", "default_target_lang": "eng_Latn"}
```

### GET /healthz, GET /readyz

Как у embedding-сервиса.

### Внутреннее устройство

Совпадает с embedding-сервисом (один uvicorn-процесс, масштабирование
репликами, веса запечены в образ) с одним отличием: NLLB - seq2seq-модель
(`model.generate`, несколько forward pass на текст, а не один encode), в
разы дороже по CPU на единицу текста, чем e5's encode - отсюда более
высокий `resources.limits.cpu` в `deploy/k8s/translate.yaml`.

### Клиент в notes-bot и деградация

`HttpTranslateClient` (`clients/translate.py`) вызывается из двух мест:
`process_note_async` (текст заметки, перед `embed_passages`) и `run_search`
(поисковый запрос, перед `embed_query`). В обоих местах вызов
best-effort - `TranslateServiceError` перехватывается и логируется, а
эмбеддинг считается по исходному, непереведённому тексту. `chunk_text` в
БД (то, что видит пользователь в карточке) - всегда исходный, непереведённый
текст независимо от исхода перевода; переводу подвергается только то, что
уходит в `embed_passages`/`embed_query`.

`TRANSLATE_ENABLED=false` отключает вызовы целиком (`translate_client=None`)
- то же поведение, что при недоступном сервисе, только без сетевого вызова
и без записи в лог на каждую заметку/поиск.

## ai-proxy

Контракт зафиксирован по репозиторию `poohpeer/ai-proxy` (README на коммите
`e648574`). ai-proxy - хост-резидентный демон с единым REST-контрактом над
несколькими провайдерами. Для этого бота используется провайдер `codex`
(Codex CLI на подписочной аутентификации ChatGPT-аккаунта), это указание
владельца продукта. Взаимодействие по-прежнему изолировано портом
`LLMClient`, чтобы смена провайдера не задевала вызывающий код.

### HTTP-контракт ai-proxy

```jsonc
// POST {AI_PROXY_URL}/v1/complete
{
  "provider": "codex",          // фиксировано для этого бота
  "prompt": "...",              // обязателен, min_length 1
  "system": "системный промпт", // опционально; для codex рендерится в developer_instructions
  "output_format": "text",      // "text" | "json"; см. ниже, почему для codex всегда "text"
  "json_schema": null,          // обязателен только при output_format="json"
  "timeout_s": 60,              // опционально; дефолт для CLI-провайдеров 300 с
  "history": []                 // опционально: [{"role":"user"|"assistant","content":"..."}], старые первыми
}
```

Особенности провайдера `codex`, влияющие на контракт:

- **`model` не передаётся.** Codex-CLI на ChatGPT-аккаунте всегда запускает
  дефолтную модель аккаунта и отвергает явное имя. В теле запроса поле `model`
  опускается (или `null`).
- **Изображения не поддерживаются** (`supports_images() == false`). Для этого
  бота не нужно - в LLM уходит только текст.
- **Структурированный вывод по схеме недоступен.** `output_format="json"`
  требует `json_schema` (иначе `400 missing_schema`), но CLI-адаптер codex
  отвечает прозой и не заполняет `structured_output` - он всегда вернётся
  `null`. Это проверено на соседнем проекте `bot-organizer`: запрос JSON по
  схеме к CLI-провайдеру приходит пустым. Поэтому для codex используется
  `output_format="text"`, а разбор JSON выполняет адаптер (см. ниже).
- **Таймаут.** Дефолт CLI-провайдеров - 300 секунд (`AI_PROXY_CLI_TIMEOUT_S`).
  Это существенно больше бюджета синхронного ожидания в чате, поэтому
  `timeout_s` передаётся явно под каждую задачу (обогащение - минуты допустимо,
  `/smart_search` - см. `04-search.md`, сделан асинхронным).

Ответ (`output_format="text"`):

```jsonc
{
  "provider": "codex",
  "model": "...",               // модель, реально обслужившая запрос
  "output_format": "text",
  "result": "текст ответа модели",
  "structured_output": null,
  "tool_calls": [],
  "metadata": { "cost_usd": 0.0, "duration_ms": 1234, "exit_code": 0, "raw_envelope": {} }
}
```

### Ошибки ai-proxy

Единый формат: `{ "error": { "type": "...", "message": "...", "detail": "..." } }`.

| HTTP | `type` | Как обрабатываем |
|---|---|---|
| 503 | `quota_exhausted` | Все Codex-аккаунты исчерпали квоту. **Не ретраить вслепую** - квота восстанавливается по времени. Обогащение: `enrich_status='failed'` после исчерпания попыток с длинной задержкой. `/smart_search`: деградация до `/search` |
| 504 | `timeout` | Превышен `timeout_s`. Обогащение ретраит, `/smart_search` деградирует |
| 500 | `cli_error` / `parse_error` | Сбой CLI или неразборный вывод. Ретрай обогащения, деградация поиска |
| 502 | `provider_error` / `provider_unreachable` | ai-proxy недоступен или провайдер упал. Как 500 |
| 400 | `unavailable_provider`, `missing_schema`, ... | Ошибка запроса - баг интеграции, не ретраить, логировать |

Ключевое отличие `503 quota_exhausted` от прочих ошибок: это не сбой запроса,
а исчерпание общего ресурса подписки на время. Обработчик обязан отличать его
и не гонять ретраи, которые лишь умножат нагрузку на исчерпанный аккаунт.

### Порт LLMClient

```python
class LLMClient(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,  # если задан - адаптер сам распарсит JSON из текста
        history: list[dict] | None = None,
        timeout_s: float = 60.0,
    ) -> LLMResult: ...

@dataclass
class LLMResult:
    text: str
    parsed: dict | None       # заполняется, если был json_schema и разбор удался
    model: str | None
    usage: dict | None        # metadata из ответа: cost_usd, duration_ms
```

Порт не содержит `max_tokens` и `temperature`: ai-proxy их не принимает
(`CompleteRequest` таких полей не имеет), и передавать их некуда.

**Fallback на `claude_code`.** `ProxyAILLMClient` сначала всегда пробует
`provider="codex"`; при `503 quota_exhausted` (все codex-аккаунты
исчерпали лимит одновременно - случалось в проде) повторяет тот же запрос
с `provider="claude_code"`, прежде чем сдаться. Это подписочный логин
Claude Code CLI (`claudeAiOauth`/`subscriptionType` в ai-proxy, не
API-ключ) - списывается квота подписки, а не деньги по счётчику токенов.
Любая другая ошибка (`timeout`, `provider_unreachable`, обрыв соединения)
fallback не запускает: другой провайдер не чинит то, что не является
нехваткой квоты. Парсинг JSON (ниже) работает одинаково независимо от
того, какой из двух провайдеров реально ответил.

Три реализации:

| Реализация | Назначение |
|---|---|
| `ProxyAILLMClient` | Боевая. `POST /v1/complete` с `provider="codex"` (fallback на `claude_code` при `quota_exhausted`), `output_format="text"`, явным `timeout_s`; при `json_schema` сама извлекает и валидирует JSON |
| `NullLLMClient` | Заглушка. `title` = первые слова текста, `tags` = пусто, `summary` = None, дубли не детектируются, `/smart_search` деградирует до `/search` |
| `FakeLLMClient` | Тесты. Детерминированные ответы по ключу промпта |

`NullLLMClient` включается флагом `LLM_ENABLED=false` и позволяет довести до
рабочего состояния всё, кроме генеративных фич, не дожидаясь готовности
интеграции. Это причина, по которой обогащение не стоит на критическом пути.

**Разбор JSON адаптером.** codex структурированный вывод по схеме не отдаёт,
поэтому `ProxyAILLMClient` реализует его сам: в промпт добавляется требование
вернуть только JSON нужной формы, ответ приходит как `result` (текст), из него
извлекается JSON (снятие ```json-ограждения), парсится и валидируется по
`json_schema`. При неудаче - одна повторная попытка с сообщением об ошибке
парсинга, затем `parsed=None`. Вызывающий код обязан корректно переживать
`parsed=None` - это не исключительная ситуация.

### Безопасность: блокер для LLM_ENABLED=true

**`LLM_ENABLED=true` включается только после фикса в ai-proxy** (см. `07-decisions.md`,
ADR-14 и риск R5). ai-proxy запускает codex командой `codex exec --json
--dangerously-bypass-approvals-and-sandbox` - модель выполняет любые
shell-команды без песочницы и подтверждения. Текст заметки - произвольный
контент из интернета (страницы, субтитры, чужие сообщения в группе), в котором
возможна инъекция инструкций. При включённом обогащении такой текст попадает в
`prompt` и может заставить codex выполнить команду в своём контейнере, где
смонтированы OAuth-токены аккаунтов.

Требование к ai-proxy: для запросов без `mcp_url` (обогащение и `/smart_search`
инструментов не используют) codex должен запускаться в песочнице только для
чтения, а не с `--dangerously-bypass-approvals-and-sandbox`. Фикс - в репозитории
`poohpeer/ai-proxy`, задача заведена там же. До него единственная безопасная
конфигурация бота - `LLM_ENABLED=false`.

### Развёртывание ai-proxy

В Kubernetes ai-proxy - внутренний `ClusterIP`-сервис `ai-proxy` на порту
`8787`, наружу не выставлен, без аутентификации. Из того же неймспейса -
`http://ai-proxy:8787`, из другого - `http://ai-proxy.<namespace>.svc.cluster.local:8787`.
`PROXY_AI_URL` в конфиге бота указывает на этот адрес; для одного неймспейса с
ai-proxy - `http://ai-proxy:8787`.

### Промпты

Живут в `notes_bot/clients/prompts/` как отдельные файлы, по одному на
задачу: `title.md`, `tags.md`, `place.md`, `summary.md`, `dupes.md`,
`query_rewrite.md`, `rag_answer.md`. Причины: их правят чаще кода, их удобно
диффать, и они нужны в тестах как эталон.

Общие требования ко всем промптам:

- отвечать на языке заметки (ru / en / he);
- теги в нижнем регистре, без решётки, не более пяти;
- при невозможности выполнить задачу возвращать пустой результат, а не
  выдумывать;
- текст заметки передаётся как данные в отдельной секции, а не вклеивается
  в инструкцию. Заметка содержит произвольный текст из интернета, включая
  попытки инъекции промпта; инструкции модели не должны быть перемешаны с
  ним.

## Внутренние интерфейсы

```python
# Экстракторы - см. 03-ingest.md
class Extractor(Protocol):
    source_type: str
    queue: Literal["fast", "heavy"]
    async def extract(self, note: Note) -> ExtractResult: ...

# Эмбеддинги
class EmbeddingClient(Protocol):
    async def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    @property
    def model_name(self) -> str: ...
    @property
    def dim(self) -> int: ...

# Перевод перед эмбеддингом - см. "Translate-сервис" выше
class TranslateClient(Protocol):
    async def translate_passages(self, texts: list[str]) -> list[str]: ...
    async def translate_query(self, text: str) -> str: ...

# Чанкинг - чистая функция, без ввода-вывода
def chunk(text: str, *, tokenizer, target: int, overlap: int, min_size: int,
          max_chunks: int) -> list[Chunk]: ...

# ACL - возвращает предикат для SQLAlchemy, не строку
def visibility_predicate(*, user_id: int, chat_id: int, is_group_chat: bool,
                         search_mode: str) -> ColumnElement[bool]: ...
```

## Конфигурация

`pydantic-settings`, всё через переменные окружения, без файлов конфигурации
в образе.

| Переменная | Пример | Источник в k8s |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | `123:ABC` | Secret |
| `DATABASE_URL` | `postgresql+psycopg://...` | Secret |
| `REDIS_URL` | `redis://redis:6379/3` | Secret |
| `EMBEDDINGS_URL` | `http://notes-embeddings:8000` | ConfigMap |
| `EMBEDDING_DIM` | `768` | ConfigMap |
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-base` | ConfigMap |
| `TRANSLATE_ENABLED` | `true` | ConfigMap |
| `TRANSLATE_URL` | `http://notes-translate:8000` | ConfigMap |
| `TRANSLATE_TARGET_LANG` | `eng_Latn` | ConfigMap |
| `TRANSLATE_TIMEOUT_S` | `20` | ConfigMap |
| `PROXY_AI_URL` | `http://ai-proxy:8787` | ConfigMap |
| `LLM_ENABLED` | `false` до фикса песочницы в ai-proxy (ADR-14) | ConfigMap |
| `LLM_ENRICH_TIMEOUT_S` | `120` | ConfigMap |
| `LLM_SMART_SEARCH_TIMEOUT_S` | `180` | ConfigMap |
| `WHISPER_MODEL` | `small` | ConfigMap |
| `MAX_AUDIO_SECONDS` | `1200` | ConfigMap |
| `MAX_DOWNLOAD_BYTES` | `104857600` | ConfigMap |
| `SEARCH_CANDIDATE_K` | `200` | ConfigMap |
| `SEARCH_PAGE_SIZE` | `5` | ConfigMap |
| `LOG_LEVEL` | `INFO` | ConfigMap |

Валидация на старте: обязательные переменные присутствуют, `EMBEDDING_DIM`
совпадает с ответом `GET /model`, соединение с Postgres и Redis проверяется
одним запросом. Падать при старте лучше, чем работать вполсилы.
