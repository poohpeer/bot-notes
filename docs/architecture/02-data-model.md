# 02. Модель данных

Схема расширяет DDL из дизайн-дока. Все расширения перечислены в конце
документа с обоснованием.

## DDL

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────────────────────────────────────
-- notes
-- ─────────────────────────────────────────────────────────────
CREATE TABLE notes (
    id              BIGSERIAL PRIMARY KEY,

    -- Идентичность в Telegram и идемпотентность
    user_id         BIGINT NOT NULL,          -- автор
    chat_id         BIGINT NOT NULL,          -- скоуп видимости: личный чат или группа
    is_group        BOOLEAN NOT NULL,
    tg_message_id   BIGINT,                   -- NULL для импортированных и для замен
    visibility      TEXT,                     -- 'private' | 'public'; NULL для групповых

    -- Контент
    source_type     TEXT NOT NULL,            -- 'text'|'voice'|'page'|'youtube'|'instagram'|'map'|'table_event'
    source_url      TEXT,
    raw_text        TEXT,                     -- ровно то, что прислал пользователь
    extracted_text  TEXT,                     -- то, что добыли экстракторы
    lang            TEXT,                     -- ISO-639-1, определяется при обработке

    -- Обогащение через proxy-ai
    title           TEXT,
    summary         TEXT,
    tags            TEXT[] NOT NULL DEFAULT '{}',
    structured      JSONB  NOT NULL DEFAULT '{}'::jsonb,

    -- Состояние обработки
    status          TEXT NOT NULL DEFAULT 'pending',
    enrich_status   TEXT NOT NULL DEFAULT 'pending',
    attempts        INT  NOT NULL DEFAULT 0,
    error           TEXT,

    deleted_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT notes_status_ck
        CHECK (status IN ('pending','processing','done','failed')),
    CONSTRAINT notes_enrich_status_ck
        CHECK (enrich_status IN ('pending','processing','done','failed','skipped')),
    CONSTRAINT notes_source_type_ck
        CHECK (source_type IN
            ('text','voice','page','youtube','instagram','map','table_event')),
    -- Ключевой инвариант приватности: групповая заметка не имеет visibility,
    -- личная - обязана иметь. Проверяется базой, а не только кодом.
    CONSTRAINT notes_visibility_ck CHECK (
        (is_group AND visibility IS NULL)
        OR (NOT is_group AND visibility IN ('private','public'))
    )
);

-- Идемпотентность приёма: повторная доставка того же апдейта не создаёт дубль
CREATE UNIQUE INDEX uq_notes_tg_message
    ON notes (chat_id, tg_message_id)
    WHERE tg_message_id IS NOT NULL;

CREATE INDEX idx_notes_tags        ON notes USING GIN (tags);
CREATE INDEX idx_notes_owner_live  ON notes (user_id, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX idx_notes_chat_live   ON notes (chat_id, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX idx_notes_trash       ON notes (user_id, deleted_at DESC) WHERE deleted_at IS NOT NULL;
CREATE INDEX idx_notes_gc          ON notes (deleted_at) WHERE deleted_at IS NOT NULL;
-- Для добора незавершённых задач после падения воркера
CREATE INDEX idx_notes_unfinished  ON notes (status, updated_at) WHERE status IN ('pending','processing');

-- ─────────────────────────────────────────────────────────────
-- note_chunks
-- ─────────────────────────────────────────────────────────────
CREATE TABLE note_chunks (
    id              BIGSERIAL PRIMARY KEY,
    note_id         BIGINT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    chunk_index     INT    NOT NULL,
    chunk_text      TEXT   NOT NULL,
    token_count     INT,
    embedding       VECTOR(768) NOT NULL,
    embedding_model TEXT   NOT NULL,          -- имя модели, которой посчитан вектор
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (note_id, chunk_index)
);

CREATE INDEX idx_note_chunks_note ON note_chunks (note_id);

-- ANN-индекс. Создаётся сразу, но на малых объёмах планировщик может
-- предпочесть точный скан - это нормально и даёт более точный результат.
CREATE INDEX idx_note_chunks_embedding ON note_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ─────────────────────────────────────────────────────────────
-- Настройки
-- ─────────────────────────────────────────────────────────────
CREATE TABLE user_settings (
    user_id            BIGINT PRIMARY KEY,
    search_mode        TEXT NOT NULL DEFAULT 'all',
    default_visibility TEXT NOT NULL DEFAULT 'private',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT user_settings_search_mode_ck
        CHECK (search_mode IN ('all','mine_only')),
    CONSTRAINT user_settings_default_visibility_ck
        CHECK (default_visibility IN ('private','public'))
);

CREATE TABLE chat_settings (
    chat_id      BIGINT PRIMARY KEY,
    title        TEXT,
    capture_mode TEXT NOT NULL DEFAULT 'mentions_and_replies',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chat_settings_capture_mode_ck
        CHECK (capture_mode IN ('all','mentions_and_replies'))
);
```

## Инварианты

1. **Приватность fail-closed.** Личная заметка не может существовать без
   `visibility`; дефолт `'private'` проставляется в момент вставки, до того
   как пользователь нажмёт кнопку. Не нажал - заметка остаётся приватной.
2. **Групповая заметка не имеет visibility.** Видимость определяется
   принадлежностью к `chat_id`. Проверяется CHECK-констрейнтом.
3. **Заметка видна в поиске только при `status='done'`.** Частично
   обработанная заметка не должна попадать в выдачу с пустыми чанками.
4. **Владение неизменно.** `user_id` не меняется никогда. Удаление,
   восстановление и редактирование доступны только при
   `notes.user_id = current_user_id`, независимо от `visibility` и группы.
5. **Чанки всегда согласованы с текстом.** Ре-чанкинг выполняется как
   `DELETE FROM note_chunks WHERE note_id = :id` плюс вставка новых в одной
   транзакции. Промежуточного состояния "часть старых, часть новых" нет.
6. **Все векторы в таблице посчитаны одной активной моделью.** См. ниже.

## Текст, идущий в индекс

Чанкинг работает не по `raw_text` и не по `extracted_text`, а по их
композиции, зависящей от типа источника:

| `source_type` | Текст для индексации |
|---|---|
| `text` | `raw_text` |
| `voice` | `extracted_text` = `caption + "\n\n" + транскрипт` (собирается экстрактором, как у `instagram`; `caption` в `raw_text`, если он был) |
| `page` | `extracted_text`, при пустом - `raw_text` (то есть сам URL) |
| `youtube` | `заголовок + описание + субтитры` (собирается экстрактором в `extracted_text`) |
| `instagram` | `caption + "\n\n" + транскрипт` (собирается экстрактором в `extracted_text`) |
| `map` | имя места из финального URL |
| `table_event` | `raw_text` - одно событие, извлечённое из скриншота таблицы (03-ingest.md, "/events_table"); нет `extracted_text`, нет экстрактора - создаётся сразу готовым по подтверждению пользователя |

Разделение `raw_text` / `extracted_text` нужно, потому что редактирование
разрешено только для `source_type='text'`: пользователь правит `raw_text`, а
`extracted_text` вообще не показывается как редактируемое. Хранение
пользовательского ввода отдельно также даёт возможность переизвлечь контент
заново, если экстрактор починили, не теряя исходную ссылку и подпись.

## Смена embedding-модели

Колонка `embedding_model` хранит имя модели для каждого чанка. Смешивать
векторы разных моделей в одном ANN-поиске нельзя - расстояния несопоставимы.
Процедура миграции:

1. Поднять новую версию embedding-сервиса рядом, под другим именем.
2. Фоновой задачей пересчитать чанки пачками, записывая новые строки с новым
   `embedding_model`; старые не удалять.
3. Поисковые запросы до конца миграции содержат
   `WHERE embedding_model = :active_model` с текущим активным значением.
4. После завершения переключить активное имя в конфиге и удалить старые
   строки.

Обе модели-кандидата из дизайн-дока дают размерность 768 (почти точно), так
что `VECTOR(768)` покрывает оба варианта и выбор модели не блокирует
миграции. Если финально будет выбрана модель другой размерности, потребуется
`ALTER TABLE ... ALTER COLUMN embedding TYPE VECTOR(N)` с полным
пересчётом - поэтому размерность вынесена в одну константу конфига и
проверяется на старте сверкой с `GET /model` embedding-сервиса.

## Расширения относительно дизайн-дока

| Что добавлено | Зачем |
|---|---|
| `notes.title`, `notes.summary` | Дизайн-док описывает LLM-фичи "заголовок заметки" и "суммаризация", но полей под них в схеме нет |
| `notes.structured` (JSONB) | Фича "структурные поля для мест (название, район, кухня)"; JSONB вместо колонок, потому что набор полей зависит от типа места и будет меняться |
| `notes.extracted_text` | Отделить пользовательский ввод от автоизвлечённого, см. раздел выше |
| `notes.tg_message_id` + уникальный индекс | Идемпотентность при ретраях Telegram и при перезапуске бота |
| `notes.enrich_status` | Обогащение через proxy-ai может упасть отдельно от основной обработки; заметка при этом остаётся валидной и находимой |
| `notes.attempts`, `notes.error` | Разбор упавших задач без чтения логов воркера |
| `notes.lang` | Мультиязычные заметки (ru/en/he); полезно для выбора промпта и для будущей фильтрации |
| `notes.updated_at` | Редактирование требует отслеживания |
| `note_chunks.chunk_index` + UNIQUE | Порядок чанков и защита от частичной вставки |
| `note_chunks.embedding_model` | Миграция между моделями, см. выше |
| `note_chunks.token_count` | Диагностика качества чанкинга |
| NOT NULL на `note_id`, `chunk_text`, `embedding` | В исходном DDL все три nullable, что допускает чанк без вектора |
| HNSW-индекс | В исходном DDL индекса на `embedding` нет вообще |
| Индексы по `user_id`, `chat_id`, `deleted_at` | Все три сценария поиска и `/list`, `/trash`, GC фильтруют по ним |
| `user_settings.default_visibility` | Пользователю, который всё делает публичным, не нужно нажимать кнопку каждый раз |
| `chat_settings` | Управление шумом в группах, см. `07-decisions.md`, ADR-10 |
| `source_type = 'voice'` | Голосовые заметки от пользователя, см. `07-decisions.md`, ADR-13 |
| CHECK-констрейнты | Инварианты приватности и статусов в базе, а не только в коде |

## Миграции

Alembic, одна миграция на изменение, обязательный `downgrade`.
`CREATE INDEX CONCURRENTLY` для индексов на `note_chunks` после того, как в
таблице появятся данные. Расширение `vector` создаётся отдельной первой
миграцией, потому что требует прав суперпользователя и на общем инстансе
может быть уже установлено.
