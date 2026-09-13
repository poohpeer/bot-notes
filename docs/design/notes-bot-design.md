# Notes Bot — дизайн-документ

## Идея
Telegram-бот для сохранения заметок (ссылки + свободный текст) с семантическим поиском через embeddings. Поддержка нескольких пользователей и общих "комнат" (Telegram-группы) для совместных поездок/тем.

## Стек
- **Язык:** Python
- **БД:** Postgres + pgvector (переиспользуем инфраструктуру существующего family-bot)
- **Очередь:** RQ + Redis (переиспользуем существующий Redis)
- **Embeddings:** отдельный микросервис FastAPI + sentence-transformers, локально, self-hosted (НЕ Google Gemini — отказались из-за непрозрачности free-tier лимитов)
- **Генеративная LLM:** через существующий `proxy-ai` (API-контракт уточняется отдельно; используется ТОЛЬКО для генеративных задач, не для эмбеддингов)
- **Speech-to-text:** faster-whisper, модель `small` (CPU-only, GPU не планируется)
- **Деплой:** существующий Kubernetes-кластер

## Обработка контента по источникам

| Источник | Что извлекаем | Инструменты |
|---|---|---|
| Обычный текст | Как есть | — |
| Обычная ссылка | Текст страницы | `trafilatura` |
| YouTube | Заголовок + описание + субтитры (если есть, БЕЗ распознавания речи) | `yt-dlp` + `youtube-transcript-api` |
| Instagram | Caption поста + Whisper-транскрипт аудио (оба источника конкатенируются) | `yt-dlp` + `faster-whisper` |
| Ссылка на карту в описании | Только имя места из финального URL после редиректа, БЕЗ полного скрапинга страницы | лёгкий HTTP-редирект-резолв |

Прочие ссылки внутри описаний постов/видео НЕ обходятся — остаются как текст внутри заметки.

## Схема БД

```sql
CREATE TABLE notes (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,          -- кто отправил
    chat_id BIGINT NOT NULL,          -- личный чат или id группы
    is_group BOOLEAN NOT NULL,
    visibility TEXT,                   -- 'private' | 'public', NULL для групповых
    source_url TEXT,
    source_type TEXT,                  -- 'text' | 'page' | 'youtube' | 'instagram'
    raw_text TEXT,
    tags TEXT[] DEFAULT '{}',
    status TEXT DEFAULT 'pending',     -- 'pending' | 'processing' | 'done' | 'failed'
    deleted_at TIMESTAMPTZ DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_notes_tags ON notes USING GIN(tags);

CREATE TABLE note_chunks (
    id BIGSERIAL PRIMARY KEY,
    note_id BIGINT REFERENCES notes(id) ON DELETE CASCADE,
    chunk_text TEXT,
    embedding VECTOR(768)  -- размерность зависит от финального выбора embedding-модели
);

CREATE TABLE user_settings (
    user_id BIGINT PRIMARY KEY,
    search_mode TEXT NOT NULL DEFAULT 'all'  -- 'all' | 'mine_only'
);
```

## Приватность и мульти-тенантность

- **Личный чат с ботом:** при отправке заметки бот спрашивает приватность кнопками (🔒 приватная / 🌐 публичная)
- **Группа (Telegram-группа = "комната"):** приватность НЕ спрашивается, заметка видна всем участникам группы автоматически; visibility = NULL

### Поиск — три сценария

```sql
-- 1. Поиск из группы: строго внутри группы
WHERE chat_id = :current_chat_id AND deleted_at IS NULL

-- 2. Поиск из личного чата, режим "всё" (по умолчанию)
WHERE deleted_at IS NULL AND (
    user_id = :current_user_id
    OR (visibility = 'public' AND is_group = false)
)

-- 3. Поиск из личного чата, режим "только свои"
WHERE user_id = :current_user_id AND deleted_at IS NULL
```

Переключение режима — команды `/search_mine` и `/search_all`.

### Дедупликация результатов по заметке

```sql
SELECT DISTINCT ON (n.id) n.id, n.source_url, n.source_type, n.tags, nc.chunk_text,
       nc.embedding <=> :query_vector AS distance
FROM note_chunks nc
JOIN notes n ON n.id = nc.note_id
WHERE <условие из сценария выше>
ORDER BY n.id, distance
LIMIT 6 OFFSET :offset;  -- LIMIT 6, чтобы понять, есть ли ещё страница; показываем первые 5
```

### Пагинация
Кнопка "Показать ещё 5" отправляет НОВОЕ сообщение (не редактирует старое) — предыдущие результаты остаются в чате, отдельная кнопка "Назад" не нужна. Состояние offset хранится в Redis:
```
key: search:{user_id}:{session_id}
value: {"query_vector": [...], "search_mode": "all", "offset": 5}
```

## Редактирование

- Разрешено ТОЛЬКО для заметок `source_type = 'text'` (то, что юзер сам напечатал) — автоматически извлечённый контент (транскрипты, тексты страниц, caption) НЕ редактируется
- Editable поля: текст заметки (полный ре-чанкинг + ре-эмбеддинг) и privacy (мгновенный тоггл без пересчёта векторов)
- Флоу редактирования текста: юзер копирует видимый в сообщении текст, правит, отправляет как новое сообщение → бот подхватывает как замену (без повторной отправки контента ботом)

## Удаление

- Мягкое удаление (`deleted_at`), с шагом подтверждения (Да/Отмена) перед удалением
- Точки входа: кнопка в выдаче поиска И команда `/list`
- Восстановление через `/trash` (♻️ кнопка)
- Права: удалить/восстановить можно только свою заметку (`user_id = current_user_id`), даже если она публичная или в общей группе
- Окончательное (hard) удаление через 30 дней — Kubernetes CronJob, паттерн как у modelbench:
```sql
DELETE FROM notes WHERE deleted_at IS NOT NULL AND deleted_at < now() - interval '30 days';
-- note_chunks удалятся каскадно
```

## Теги
LLM (через proxy-ai) генерирует несколько тегов при сохранении заметки, хранятся в `notes.tags` (TEXT[] + GIN индекс), отображаются под заметкой в выдаче поиска.

## Группы как "комнаты" (для совместных поездок)
Вместо кастомной системы приглашений — используются обычные Telegram-группы:
- Друзья добавляются в группу "Грузия 2026" вместе с ботом (нужно выключить privacy mode бота у BotFather)
- Все сообщения в группе автоматически сохраняются с `chat_id` этой группы
- Поиск из группы скоупится строго на неё
- Заметки, отправленные пользователем в группе, ТАКЖЕ видны ему в личном поиске (через `user_id`), но не видны другим людям вне группы

### Импорт старой группы (например, с рецептами)
Bot API не даёт доступа к истории сообщений до момента добавления бота — это ограничение платформы, обхода через Bot API нет. Решение: экспорт истории через официальный Telegram Desktop (Settings → Export chat history → JSON) + одноразовый скрипт-импортёр, прогоняющий сообщения через тот же пайплайн (chunking → embedding → сохранение).

## LLM-фичи (генеративные, через proxy-ai)

| Фича | Когда | Sync/Async |
|---|---|---|
| Заголовок заметки | При сохранении | Async (через RQ) |
| Автотеги | При сохранении | Async (через RQ) |
| Структурные поля для мест (название, район, кухня) | При сохранении | Async (через RQ) |
| Детекция дублей | При сохранении | Async (через RQ) |
| Суммаризация длинных транскриптов | При сохранении | Async (через RQ) |
| Query rewriting | Только `/smart_search` | Sync, юзер ждёт |
| RAG-синтез ответа | Только `/smart_search` | Sync, юзер ждёт |

Обычный `/search` остаётся быстрым — чисто векторный поиск без LLM. `/smart_search` — отдельный режим с осознанным компромиссом "дольше, но качественнее" (CPU-only железо, без GPU).

## Открытые вопросы (не заблокированы, но не решены)
- API-контракт `proxy-ai` — будет добавлен отдельно
- Финальный выбор embedding-модели (`paraphrase-multilingual-mpnet-base-v2` vs `intfloat/multilingual-e5-base`) — нужно протестировать вручную на реальных ru/en/he заметках
- Финальный выбор LLM-модели для генеративных задач через proxy-ai — зависит от того, что proxy-ai предоставляет
