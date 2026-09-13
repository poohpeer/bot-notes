# Notes Bot

Telegram-бот для сохранения заметок (ссылки и свободный текст) с
семантическим поиском через embeddings. Поддержка нескольких пользователей и
общих «комнат» на базе Telegram-групп.

## Состояние

Стадия проектирования. Кода пока нет.

- [Дизайн-документ](docs/design/notes-bot-design.md) - исходные требования
- [Архитектура](docs/architecture/README.md) - целевое устройство системы
- [Этапы имплементации](docs/architecture/08-roadmap.md) - план работ M0-M9

## Стек

Python, Postgres + pgvector, Redis + RQ, FastAPI + sentence-transformers для
эмбеддингов, faster-whisper для транскрипции, Kubernetes.
