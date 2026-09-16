# notes-translate

Translation microservice — see `docs/architecture/05-contracts.md`, "Translate-сервис".

notes-bot searches by comparing embeddings, and multilingual-e5's cross-lingual
matching turned out weaker than translating both sides to one language first
(measured: a Hebrew query against a Hebrew note scored distance 0.137, the
same query translated to match a Russian note's translated text scored 0.247
— see `docs/architecture/04-search.md`, "Перевод перед эмбеддингом"). This service is that translation step: `POST
/translate` takes a batch of texts and a target FLORES-200 code, detects each
text's own language (py3langid), and runs NLLB-200-distilled-600M for any
that aren't already in the target language.

## Development

```bash
cd services/translate
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format .
```

Tests never load real weights — `tests/conftest.py`'s `FakeTranslationModel`
stands in for `NllbTranslationModel`. Run the real thing locally:

```bash
uv run uvicorn translate_service.main:app --port 8000
curl -s localhost:8000/model
curl -s -X POST localhost:8000/translate \
  -H 'content-type: application/json' \
  -d '{"texts":["где покататься на велике в тель авиве"],"target_lang":"eng_Latn"}'
```

## Docker

```bash
docker build -t notes-translate .
```

Downloads and bakes in NLLB's weights at build time — no network access
needed at container start, see the Dockerfile's comments.
