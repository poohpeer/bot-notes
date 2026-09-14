# notes-embeddings

Embedding microservice — see `docs/architecture/05-contracts.md` for the full
contract and `docs/architecture/08-roadmap.md`, M1.

The rest of the system asks for a vector "for a text in the role of a query
or a passage"; everything model-specific (e5's `query: `/`passage: ` prefixes,
normalization, dimension) lives here, behind `GET /model` and `POST /embed`.

## Development

```bash
cd services/embeddings
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format .
```

Tests never load real weights — `tests/conftest.py`'s `FakeEmbeddingModel`
stands in for `SentenceTransformerModel`. Run the real thing locally:

```bash
EMBEDDINGS_MODEL_NAME=intfloat/multilingual-e5-base \
  uv run uvicorn embeddings_service.main:app --port 8000
curl -s localhost:8000/model
curl -s -X POST localhost:8000/embed \
  -H 'content-type: application/json' \
  -d '{"texts":["Хинкальная на Руставели"],"kind":"passage"}'
```

## Choosing the model — still open

`EMBEDDING_MODEL_NAME` picks the model; the service does not care which of
the two candidates it is. The roadmap's M1 "Готово, когда" asks for a
benchmark on real notes, which needs a labeled dataset this repo does not
have:

```bash
uv run python scripts/benchmark_models.py your_dataset.json
```

See the script's docstring for the dataset shape. Until this runs against
real ru/en/he notes, `EMBEDDING_DIM=768` and the default model name in
`deploy/k8s/configmap.yaml` are a starting point, not a decision — ADR-3
covers why both candidates fit `VECTOR(768)` and switching later only costs a
redeploy plus a vector recompute, not a schema change.

## Docker

```bash
docker build -t notes-embeddings --build-arg BUILD_MODEL_NAME=intfloat/multilingual-e5-base .
```

Downloads and bakes in the model's weights at build time — no network access
needed at container start, see the Dockerfile's comments.
