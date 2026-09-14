# Weights baked in at build time (BUILD_MODEL_NAME), not downloaded on
# container start — see docs/architecture/06-deployment.md, "Образы": a
# cold start should not depend on external network or hub rate limits.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ARG BUILD_MODEL_NAME=intfloat/multilingual-e5-base

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY src ./src
RUN uv sync --no-dev --no-editable

ENV EMBEDDINGS_MODEL_NAME=${BUILD_MODEL_NAME}
ENV EMBEDDINGS_MODEL_CACHE_DIR=/models

# Downloads once at build time and caches under EMBEDDINGS_MODEL_CACHE_DIR;
# nothing at container start touches the network for this. No leading
# whitespace on the continuation lines: Docker's backslash-continuation
# preserves it verbatim in the shell string handed to `python -c`, which
# then fails with IndentationError — this must be flush left.
RUN uv run --no-sync python -c "\
from embeddings_service.model import SentenceTransformerModel; \
SentenceTransformerModel('${BUILD_MODEL_NAME}', '/models')"

EXPOSE 8000

# workers=1: several would mean several copies of the model in one pod's
# memory — scale by replica instead, see main.py's module docstring.
CMD ["uv", "run", "--no-sync", "uvicorn", "embeddings_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
