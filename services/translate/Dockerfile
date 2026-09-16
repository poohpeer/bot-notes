# Weights baked in at build time (BUILD_MODEL_NAME), not downloaded on
# container start — same reasoning as services/embeddings/Dockerfile: a cold
# start should not depend on external network or hub rate limits.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ARG BUILD_MODEL_NAME=facebook/nllb-200-distilled-600M

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY src ./src
RUN uv sync --no-dev --no-editable

ENV TRANSLATE_MODEL_NAME=${BUILD_MODEL_NAME}
ENV TRANSLATE_MODEL_CACHE_DIR=/models

# Downloads once at build time and caches under TRANSLATE_MODEL_CACHE_DIR;
# nothing at container start touches the network for this. No leading
# whitespace on the continuation lines — see embeddings/Dockerfile's own
# comment on why (Docker's backslash-continuation preserves it verbatim,
# breaking python -c with IndentationError).
RUN uv run --no-sync python -c "\
from translate_service.model import NllbTranslationModel; \
NllbTranslationModel('${BUILD_MODEL_NAME}', '/models')"

EXPOSE 8000

# workers=1: several would mean several copies of the model in one pod's
# memory — scale by replica instead, see main.py's module docstring.
CMD ["uv", "run", "--no-sync", "uvicorn", "translate_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
