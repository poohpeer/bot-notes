"""FastAPI app — see docs/architecture/05-contracts.md, "Embedding-сервис".

`workers=1` at the uvicorn level (see the service's Dockerfile CMD): several
workers would mean several copies of the model in one pod's memory: scaling
is by replica, not by process, per that doc.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from embeddings_service.config import Settings, get_settings
from embeddings_service.model import EmbeddingModel, Kind, SentenceTransformerModel

# uvicorn's own dictConfig (see its --log-config default) only wires up the
# "uvicorn"/"uvicorn.error"/"uvicorn.access" loggers, not the root logger —
# without this, every log.info/log.warning call in this module (this file's
# "model ready" line included) propagates to a handler-less root logger and
# is silently dropped, while uvicorn's own access log lines print fine right
# next to them. Observed in CI: the deploy job's own log-line check for
# "model ready" failed even though the service was actually up and serving.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")

log = logging.getLogger(__name__)

ModelLoader = Callable[[], EmbeddingModel]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    kind: Literal["passage", "query"]


class EmbedResponse(BaseModel):
    model: str
    dim: int
    normalized: bool
    vectors: list[list[float]]


class ModelInfo(BaseModel):
    name: str
    dim: int
    max_seq_length: int
    tokenizer: str
    normalized: bool
    requires_prefix: bool


def _default_loader(settings: Settings) -> ModelLoader:
    return lambda: SentenceTransformerModel(settings.model_name, settings.model_cache_dir)


def create_app(
    settings: Settings | None = None, model_loader: ModelLoader | None = None
) -> FastAPI:
    settings = settings or get_settings()
    loader = model_loader or _default_loader(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.ready = False
        # Loading is blocking (disk + tensor init); off the event loop so
        # /healthz answers immediately while it runs — see 06-deployment.md
        # on why liveness must not wait for this.
        app.state.model = await asyncio.to_thread(loader)
        # One trial run warms lazy-initialized state (tokenizer caches,
        # thread pools) so the first real request isn't the slow one — see
        # "GET /healthz, GET /readyz".
        await asyncio.to_thread(app.state.model.embed, ["warmup"], "passage")
        app.state.ready = True
        log.info("model ready: %s (dim=%d)", app.state.model.name, app.state.model.dim)
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.semaphore = semaphore
    app.state.settings = settings

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request):
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok"}

    @app.get("/model", response_model=ModelInfo)
    async def model_info(request: Request):
        model: EmbeddingModel | None = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return ModelInfo(
            name=model.name,
            dim=model.dim,
            max_seq_length=model.max_seq_length,
            tokenizer=model.tokenizer_name,
            normalized=True,
            requires_prefix=model.requires_prefix,
        )

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(body: EmbedRequest, request: Request):
        model: EmbeddingModel | None = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        if any(not t for t in body.texts):
            raise HTTPException(status_code=400, detail="texts must not contain empty strings")
        if len(body.texts) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"batch of {len(body.texts)} exceeds max_batch_size={settings.max_batch_size}"
                ),
            )
        total_chars = sum(len(t) for t in body.texts)
        if total_chars > settings.max_total_chars:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"total length {total_chars} exceeds max_total_chars={settings.max_total_chars}"
                ),
            )

        try:
            async with asyncio.timeout(settings.queue_timeout_s):
                async with semaphore:
                    vectors = await asyncio.to_thread(model.embed, body.texts, _kind(body.kind))
        except TimeoutError as exc:
            # Queued behind max_concurrency other runs for too long — the
            # service is saturated. 429, not 503: the model itself is fine.
            raise HTTPException(status_code=429, detail="embedding service is at capacity") from exc

        return EmbedResponse(model=model.name, dim=model.dim, normalized=True, vectors=vectors)

    return app


def _kind(value: str) -> Kind:
    # EmbedRequest.kind is already validated as one of "passage"/"query" by
    # pydantic's Literal; this only narrows the type for model.embed.
    return value  # type: ignore[return-value]


app = create_app()
