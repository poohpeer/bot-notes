"""FastAPI app — see docs/architecture/05-contracts.md, "Translate-сервис".

Structure mirrors embeddings_service/main.py: `workers=1` (see this
service's Dockerfile CMD) — several workers would mean several copies of
the ~2.5GB NLLB model in one pod's memory, so scaling is by replica.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from translate_service.config import Settings, get_settings
from translate_service.model import NllbTranslationModel, TranslationModel

# Same root-logger gap as embeddings_service.main — see that file's comment.
# Kept in sync: the deploy job's own log-line check depends on this exact
# format for "model ready".
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")

log = logging.getLogger(__name__)

ModelLoader = Callable[[], TranslationModel]


class TranslateRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    target_lang: str | None = None


class TranslateItem(BaseModel):
    text: str
    source_lang: str | None
    translated: bool


class TranslateResponse(BaseModel):
    model: str
    items: list[TranslateItem]


class ModelInfo(BaseModel):
    name: str
    default_target_lang: str


def _default_loader(settings: Settings) -> ModelLoader:
    return lambda: NllbTranslationModel(settings.model_name, settings.model_cache_dir)


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
        # /healthz answers immediately while it runs — see 06-deployment.md.
        app.state.model = await asyncio.to_thread(loader)
        # Warms lazy-initialized state (tokenizer caches) so the first real
        # request isn't the slow one — see "GET /healthz, GET /readyz".
        await asyncio.to_thread(app.state.model.translate, ["warmup"], settings.default_target_lang)
        app.state.ready = True
        log.info("model ready: %s", app.state.model.name)
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
        model: TranslationModel | None = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return ModelInfo(name=model.name, default_target_lang=settings.default_target_lang)

    @app.post("/translate", response_model=TranslateResponse)
    async def translate(body: TranslateRequest, request: Request):
        model: TranslationModel | None = request.app.state.model
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

        target_lang = body.target_lang or settings.default_target_lang
        try:
            async with asyncio.timeout(settings.queue_timeout_s):
                async with semaphore:
                    results = await asyncio.to_thread(model.translate, body.texts, target_lang)
        except TimeoutError as exc:
            # Queued behind max_concurrency other runs for too long — the
            # service is saturated. 429, not 503: the model itself is fine.
            raise HTTPException(status_code=429, detail="translate service is at capacity") from exc

        return TranslateResponse(
            model=model.name,
            items=[
                TranslateItem(text=r.text, source_lang=r.source_lang, translated=r.translated)
                for r in results
            ],
        )

    return app


app = create_app()
