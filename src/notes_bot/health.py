"""HTTP health server for the bot process — see docs/architecture/06-deployment.md,
"Health-пробы": liveness is `/healthz` (the process is alive); readiness is
the same idea plus a successful `getMe`, so the pod isn't marked Ready
before it has actually resolved its own identity against Telegram.

A separate small aiohttp server, not the bot's own long-polling loop:
aiogram already depends on aiohttp, so this adds no new dependency, and
long polling has no HTTP surface of its own for Kubernetes to probe.
"""

from __future__ import annotations

from aiogram import Bot
from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

_READY = web.AppKey("ready", bool)


def create_health_app(bot: Bot) -> web.Application:
    app = web.Application()
    app[_READY] = False

    async def healthz(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def readyz(request: web.Request) -> web.Response:
        if not request.app[_READY]:
            return web.json_response({"status": "not ready"}, status=503)
        return web.json_response({"status": "ok"})

    async def metrics(request: web.Request) -> web.Response:
        # 06-deployment.md, "Наблюдаемость" — same port as the health
        # probes, no separate Service/port needed for Prometheus to scrape.
        # aiohttp's `content_type=` param rejects a value carrying its own
        # charset (CONTENT_TYPE_LATEST is "text/plain; version=...;
        # charset=utf-8") — split it so aiohttp sets the header verbatim.
        return web.Response(body=generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/metrics", metrics)
    return app


async def mark_ready_after_get_me(bot: Bot, app: web.Application) -> None:
    """Called once at startup — see cli/bot.py. A failed getMe leaves
    `ready` False; the pod stays un-Ready rather than serving traffic it
    can't actually handle."""
    await bot.get_me()
    mark_ready(app)


def mark_ready(app: web.Application) -> None:
    """For a caller that already has its own successful getMe result (see
    cli/bot.py, which also needs the bot's username) and would otherwise
    have to call getMe a second time just for this."""
    app[_READY] = True
