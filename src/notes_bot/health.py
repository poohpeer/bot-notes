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

    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    return app


async def mark_ready_after_get_me(bot: Bot, app: web.Application) -> None:
    """Called once at startup — see cli/bot.py. A failed getMe leaves
    `ready` False; the pod stays un-Ready rather than serving traffic it
    can't actually handle."""
    await bot.get_me()
    app[_READY] = True
