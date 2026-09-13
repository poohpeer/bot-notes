from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from notes_bot.health import create_health_app, mark_ready_after_get_me


class FakeBot:
    def __init__(self, *, fails: bool = False):
        self._fails = fails

    async def get_me(self):
        if self._fails:
            raise RuntimeError("network error")
        return object()


async def test_healthz_is_always_ok():
    app = create_health_app(FakeBot())
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/healthz")
        assert response.status == 200
        assert (await response.json())["status"] == "ok"


async def test_readyz_is_not_ready_before_get_me_succeeds():
    app = create_health_app(FakeBot())
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/readyz")
        assert response.status == 503


async def test_readyz_is_ok_after_get_me_succeeds():
    bot = FakeBot()
    app = create_health_app(bot)
    await mark_ready_after_get_me(bot, app)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/readyz")
        assert response.status == 200


async def test_mark_ready_leaves_app_not_ready_if_get_me_fails():
    bot = FakeBot(fails=True)
    app = create_health_app(bot)
    try:
        await mark_ready_after_get_me(bot, app)
    except RuntimeError:
        pass
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/readyz")
        assert response.status == 503
