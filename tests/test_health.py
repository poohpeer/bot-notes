from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from notes_bot.health import create_health_app, mark_ready, mark_ready_after_get_me


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


async def test_mark_ready_directly_without_a_fresh_get_me_call():
    """cli/bot.py's path: it already has its own getMe result (for
    bot_username) and shouldn't need a second call just for readiness."""
    app = create_health_app(FakeBot())
    mark_ready(app)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/readyz")
        assert response.status == 200


async def test_metrics_endpoint_serves_prometheus_text_format():
    """06-deployment.md, "Наблюдаемость": scraped on the same port as the
    health probes, no separate Service needed."""
    app = create_health_app(FakeBot())
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/metrics")
        assert response.status == 200
        assert response.content_type == "text/plain"
        body = await response.text()
        # A couple of this codebase's own metric names, not just any output.
        assert "notes_search_duration_seconds" in body
        assert "notes_extractor_failures_total" in body
