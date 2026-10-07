"""Dashboard backend tests (aiohttp test client + the FakeController).

    .venv/bin/python -m pytest tests/test_web.py -q
"""
import asyncio
import json
import logging
import os
import stat
import sys
import threading
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events import EventBus  # noqa: E402
from web.dev_server import FakeController  # noqa: E402
from web.server import BusLogHandler, create_app, load_or_create_token, start_dashboard  # noqa: E402

pytestmark = pytest.mark.asyncio

TOKEN = "test-token-123"
GUILD = "1100000000000000001"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def controller(bus):
    return FakeController(bus)


@pytest_asyncio.fixture
async def client(aiohttp_client, controller, bus):
    app = create_app(controller, bus, token=TOKEN, status_interval=0.2)
    return await aiohttp_client(app)


async def login(client):
    r = await client.get(f"/?token={TOKEN}", allow_redirects=False)
    assert r.status == 302
    assert r.headers["Location"] == "/"
    assert "HttpOnly" in r.headers["Set-Cookie"]
    assert TOKEN not in r.headers["Set-Cookie"]  # the cookie holds a derived value


# ---------------------------------------------------------------- auth

async def test_api_needs_auth(client):
    assert (await client.get("/api/status")).status == 401
    r = await client.post("/api/control", json={"action": "reload"})
    assert r.status == 401
    assert (await client.get("/ws")).status == 401


async def test_page_redirects_to_login(client):
    r = await client.get("/", allow_redirects=False)
    assert r.status == 302 and r.headers["Location"] == "/login"
    r = await client.get("/login")
    assert r.status == 200 and "<form" in await r.text()


async def test_wrong_token(client):
    r = await client.get("/?token=nope", allow_redirects=False)
    assert r.status == 302 and r.headers["Location"].startswith("/login")
    assert (await client.get("/api/status", headers={"Authorization": "Bearer nope"})).status == 401
    r = await client.post("/login", data={"token": "nope"})
    assert "token didn’t work" in await r.text()
    assert (await client.get("/api/status")).status == 401


async def test_token_link_sets_cookie(client):
    await login(client)
    r = await client.get("/api/status")
    assert r.status == 200
    r = await client.get("/")
    assert r.status == 200 and "app.js" in await r.text()


async def test_login_form_and_logout(client):
    r = await client.post("/login", data={"token": TOKEN}, allow_redirects=False)
    assert r.status == 302 and r.headers["Location"] == "/"
    assert (await client.get("/api/status")).status == 200
    await client.post("/logout", allow_redirects=False)
    assert (await client.get("/api/status")).status == 401


async def test_cross_origin_post_refused(client):
    await login(client)
    r = await client.post("/api/control", json={"action": "reload"},
                          headers={"Origin": "https://evil.example"})
    assert r.status == 403


async def test_static_is_public(client):
    r = await client.get("/static/app.js")
    assert r.status == 200
    assert "javascript" in r.headers["Content-Type"]
    assert (await client.get("/static/vendor/preact.js")).status == 200


# ---------------------------------------------------------------- API

async def test_status(client, controller):
    r = await client.get("/api/status", headers=AUTH)
    data = await r.json()
    assert data["bot"]["name"] == controller.bot_name
    assert {g["id"] for g in data["guilds"]} == set(controller.guilds)
    assert data["tts_queue"]["pending"] == len(controller.tts_items)


async def test_control_round_trip(client, controller):
    r = await client.post("/api/control", headers=AUTH,
                          json={"action": "toggle", "guild_id": GUILD, "name": "record", "value": True})
    assert r.status == 200
    data = await r.json()
    assert data["ok"] is True and data["message"] == "record on"
    assert controller.calls[-1] == ("toggle", {"guild_id": GUILD, "name": "record", "value": True})
    guild = next(g for g in data["status"]["guilds"] if g["id"] == GUILD)
    assert guild["toggles"]["record"] is True

    r = await client.post("/api/control", headers=AUTH,
                          json={"action": "say", "guild_id": GUILD, "text": "hola"})
    assert (await r.json())["message"] == "Queued"
    assert controller.tts_items[-1]["text"] == "hola"


async def test_control_value_error_is_400(client):
    r = await client.post("/api/control", headers=AUTH, json={"action": "explode"})
    assert r.status == 400
    assert (await r.json())["error"] == "unknown action: explode"
    r = await client.post("/api/control", headers=AUTH,
                          json={"action": "join", "guild_id": GUILD, "channel_id": "1"})
    assert r.status == 400 and "unknown voice channel" in (await r.json())["error"]


async def test_control_bad_body(client):
    r = await client.post("/api/control", headers=AUTH, data=b"not json")
    assert r.status == 400
    r = await client.post("/api/control", headers=AUTH, json={"guild_id": GUILD})
    assert r.status == 400


# ---------------------------------------------------------------- WebSocket

async def test_ws_hello_event_and_status(client, bus):
    bus.publish({"type": "feed", "id": 1, "guild_id": 2**62, "text": "before"})
    await login(client)
    async with client.ws_connect("/ws") as ws:
        hello = await ws.receive_json(timeout=2)
        assert hello["type"] == "hello"
        assert hello["recent"][0]["text"] == "before"
        assert hello["recent"][0]["guild_id"] == str(2**62)  # snowflakes go out as strings

        first = await ws.receive_json(timeout=2)
        assert first["type"] == "status" and "guilds" in first

        bus.publish({"type": "feed_update", "id": 1, "outcome": "played"})
        seen = []
        while True:
            msg = await ws.receive_json(timeout=2)
            seen.append(msg["type"])
            if msg["type"] == "feed_update":
                assert msg["outcome"] == "played"
                break

        # the periodic status push keeps coming
        while (await ws.receive_json(timeout=2))["type"] != "status":
            pass
    assert not bus._subscribers  # unsubscribed on close


async def test_ws_bearer_auth(client):
    async with client.ws_connect("/ws", headers=AUTH) as ws:
        assert (await ws.receive_json(timeout=2))["type"] == "hello"


# ---------------------------------------------------------------- helpers

async def test_bus_log_handler_from_thread(bus):
    loop = asyncio.get_running_loop()
    logger = logging.getLogger("test_web.bus")
    logger.propagate = False
    handler = BusLogHandler(bus, loop)
    logger.addHandler(handler)
    try:
        logger.info("too quiet")  # below WARNING: dropped
        t = threading.Thread(target=lambda: logger.error("boom %d", 42))
        t.start()
        t.join()
        await asyncio.sleep(0.05)
    finally:
        logger.removeHandler(handler)
    events = bus.recent()
    assert len(events) == 1
    assert events[0]["type"] == "log" and events[0]["level"] == "ERROR"
    assert events[0]["message"] == "boom 42"


async def test_load_or_create_token(tmp_path):
    path = tmp_path / "data" / "dashboard_token"
    token = load_or_create_token(path)
    assert len(token) >= 32
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert load_or_create_token(path) == token
    path.write_text("")
    assert load_or_create_token(path) != token


async def test_start_dashboard(controller, bus, aiohttp_client):
    runner = await start_dashboard(controller, bus, host="127.0.0.1", port=0, token=TOKEN)
    try:
        port = runner.addresses[0][1]
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/api/status", headers=AUTH) as r:
                assert r.status == 200
                assert json.loads(await r.text())["bot"]["connected"] is True
    finally:
        await runner.cleanup()
