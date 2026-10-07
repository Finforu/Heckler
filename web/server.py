"""Heckler's admin web dashboard.

Runs inside the bot's asyncio loop (aiohttp, no threads). Everything except the
login page and the static files needs the admin token, given once through the
login form or a `?token=` link; after that an HttpOnly cookie carries a value
derived from it (the raw token never sits in the cookie).

The bot side is a Controller (see docs/PLAN.md):

    status() -> dict                      # shape in docs/PLAN.md
    async control(action, **params) -> str  # ValueError on bad input
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import logging
import os
import secrets
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

from aiohttp import WSMsgType, web

from .routes_content import setup_content_routes
from .routes_media import setup_media_routes
from .routes_stats import setup_stats_routes
from .util import dumps, json_response, jsonable  # noqa: F401  (jsonable re-exported)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
COOKIE = "dash_session"
COOKIE_MAX_AGE = 30 * 24 * 3600
STATUS_INTERVAL_S = 2.0
MAX_BODY = 1024 * 1024  # JSON bodies; pack uploads stream with their own cap

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


class Controller(Protocol):
    def status(self) -> dict: ...
    async def control(self, action: str, **params: Any) -> str: ...


# ---------------------------------------------------------------- token

def load_or_create_token(path: str | os.PathLike) -> str:
    """The dashboard token saved at `path`; a new random one (mode 0600) if
    the file is missing or empty."""
    path = Path(path)
    try:
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    os.chmod(path, 0o600)  # in case the file already existed with wider permissions
    return token


def login_url(host: str, port: int, token: str) -> str:
    """A one-click login link for the startup log."""
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}/?{urlencode({'token': token})}"


def _session_value(token: str) -> str:
    return hmac.new(token.encode(), b"dc-bot dashboard session v1", hashlib.sha256).hexdigest()


def _token_ok(given: str | None, token: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), token.encode())


def _authed(request: web.Request) -> bool:
    token = request.app["token"]
    cookie = request.cookies.get(COOKIE)
    if cookie and hmac.compare_digest(cookie.encode(), request.app["session"].encode()):
        return True
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return _token_ok(auth[7:].strip(), token)
    return False


def _same_origin(request: web.Request) -> bool:
    """Browsers send Origin on POST and WebSocket requests; refuse other sites.
    No Origin (curl, scripts) is fine: those need the token anyway."""
    origin = request.headers.get("Origin")
    if not origin:
        return True
    return urlsplit(origin).netloc == request.host


def _set_session(response: web.StreamResponse, request: web.Request) -> None:
    response.set_cookie(COOKIE, request.app["session"], max_age=COOKIE_MAX_AGE, path="/",
                        httponly=True, samesite="Strict", secure=request.secure)


# ---------------------------------------------------------------- JSON

_dumps = dumps
_json = json_response


# ---------------------------------------------------------------- middleware

@web.middleware
async def auth_middleware(request: web.Request, handler):
    path = request.path
    if path.startswith("/static/") or path in ("/login", "/favicon.ico"):
        return await handler(request)

    # ?token=... link: check it, set the cookie, drop the token from the URL.
    if request.method == "GET" and "token" in request.query and not path.startswith(("/api/", "/ws")):
        if _token_ok(request.query.get("token"), request.app["token"]):
            rest = {k: v for k, v in request.query.items() if k != "token"}
            target = path + (f"?{urlencode(rest)}" if rest else "")
            response = web.HTTPFound(target)
            _set_session(response, request)
            raise response
        await asyncio.sleep(0.5)
        raise web.HTTPFound("/login?bad=1")

    if not _authed(request):
        if path.startswith(("/api/", "/ws")):
            return _json({"error": "unauthorized"}, status=401)
        raise web.HTTPFound("/login")
    if request.method != "GET" and not _same_origin(request):
        return _json({"error": "cross-origin request refused"}, status=403)
    return await handler(request)


async def _security_headers(request: web.Request, response: web.StreamResponse) -> None:
    h = response.headers
    h.setdefault("X-Content-Type-Options", "nosniff")
    h.setdefault("X-Frame-Options", "DENY")
    h.setdefault("Referrer-Policy", "no-referrer")
    h.setdefault("Content-Security-Policy", CSP)
    h.setdefault("Cache-Control", "no-cache" if request.path.startswith("/static/") else "no-store")


# ---------------------------------------------------------------- pages

LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>Sign in · Heckler</title>
<link rel="icon" href="/static/icon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/static/app.css">
<script src="/static/theme.js"></script>
</head><body class="login-body">
<form class="login card" method="post" action="/login">
  <div class="brand">
    <svg class="mark" width="34" height="34" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" class="mark-bg"/>
      <path class="mark-fg" d="M9 9.5h14a2.5 2.5 0 0 1 2.5 2.5v7a2.5 2.5 0 0 1-2.5 2.5h-7.2L11 25v-3.5H9A2.5 2.5 0 0 1 6.5 19v-7A2.5 2.5 0 0 1 9 9.5z"/>
      <path class="mark-bang" d="M16 12.6v4" stroke-width="2.4" stroke-linecap="round"/><circle class="mark-dot" cx="16" cy="19.2" r="1.35"/></svg>
    <span class="wordmark">Heckler</span>
  </div>
  <h1>Sign in to the dashboard</h1>
  <p class="muted">Paste the admin token, or open the login link the bot prints when it starts.</p>
  {error}
  <label class="visually-hidden" for="token">Admin token</label>
  <input type="password" id="token" name="token" placeholder="Admin token" autocomplete="current-password" autofocus required>
  <button class="btn primary" type="submit">Sign in</button>
</form>
</body></html>
"""


def _login_page(error: str = "") -> web.Response:
    error_html = f'<p class="login-error">{html.escape(error)}</p>' if error else ""
    return web.Response(text=LOGIN_PAGE.replace("{error}", error_html), content_type="text/html")


async def login_get(request: web.Request) -> web.Response:
    if _authed(request):
        raise web.HTTPFound("/")
    return _login_page("That token didn’t work. Check it and try again." if request.query.get("bad") else "")


async def login_post(request: web.Request) -> web.Response:
    if not _same_origin(request):
        return _login_page("Cross-origin login refused.")
    form = await request.post()
    if not _token_ok(str(form.get("token", "")).strip(), request.app["token"]):
        await asyncio.sleep(0.5)  # slows guessing a little
        return _login_page("That token didn’t work. Check it and try again.")
    response = web.HTTPFound("/")
    _set_session(response, request)
    raise response


async def logout(request: web.Request) -> web.Response:
    response = web.HTTPFound("/login")
    response.del_cookie(COOKIE, path="/")
    raise response


async def favicon(request: web.Request) -> web.Response:
    raise web.HTTPFound("/static/icon.svg")


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------- API

async def api_status(request: web.Request) -> web.Response:
    return _json(request.app["controller"].status())


async def api_control(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _json({"error": "body must be JSON"}, status=400)
    if not isinstance(body, dict) or not isinstance(body.get("action"), str):
        return _json({"error": 'expected {"action": "...", ...params}'}, status=400)
    params = dict(body)
    action = params.pop("action")
    controller = request.app["controller"]
    try:
        message = await controller.control(action, **params)
    except ValueError as e:
        return _json({"error": str(e) or "bad request"}, status=400)
    except Exception as e:
        log.exception("Dashboard control %r failed", action)
        return _json({"error": f"{type(e).__name__}: {e}"}, status=500)
    try:
        status = controller.status()
    except Exception:
        status = None
    return _json({"ok": True, "message": message or "", "status": status})


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    if not _same_origin(request):
        raise web.HTTPForbidden(text="cross-origin WebSocket refused")
    app = request.app
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    bus = app["bus"]
    # No await between these two lines: the snapshot and the queue line up exactly.
    recent = bus.recent()
    queue = bus.subscribe()
    app["websockets"].add(ws)
    app["ws_queues"].add(queue)  # also gets dashboard-only messages (content_changed)
    pump = None
    try:
        await ws.send_str(_dumps({"type": "hello", "recent": recent}))
        await _send_status(ws, app)
        pump = asyncio.create_task(_pump(ws, queue, app))
        async for msg in ws:  # we only read to notice the close
            if msg.type == WSMsgType.ERROR:
                break
    finally:
        bus.unsubscribe(queue)
        app["websockets"].discard(ws)
        app["ws_queues"].discard(queue)
        if pump:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
    return ws


async def _send_status(ws: web.WebSocketResponse, app: web.Application) -> None:
    try:
        status = app["controller"].status()
    except Exception:
        log.exception("Dashboard status() failed")
        return
    await ws.send_str(_dumps({"type": "status", **status}))


async def _pump(ws: web.WebSocketResponse, queue: asyncio.Queue, app: web.Application) -> None:
    """Forward bus events, and push the status every few seconds."""
    loop = asyncio.get_running_loop()
    interval = app["status_interval"]
    next_status = loop.time() + interval
    try:
        while not ws.closed:
            try:
                async with asyncio.timeout_at(next_status):
                    event = await queue.get()
            except TimeoutError:
                await _send_status(ws, app)
                next_status = loop.time() + interval
                continue
            await ws.send_str(_dumps(event))
    except (ConnectionError, RuntimeError):
        pass  # the tab went away mid-send


async def _close_websockets(app: web.Application) -> None:
    for ws in list(app["websockets"]):
        await ws.close(code=1001, message=b"server shutdown")


# ---------------------------------------------------------------- app

def create_app(controller: Controller, bus, *, token: str, store=None, engine=None, library=None,
               sounds=None, status_interval: float = STATUS_INTERVAL_S,
               base_dir: str | os.PathLike | None = None) -> web.Application:
    """base_dir: where relative media paths live and data/tmp, data/media go
    (default: the repo root, like packs.py). Tests point it at a tmp dir."""
    if not token:
        raise ValueError("the dashboard needs a token (see load_or_create_token)")
    app = web.Application(middlewares=[auth_middleware], client_max_size=MAX_BODY)
    app["controller"] = controller
    app["bus"] = bus
    app["store"] = store
    app["token"] = token
    app["session"] = _session_value(token)
    app["status_interval"] = status_interval
    app["websockets"] = set()
    app["ws_queues"] = set()
    app.on_response_prepare.append(_security_headers)
    app.on_shutdown.append(_close_websockets)

    r = app.router
    r.add_get("/", index)
    r.add_get("/login", login_get)
    r.add_post("/login", login_post)
    r.add_post("/logout", logout)
    r.add_get("/api/status", api_status)
    r.add_post("/api/control", api_control)
    r.add_get("/ws", ws_handler)
    setup_content_routes(app, store, engine, base_dir=base_dir)  # editors; nothing without a store
    setup_media_routes(app, store, library, sounds)  # voices / sounds; only what the bot passes
    setup_stats_routes(app, store)  # stats and history from the events table
    r.add_get("/favicon.ico", favicon)
    r.add_static("/static/", STATIC_DIR, follow_symlinks=False)
    return app


async def start_dashboard(controller: Controller, bus, *, host: str, port: int, token: str,
                          store=None, engine=None, library=None, sounds=None, base_dir=None) -> web.AppRunner:
    """Serve the dashboard on the running loop. Stop it with `await runner.cleanup()`.

    The access log is off on purpose: `?token=` login links would end up in it."""
    app = create_app(controller, bus, token=token, store=store, engine=engine, library=library, sounds=sounds,
                     base_dir=base_dir)
    runner = web.AppRunner(app, access_log=None, handle_signals=False)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("Dashboard on http://%s:%s/", host, port)
    return runner


# ---------------------------------------------------------------- logging

class BusLogHandler(logging.Handler):
    """Publishes log records as {"type": "log", ...} events, from any thread.

        handler = BusLogHandler(bus, asyncio.get_running_loop())
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, bus, loop: asyncio.AbstractEventLoop, level: int = logging.WARNING) -> None:
        super().__init__(level)
        self.bus = bus
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info and record.exc_info[1] is not None:
                exc = record.exc_info[1]
                message += f" ({''.join(traceback.format_exception_only(exc)).strip()})"
            event = {
                "type": "log",
                "time": datetime.fromtimestamp(record.created).isoformat(timespec="seconds"),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
            if self.loop.is_closed():
                return
            self.loop.call_soon_threadsafe(self.bus.publish, event)
        except RuntimeError:
            pass  # loop shutting down
        except Exception:
            self.handleError(record)
