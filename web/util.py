"""JSON helpers shared by the dashboard routes."""
from __future__ import annotations

import json
from typing import Any

from aiohttp import web

MAX_SAFE_INT = 2**53 - 1  # larger ints lose precision in a JS number


def jsonable(value: Any, key: str | None = None) -> Any:
    """Copy of `value` safe for the browser: ints under a `*_id` key and any
    int too big for a JS number (Discord snowflakes) become strings."""
    if isinstance(value, dict):
        return {k: jsonable(v, k if isinstance(k, str) else None) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > MAX_SAFE_INT or (key is not None and key.endswith("_id")):
            return str(value)
    return value


def dumps(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, default=str)


def json_response(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, dumps=dumps)


def error(message: str, status: int = 400, **extra) -> web.Response:
    return json_response({"error": message, **extra}, status=status)
