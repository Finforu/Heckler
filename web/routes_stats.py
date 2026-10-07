"""Stats and history, from the store's `events` table (one row per
transcribed sentence: outcome, who, reaction, and the text only when that
person agreed to have their words kept; otherwise it is null).

    GET    /api/g/{gid}/stats?range=24h|7d|30d|90d    aggregates for the Stats tab
    GET    /api/g/{gid}/history?before_id=&limit=&user_id=&outcome=&q=
                                                   newest first, paged with before_id
    DELETE /api/g/{gid}/history/{uid}              forget one person's history here

Speed: event ids only grow, and so do their times, and (guild_id, id) is
indexed. So a time range becomes an id range: a binary search finds the first
event inside it with ~20 indexed lookups, and every aggregate then reads just
that window (guild_id = ? AND id >= ?) instead of the whole history.

Read-only SQL goes through store._query (the store's own connection, under its
lock). Deleting history uses store.delete_events(guild_id, user_id) when the
store has it, else one DELETE through store._write.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from aiohttp import web

from .routes_content import FieldError, _gid, _int, _names, api, to_user_id
from .util import json_response

RANGES = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30), "90d": timedelta(days=90)}
WORD_TRIGGERS = ("phrase", "swap", "command")  # what a sentence can set off
TOP = 15
HISTORY_MAX = 200


def setup_stats_routes(app: web.Application, store) -> None:
    if store is None:
        return
    r = app.router
    g = "/api/g/{gid:\\d+}"
    r.add_get(f"{g}/stats", stats)
    r.add_get(f"{g}/history", history)
    r.add_delete(f"{g}/history/{{uid:\\d+}}", history_delete)


# ───────────────────────────── helpers ─────────────────────────────

def _parse(stamp: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.astimezone()


def first_id_since(store, guild_id: int, since: datetime) -> int | None:
    """The id of the server's first event at or after `since` (None: none).
    Binary search on ids; each step is one indexed lookup."""
    q = store._query
    bounds = q("SELECT MIN(id), MAX(id) FROM events WHERE guild_id = ?", (guild_id,))[0]
    lo, hi = bounds[0], bounds[1]
    if lo is None:
        return None

    def first_at_or_after(x: int):
        rows = q("SELECT id, time FROM events WHERE guild_id = ? AND id >= ? ORDER BY id LIMIT 1", (guild_id, x))
        return rows[0] if rows else None

    def inside(row) -> bool:
        t = _parse(row["time"])
        return t is not None and t >= since

    if not inside(first_at_or_after(hi)):
        return None  # even the newest is older
    # Smallest x in [lo, hi] whose next event is inside the range (true at hi).
    while lo < hi:
        mid = (lo + hi) // 2
        row = first_at_or_after(mid)
        if inside(row):
            hi = mid
        else:
            lo = row["id"] + 1  # every x up to row's id leads to that same (too old) event
    return first_at_or_after(lo)["id"]


def _reaction_map(store, guild_id: int) -> dict[int, dict]:
    return {r["id"]: r for r in store.list_reactions(guild_id)}


def _creator(row: dict, names: dict) -> str:
    return "admin" if row.get("created_by") is None else names.get(row["created_by"], str(row["created_by"]))


def _person_names(store, guild_id: int, since_id: int | None = None) -> dict:
    """user_id -> what to call them: nickname, else display name, else the
    newest name the events saw."""
    names = _names(store, guild_id)
    sql = "SELECT user_id, user_name, MAX(id) FROM events WHERE guild_id = ? AND user_id IS NOT NULL"
    params: list = [guild_id]
    if since_id is not None:
        sql += " AND id >= ?"
        params.append(since_id)
    # SQLite: with MAX(), the other columns come from that (newest) row.
    for row in store._query(sql + " GROUP BY user_id", params):
        names.setdefault(row["user_id"], row["user_name"] or str(row["user_id"]))
    return names


# ───────────────────────────── stats ─────────────────────────────

@api
async def stats(request: web.Request) -> web.Response:
    store, gid = request.app["store"], _gid(request)
    key = request.query.get("range", "7d")
    if key not in RANGES:
        raise FieldError(f"range must be one of {', '.join(RANGES)}", "range")
    now = datetime.now().astimezone()
    since = now - RANGES[key]
    start = first_id_since(store, gid, since)
    q = store._query
    reactions = _reaction_map(store, gid)
    names = _names(store, gid)

    empty = start is None
    window, params = "guild_id = ? AND id >= ?", (gid, start if start is not None else 0)

    outcomes = {} if empty else {r[0]: r[1] for r in q(
        f"SELECT outcome, COUNT(*) FROM events WHERE {window} GROUP BY outcome", params)}
    heard = sum(outcomes.values())

    hours = [{"hour": h, "heard": 0, "matched": 0} for h in range(24)]
    if not empty:
        for row in q(f"SELECT substr(time, 12, 2), COUNT(*), SUM(outcome = 'matched') FROM events "
                     f"WHERE {window} GROUP BY 1", params):
            if row[0] and row[0].isdigit() and 0 <= int(row[0]) < 24:
                hours[int(row[0])].update(heard=row[1], matched=row[2] or 0)

    # (reaction, outcome) counts: fires, and why the rest were skipped
    per_reaction: dict[int, dict[str, int]] = {}
    if not empty:
        for row in q(f"SELECT reaction_id, outcome, COUNT(*) FROM events WHERE {window} "
                     f"AND reaction_id IS NOT NULL GROUP BY reaction_id, outcome", params):
            per_reaction.setdefault(row[0], {})[row[1]] = row[2]

    def reaction_info(rid: int) -> dict:
        row = reactions.get(rid)
        if row is None:
            return {"reaction_id": rid, "name": f"#{rid} (deleted)", "kind": None, "deleted": True}
        return {"reaction_id": rid, "name": row["name"], "kind": row["kind"], "uses": row["uses"],
                "last_used_at": row["last_used_at"], "creator": _creator(row, names), "enabled": row["enabled"]}

    top = sorted(((rid, c.get("matched", 0)) for rid, c in per_reaction.items() if c.get("matched")),
                 key=lambda x: -x[1])[:TOP]
    top_reactions = [{**reaction_info(rid), "fires": n} for rid, n in top]

    skipped = []
    for rid, counts in per_reaction.items():
        skips = {o: n for o, n in counts.items() if o.startswith("skipped")}
        if not skips:
            continue
        total = counts.get("matched", 0) + sum(skips.values())
        skipped.append({**reaction_info(rid), "fires": counts.get("matched", 0), "skips": skips,
                        "skip_share": round(sum(skips.values()) / total, 3) if total else 0})
    skipped.sort(key=lambda x: (-sum(x["skips"].values()), x["name"]))

    never = []
    for row in reactions.values():
        if not row["enabled"] or row["status"] != "approved":
            continue
        if not any(t["type"] in WORD_TRIGGERS for t in row["triggers"]):
            continue  # set off by events (greetings, timers): no sentences to count
        if per_reaction.get(row["id"], {}).get("matched"):
            continue
        never.append({"reaction_id": row["id"], "name": row["name"], "kind": row["kind"], "uses": row["uses"],
                      "last_used_at": row["last_used_at"], "created_at": row["created_at"],
                      "creator": _creator(row, names)})
    never.sort(key=lambda x: (x["uses"] or 0, x["last_used_at"] or "", x["name"]))

    people = []
    if not empty:
        names = _person_names(store, gid, start)
        totals = {r[0]: r[1] for r in q(f"SELECT user_id, COUNT(*) FROM events WHERE {window} "
                                         f"AND user_id IS NOT NULL GROUP BY user_id", params)}
        fired: dict[int, dict[int, int]] = {}
        for row in q(f"SELECT user_id, reaction_id, COUNT(*) FROM events WHERE {window} AND outcome = 'matched' "
                     f"AND user_id IS NOT NULL AND reaction_id IS NOT NULL GROUP BY user_id, reaction_id", params):
            fired.setdefault(row[0], {})[row[1]] = row[2]
        for uid, count in totals.items():
            mine = fired.get(uid, {})
            gags = {rid: n for rid, n in mine.items() if reactions.get(rid, {}).get("kind") == "gag"}
            best = max(gags.items(), key=lambda x: x[1]) if gags else None
            people.append({
                "user_id": uid, "name": names.get(uid, str(uid)), "heard": count,
                "matched": sum(mine.values()), "gags": sum(gags.values()),
                "top_gag": {"name": reactions[best[0]]["name"], "count": best[1]} if best else None,
            })
        people.sort(key=lambda x: (-x["heard"], x["name"]))

    return json_response({
        "range": key, "since": since.isoformat(timespec="seconds"), "now": now.isoformat(timespec="seconds"),
        "totals": {
            "heard": heard, "matched": outcomes.get("matched", 0), "no_match": outcomes.get("no match", 0),
            "skipped": {o: n for o, n in outcomes.items() if o.startswith("skipped")},
            "echo": sum(n for o, n in outcomes.items() if "echo" in o),
            "outcomes": outcomes,
        },
        "hours": hours,
        "top_reactions": top_reactions,
        "skipped_reactions": skipped[:TOP],
        "never_fired": never,
        "people": people,
    })


# ───────────────────────────── history ─────────────────────────────

def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


@api
async def history(request: web.Request) -> web.Response:
    store, gid = request.app["store"], _gid(request)
    query = request.query
    try:
        limit = max(1, min(int(query.get("limit", 50)), HISTORY_MAX))
    except ValueError:
        raise FieldError("limit must be a number", "limit") from None
    sql, params = "SELECT * FROM events WHERE guild_id = ?", [gid]
    if query.get("before_id"):
        before = to_user_id(query["before_id"])
        if before is None:
            raise FieldError("before_id must be an event id", "before_id")
        sql, params = sql + " AND id < ?", [*params, before]
    if query.get("user_id"):
        uid = to_user_id(query["user_id"])
        if uid is None:
            raise FieldError("user_id must be a user id", "user_id")
        sql, params = sql + " AND user_id = ?", [*params, uid]
    if query.get("outcome"):
        sql, params = sql + " AND outcome = ?", [*params, query["outcome"]]
    if query.get("q", "").strip():
        sql, params = sql + " AND text LIKE ? ESCAPE '\\'", [*params, _like(query["q"].strip())]
    rows = [store._decode(r) for r in store._query(sql + " ORDER BY id DESC LIMIT ?", [*params, limit])]

    reactions = _reaction_map(store, gid)
    names = _names(store, gid)
    events = []
    for e in rows:
        details = e.get("details") or {}
        reaction = reactions.get(e["reaction_id"]) if e["reaction_id"] is not None else None
        events.append({
            "id": e["id"], "time": e["time"], "user_id": e["user_id"],
            "user": names.get(e["user_id"]) or e["user_name"] or (str(e["user_id"]) if e["user_id"] else "?"),
            "text": e["text"], "outcome": e["outcome"], "reaction_id": e["reaction_id"],
            "reaction": reaction["name"] if reaction else (details.get("matched") or {}).get("name"),
            "kind": reaction["kind"] if reaction else (details.get("matched") or {}).get("kind"),
            "lang": details.get("lang"), "duration": details.get("duration"),
        })
    out = {"events": events, "next_before_id": rows[-1]["id"] if len(rows) == limit else None}
    if not query.get("before_id"):  # first page: what the filters can offer
        people = _person_names(store, gid)
        counts = store._query("SELECT user_id, COUNT(*) FROM events WHERE guild_id = ? AND user_id IS NOT NULL "
                              "GROUP BY user_id", (gid,))
        out["people"] = sorted(({"user_id": r[0], "name": people.get(r[0], str(r[0])), "count": r[1]} for r in counts),
                               key=lambda p: p["name"].casefold())
        out["outcomes"] = [r[0] for r in store._query(
            "SELECT DISTINCT outcome FROM events WHERE guild_id = ? ORDER BY outcome", (gid,))]
    return json_response(out)


@api
async def history_delete(request: web.Request) -> web.Response:
    store, gid, uid = request.app["store"], _gid(request), _int(request, "uid")
    if hasattr(store, "delete_events"):
        count = store.delete_events(gid, uid)
    else:
        count = store._write("DELETE FROM events WHERE guild_id = ? AND user_id = ?", (gid, uid)).rowcount
    return json_response({"deleted": count})
