"""Content editors: reactions, approvals, quotas, people, settings, voices and
sounds (read-only for now), the test bench and packs.

Everything lives under /api/g/{guild_id}/ and goes through the same auth and
Origin checks as the rest of /api (see server.py). The store is synchronous,
thread-safe and fast, so handlers call it on the loop; pack import/export is
blocking file work and runs in a thread.

IDs: everything under a `*_id` key, and any int too big for JavaScript, goes
out as a string (see util.jsonable). Requests may send ids as strings or ints;
they are stored as ints.

Routes:
    GET    /api/g/{gid}/content                       everything the editors show, in one go
    POST   /api/g/{gid}/reactions                     create             -> {"id", "reaction"}
    PATCH  /api/g/{gid}/reactions/{id}                update some fields -> {"reaction"}
    DELETE /api/g/{gid}/reactions/{id}
    POST   /api/g/{gid}/reactions/order               {"ids": [...]} (match order)
    POST   /api/g/{gid}/reactions/{id}/phrase         {"trigger": index, "phrase"}  (test bench: near miss)
    PATCH  /api/g/{gid}/people/{uid}                  {"nickname"}
    POST   /api/g/{gid}/people/{uid}/revoke-consent
    PUT    /api/g/{gid}/settings/{key}                {"value"}  (gid 0: global default)
    DELETE /api/g/{gid}/settings/{key}                back to inherited
    PUT    /api/g/{gid}/quota/{uid}/{resource}        {"limit": int | null}
    POST   /api/g/{gid}/quota-requests/{id}/decide    {"approve": bool, "amount"?: int}
    POST   /api/g/{gid}/test                          {"text", "user_id"?} -> engine.explain(...)
    GET    /api/g/{gid}/pack/export[?personal=0]      a .zip: pack.yaml + its audio files
    POST   /api/g/{gid}/pack/import                   multipart: mode (merge|replace), file (.yaml or .zip)

Each write fires store.on_change; the dashboard turns that into a
{"type": "content_changed", "guild_id", "tables"} WebSocket message so other
open tabs refresh.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import shutil
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any

from aiohttp import web

from .util import error, json_response

log = logging.getLogger(__name__)

MAX_UPLOAD = 200 * 1024 * 1024        # pack upload, bytes
MAX_UNZIPPED = 1024 * 1024 * 1024     # total size of a pack zip's contents
MAX_ZIP_ENTRIES = 5000
NOTIFY_DELAY_S = 0.25                 # coalesce bursts of store changes
# Settings that only exist globally (guild 0); see the comments in store.DEFAULT_SETTINGS.
GLOBAL_ONLY_SETTINGS = ("voice.bot", "cache.tts_mb")
REACTION_FIELDS = ("kind", "name", "enabled", "status", "triggers", "options", "voice_id",
                   "by_users", "cooldown_s", "chance")


class FieldError(ValueError):
    """A ValueError that knows which part of the form is wrong ("triggers.1", "options.0", "name")."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


# ───────────────────────────── setup ─────────────────────────────

def setup_content_routes(app: web.Application, store, engine=None, *, base_dir=None) -> None:
    """Register the editor routes on `app`. No store: nothing (the page then
    shows the status view only)."""
    if store is None:
        return
    import packs  # the repo's packs.py; imported here so Phase 1 runs without PyYAML

    app["store"] = store
    app["engine"] = engine
    app["base_dir"] = Path(base_dir) if base_dir else Path(packs.ROOT)
    app.on_startup.append(_listen)
    app.on_cleanup.append(_unlisten)

    r = app.router
    g = "/api/g/{gid:\\d+}"
    r.add_get("/api/languages", languages)
    r.add_get(f"{g}/content", content)
    r.add_post(f"{g}/reactions", reaction_create)
    r.add_post(f"{g}/reactions/order", reaction_order)
    r.add_patch(f"{g}/reactions/{{rid:\\d+}}", reaction_update)
    r.add_delete(f"{g}/reactions/{{rid:\\d+}}", reaction_delete)
    r.add_post(f"{g}/reactions/{{rid:\\d+}}/phrase", reaction_add_phrase)
    r.add_patch(f"{g}/people/{{uid:\\d+}}", person_update)
    r.add_post(f"{g}/people/{{uid:\\d+}}/revoke-consent", consent_revoke)
    r.add_post(f"{g}/people/{{uid:\\d+}}/delete-text", person_delete_text)
    r.add_put(f"{g}/settings/{{key}}", setting_put)
    r.add_delete(f"{g}/settings/{{key}}", setting_delete)
    r.add_put(f"{g}/quota/{{uid:\\d+}}/{{resource}}", quota_put)
    r.add_post(f"{g}/quota-requests/{{qid:\\d+}}/decide", quota_decide)
    r.add_post(f"{g}/test", test_bench)
    r.add_get(f"{g}/pack/export", pack_export)
    r.add_post(f"{g}/pack/import", pack_import)


async def _listen(app: web.Application) -> None:
    """Store changes (from any thread) -> one content_changed per server, debounced."""
    loop = asyncio.get_running_loop()
    pending: dict[Any, set[str]] = {}

    def flush() -> None:
        for guild_id, tables in pending.items():
            message = {"type": "content_changed", "guild_id": guild_id, "tables": sorted(tables)}
            for queue in list(app["ws_queues"]):
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    pass
        pending.clear()

    def on_loop(table: str, guild_id) -> None:
        if not pending:
            loop.call_later(NOTIFY_DELAY_S, flush)
        pending.setdefault(guild_id, set()).add(table)

    def listener(table: str, guild_id) -> None:
        if not loop.is_closed():
            try:
                loop.call_soon_threadsafe(on_loop, table, guild_id)
            except RuntimeError:
                pass  # loop shutting down

    app["store_listener"] = listener
    app["store"].on_change(listener)


async def _unlisten(app: web.Application) -> None:
    listener = app.get("store_listener")
    if listener:
        app["store"].off_change(listener)


def api(handler):
    """Turn ValueError into 400 (with the field, if known) and KeyError into 404."""
    @wraps(handler)
    async def wrapped(request: web.Request) -> web.StreamResponse:
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except KeyError as e:
            return error(str(e.args[0]) if e.args else "not found", 404)
        except ValueError as e:
            extra = {"field": e.field} if isinstance(e, FieldError) and e.field else {}
            return error(str(e) or "bad request", 400, **extra)
    return wrapped


# ───────────────────────────── helpers ─────────────────────────────

def _gid(request: web.Request) -> int:
    return int(request.match_info["gid"])


def _int(request: web.Request, key: str) -> int:
    return int(request.match_info[key])


async def _body(request: web.Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        raise ValueError("body must be a JSON object") from None
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    return body


def to_user_id(value) -> int | None:
    """A Discord user id from JSON (string or int); None if it isn't one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _voice_ref(value, store, guild_id: int, where: str):
    """null / "" -> None, "@speaker" stays, a voice id (string or int) -> int (must exist here)."""
    import reactions as rules

    if value is None or value == "":
        return None
    if value == rules.SPEAKER:
        return value
    voice_id = to_user_id(value)
    if voice_id is None:
        raise FieldError(f"unknown voice {value!r}", where)
    voice = store.get_voice(voice_id)
    if voice is None or voice["guild_id"] not in (None, guild_id):
        raise FieldError(f"unknown voice {value!r}", where)
    return voice_id


def _str_list(value, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise FieldError("expected a list of words", where)
    return [str(v).strip() for v in value if str(v).strip()]


def clean_reaction(body: dict, store, guild_id: int, *, partial: bool) -> dict:
    """The writable reaction fields from a request body, ids converted and
    checked. Raises FieldError pointing at the bad part."""
    import reactions as rules
    from store import REACTION_KINDS, REACTION_STATUSES

    unknown = set(body) - set(REACTION_FIELDS) - {"id", "guild_id", "created_by", "position", "uses",
                                                    "last_used_at", "created_at", "updated_at", "summary",
                                                    "creator"}
    if unknown:
        raise FieldError(f"unknown fields: {sorted(unknown)}")
    fields = {k: body[k] for k in REACTION_FIELDS if k in body}
    if not partial:
        for key in ("kind", "name", "triggers", "options"):
            if key not in fields:
                raise FieldError(f"{key} is required", key)

    if "name" in fields:
        fields["name"] = str(fields["name"] or "").strip()
        if not fields["name"]:
            raise FieldError("the reaction needs a name", "name")
    if "kind" in fields and fields["kind"] not in REACTION_KINDS:
        raise FieldError(f"kind must be one of {REACTION_KINDS}", "kind")
    if "status" in fields and fields["status"] not in REACTION_STATUSES:
        raise FieldError(f"status must be one of {REACTION_STATUSES}", "status")
    if "enabled" in fields:
        fields["enabled"] = bool(fields["enabled"])
    if "chance" in fields:
        try:
            fields["chance"] = float(fields["chance"])
        except (TypeError, ValueError):
            raise FieldError("chance must be a number between 0 and 1", "chance") from None
        if not 0 <= fields["chance"] <= 1:
            raise FieldError("chance must be between 0 and 1", "chance")
    if "cooldown_s" in fields:
        value = fields["cooldown_s"]
        if value in (None, ""):
            fields["cooldown_s"] = None
        else:
            try:
                fields["cooldown_s"] = float(value)
            except (TypeError, ValueError):
                raise FieldError("cooldown must be a number of seconds", "cooldown_s") from None
            if fields["cooldown_s"] < 0:
                raise FieldError("cooldown can't be negative", "cooldown_s")
    if "voice_id" in fields:
        fields["voice_id"] = _voice_ref(fields["voice_id"], store, guild_id, "voice_id")
    if "by_users" in fields:
        people = fields["by_users"]
        if people in (None, "", []):
            fields["by_users"] = None
        else:
            if not isinstance(people, list):
                raise FieldError("by_users must be a list of user ids or names", "by_users")
            out = []
            for p in people:
                uid = to_user_id(p)
                if uid is not None:
                    out.append(uid)
                elif str(p).strip():
                    out.append(str(p).strip())
            fields["by_users"] = out or None

    if "triggers" in fields:
        triggers = fields["triggers"]
        if not isinstance(triggers, list) or not triggers:
            raise FieldError("add at least one trigger", "triggers")
        cleaned = []
        for i, t in enumerate(triggers):
            where = f"triggers.{i}"
            if not isinstance(t, dict):
                raise FieldError("bad trigger", where)
            kind = t.get("type")
            if kind in ("phrase", "command"):
                t = {"type": kind, "phrases": _str_list(t.get("phrases"), where)}
            elif kind == "swap":
                t = {"type": "swap", "word": str(t.get("word") or "").strip(), "to": str(t.get("to") or "").strip(),
                     "also": _str_list(t.get("also"), where),
                     "connectors": _str_list(t.get("connectors"), where) or list(rules.DEFAULT_CONNECTORS)}
            elif kind == "slash":
                t = {"type": "slash", "name": str(t.get("name") or "").strip().lstrip("/")}
            elif kind == "event":
                out = {"type": "event", "event": t.get("event")}
                if t.get("user_id") not in (None, ""):
                    uid = to_user_id(t["user_id"])
                    if uid is None:
                        raise FieldError("the person must be a user id", where)
                    out["user_id"] = uid
                t = out
            _check(lambda: rules.validate([t], [[{"type": "say", "text": "x"}]]), where)
            cleaned.append(t)
        fields["triggers"] = cleaned

    if "options" in fields:
        options = fields["options"]
        if not isinstance(options, list) or not options:
            raise FieldError("add at least one option", "options")
        sounds = {s["id"] for s in store.list_sounds(guild_id)}
        cleaned = []
        for i, option in enumerate(options):
            where = f"options.{i}"
            if not isinstance(option, list) or not option:
                raise FieldError("each option needs at least one step", where)
            steps = []
            for j, step in enumerate(option):
                at = f"{where}.{j}"
                kind = step.get("type") if isinstance(step, dict) else None
                if kind == "say":
                    step = {"type": "say", "text": str(step.get("text") or "")}
                    voice = _voice_ref(option[j].get("voice_id"), store, guild_id, at)
                    if voice is not None:
                        step["voice_id"] = voice
                elif kind == "sound":
                    sound_id = to_user_id(step.get("sound_id"))
                    if sound_id not in sounds:
                        raise FieldError("pick a sound", at)
                    step = {"type": "sound", "sound_id": sound_id}
                elif kind == "builtin":
                    step = {"type": "builtin", "action": step.get("action")}
                steps.append(step)
            _check(lambda: rules.validate([{"type": "phrase", "phrases": ["x"]}], [steps]), where)
            cleaned.append(steps)
        fields["options"] = cleaned
    return fields


def _check(fn, where: str) -> None:
    try:
        fn()
    except ValueError as e:
        raise FieldError(str(e), where) from None


def _names(store, guild_id: int) -> dict[int, str]:
    return {p["user_id"]: p["nickname"] or p["display_name"] or str(p["user_id"]) for p in store.list_people(guild_id)}


def _reaction_out(row: dict, names: dict[int, str]) -> dict:
    import reactions as rules

    try:
        summary = rules.describe(row)
    except Exception:
        summary = ""
    creator = "admin" if row.get("created_by") is None else names.get(row["created_by"], str(row["created_by"]))
    return {**row, "summary": summary, "creator": creator}


def _own_reaction(store, guild_id: int, reaction_id: int) -> dict:
    row = store.get_reaction(reaction_id)
    if row is None or row["guild_id"] != guild_id:
        raise KeyError(f"no reaction {reaction_id} in this server")
    return row


# ───────────────────────────── read ─────────────────────────────

@api
async def content(request: web.Request) -> web.Response:
    import reactions as rules
    from store import (CONSENT_STATUSES, DEFAULT_SETTINGS, GLOBAL, QUOTA_RESOURCES, REACTION_KINDS,
                       REACTION_STATUSES)

    store, gid = request.app["store"], _gid(request)
    people = store.list_people(gid)
    names = _names(store, gid)
    reactions = store.list_reactions(gid)
    sounds = store.list_sounds(gid)
    voices = store.list_voices(gid)
    requests = store.list_quota_requests(gid)
    overrides = store.list_quota_overrides(gid)

    # Everyone quotas are worth showing for: known people, and anyone with an
    # override, a request or something they created here.
    users = {p["user_id"] for p in people}
    users |= {o["user_id"] for o in overrides} | {q["user_id"] for q in requests}
    users |= {x["created_by"] for x in (*reactions, *sounds, *voices)
              if isinstance(x.get("created_by"), int)}
    override_map = {(o["user_id"], o["resource"]): o["limit"] for o in overrides}
    quota_users = []
    for uid in sorted(users, key=lambda u: names.get(u, str(u)).casefold()):
        row = {"user_id": uid, "name": names.get(uid, str(uid))}
        for res in QUOTA_RESOURCES:
            used, limit = store.quota(gid, uid, res)
            row[res] = {"used": used, "limit": limit, "override": override_map.get((uid, res))}
        quota_users.append(row)

    engine = request.app.get("engine")
    return json_response({
        "guild_id": gid,
        "meta": {
            "kinds": REACTION_KINDS, "statuses": REACTION_STATUSES,
            "trigger_types": rules.TRIGGER_TYPES, "events": rules.EVENTS, "step_types": rules.STEP_TYPES,
            "builtins": rules.BUILTINS, "placeholders": rules.PLACEHOLDERS, "speaker": rules.SPEAKER,
            "default_connectors": rules.DEFAULT_CONNECTORS, "quota_resources": QUOTA_RESOURCES,
            "consent_statuses": CONSENT_STATUSES, "global_only_settings": GLOBAL_ONLY_SETTINGS,
            "consent_purposes": _purposes(), "languages": _languages()[0], "default_language": _languages()[1],
            "starter_packs": starter_packs(), "controller_settings": list(CONTROLLER_SETTINGS),
            "test_bench": engine is not None and hasattr(engine, "explain"),
            "media": {"voices": request.app.get("library") is not None,
                      "sounds": request.app.get("sounds") is not None},
        },
        "reactions": [_reaction_out(r, names) for r in reactions],
        "voices": voices,
        "sounds": sounds,
        "people": [{**p, "consent": {purpose: _consent(store, p["user_id"], purpose) for purpose in _purposes()}}
                   for p in people],
        "settings": {
            "defaults": DEFAULT_SETTINGS,
            "global": store.settings(GLOBAL, effective=False),
            "own": store.settings(gid, effective=False) if gid != GLOBAL else {},
            "effective": store.settings(gid),
        },
        "quotas": {"defaults": {res: store.get_setting(gid, f"quota.{res}", 0) for res in QUOTA_RESOURCES},
                   "users": quota_users},
        "requests": [{**q, "name": names.get(q["user_id"], str(q["user_id"])),
                      "used": store.quota(gid, q["user_id"], q["resource"])[0],
                      "limit": store.quota_limit(gid, q["user_id"], q["resource"])} for q in requests],
    })


# ───────────────────────────── reactions ─────────────────────────────

@api
async def reaction_create(request: web.Request) -> web.Response:
    store, gid = request.app["store"], _gid(request)
    fields = clean_reaction(await _body(request), store, gid, partial=False)
    kind, name = fields.pop("kind"), fields.pop("name")
    triggers, options = fields.pop("triggers"), fields.pop("options")
    reaction_id = store.add_reaction(gid, kind, name, triggers, options, **fields)
    return json_response({"id": reaction_id,
                          "reaction": _reaction_out(store.get_reaction(reaction_id), _names(store, gid))})


@api
async def reaction_update(request: web.Request) -> web.Response:
    store, gid, rid = request.app["store"], _gid(request), _int(request, "rid")
    _own_reaction(store, gid, rid)
    fields = clean_reaction(await _body(request), store, gid, partial=True)
    if fields:
        store.update_reaction(rid, **fields)
    return json_response({"reaction": _reaction_out(store.get_reaction(rid), _names(store, gid))})


@api
async def reaction_delete(request: web.Request) -> web.Response:
    store, gid, rid = request.app["store"], _gid(request), _int(request, "rid")
    _own_reaction(store, gid, rid)
    store.delete_reaction(rid)
    return json_response({"ok": True})


@api
async def reaction_order(request: web.Request) -> web.Response:
    store, gid = request.app["store"], _gid(request)
    ids = (await _body(request)).get("ids")
    if not isinstance(ids, list):
        raise ValueError('expected {"ids": [...]}')
    ids = [to_user_id(i) for i in ids]
    known = {r["id"] for r in store.list_reactions(gid)}
    if any(i not in known for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("ids must be this server's reactions, each once")
    store.reorder_reactions(gid, ids)
    return json_response({"ok": True})


@api
async def reaction_add_phrase(request: web.Request) -> web.Response:
    """Add a heard phrase to one trigger (a near miss from the test bench):
    phrase/command triggers get it in `phrases`, swap triggers in `also`."""
    store, gid, rid = request.app["store"], _gid(request), _int(request, "rid")
    row = _own_reaction(store, gid, rid)
    body = await _body(request)
    phrase = str(body.get("phrase") or "").strip()
    if not phrase:
        raise ValueError("phrase is required")
    index = body.get("trigger")
    triggers = [dict(t) for t in row["triggers"]]
    if isinstance(index, dict):  # the trigger itself, as explain() returned it
        index = next((i for i, t in enumerate(triggers) if t == index or
                      all(str(t.get(k)) == str(v) for k, v in index.items())), None)
    elif index is None:
        index = next((i for i, t in enumerate(triggers) if t["type"] in ("phrase", "command", "swap")), None)
    if not isinstance(index, int) or not 0 <= index < len(triggers):
        raise ValueError("no such trigger in this reaction")
    t = triggers[index]
    key = {"phrase": "phrases", "command": "phrases", "swap": "also"}.get(t["type"])
    if key is None:
        raise ValueError(f"a {t['type']} trigger has no phrases")
    words = list(t.get(key) or [])
    if phrase.casefold() not in (w.casefold() for w in words):
        words.append(phrase)
    t[key] = words
    store.update_reaction(rid, triggers=triggers)
    return json_response({"reaction": _reaction_out(store.get_reaction(rid), _names(store, gid))})


# ───────────────────────────── people ─────────────────────────────

@api
async def person_update(request: web.Request) -> web.Response:
    store, gid, uid = request.app["store"], _gid(request), _int(request, "uid")
    body = await _body(request)
    if "nickname" not in body:
        raise ValueError("expected {\"nickname\": ...}")
    nickname = str(body["nickname"] or "").strip() or None
    store.set_person(gid, uid, nickname=nickname)
    return json_response({"person": store.get_person(gid, uid)})


@api
async def consent_revoke(request: web.Request) -> web.Response:
    """Withdraw someone's consent for one purpose ("voice" / "transcripts"),
    as if they had revoked it themselves."""
    store, uid = request.app["store"], _int(request, "uid")
    body = await _body(request) if request.can_read_body else {}
    purpose = body.get("purpose") or "voice"
    if purpose not in _purposes():
        raise FieldError(f"purpose must be one of {', '.join(_purposes())}", "purpose")
    if len(_purposes()) > 1:
        store.set_consent(uid, "revoked", purpose=purpose)
    else:
        store.set_consent(uid, "revoked")
    return json_response({"purpose": purpose, "status": _consent(store, uid, purpose)})


@api
async def person_delete_text(request: web.Request) -> web.Response:
    """Forget what one person said in this server (their history stays, without the words)."""
    store, gid, uid = request.app["store"], _gid(request), _int(request, "uid")
    if hasattr(store, "delete_user_text"):
        count = store.delete_user_text(uid, guild_id=gid)
    else:
        count = store._write("UPDATE events SET text = NULL WHERE guild_id = ? AND user_id = ? AND text IS NOT NULL",
                             (gid, uid)).rowcount
    return json_response({"cleared": count})


async def languages(request: web.Request) -> web.Response:
    langs, default = _languages()
    return json_response({"languages": langs, "default": default})


# ───────────────────────────── settings ─────────────────────────────

# Settings the dashboard must change through the bot (it has to tell the
# server), never by writing the store: setting key -> controller toggle name.
CONTROLLER_SETTINGS = {"transcripts.enabled": "transcripts"}


def _purposes() -> tuple[str, ...]:
    import store as store_module

    return tuple(getattr(store_module, "CONSENT_PURPOSES", ("voice",)))


def _consent(store, user_id: int, purpose: str) -> str | None:
    try:
        row = store.get_consent(user_id, purpose) if len(_purposes()) > 1 else store.get_consent(user_id)
    except (TypeError, ValueError):
        row = None
    return (row or {}).get("status")


def _languages() -> tuple[list[str], str]:
    try:
        import i18n

        return list(i18n.languages()), i18n.default_language()
    except Exception:
        return ["en"], "en"


def starter_packs() -> list[str]:
    import packs

    folder = Path(packs.ROOT) / "packs"
    return sorted(p.stem for p in folder.glob("*.yaml")) if folder.is_dir() else []


def clean_setting(key: str, value, store, guild_id: int):
    """Check a setting's value against its built-in default's type."""
    from store import DEFAULT_SETTINGS

    if key not in DEFAULT_SETTINGS:
        raise KeyError(f"unknown setting {key!r}")
    if key in GLOBAL_ONLY_SETTINGS and guild_id != 0:
        raise FieldError(f"{key} is a global setting (server 0)", "value")
    if key in CONTROLLER_SETTINGS:
        raise FieldError(f"{key} is changed with the bot's {CONTROLLER_SETTINGS[key]!r} toggle "
                         "(it posts a notice in the server)", "value")
    default = DEFAULT_SETTINGS[key]
    if key == "language":
        if value in (None, ""):
            return None
        if value not in _languages()[0]:
            raise FieldError(f"language must be one of {', '.join(_languages()[0])}", "value")
        return value
    if key == "content.starter_pack":
        if value in (None, ""):
            return None
        if value != "none" and value not in starter_packs():
            raise FieldError(f"no starter pack called {value!r}", "value")
        return value
    if key == "time.zone":
        if value in (None, ""):
            return None
        from zoneinfo import ZoneInfo

        try:
            ZoneInfo(str(value).strip())
        except Exception:
            raise FieldError("time.zone must be a time zone name such as Europe/Madrid or America/Mexico_City",
                             "value") from None
        return str(value).strip()
    if key == "voice.bot":
        if value == "@speaker":
            raise FieldError("the bot's voice must be a saved global voice", "value")
        value = _voice_ref(value, store, 0, "value")
    elif key.endswith("_id"):  # a Discord role / channel id
        if value in (None, ""):
            value = None
        else:
            value = to_user_id(value)
            if value is None:
                raise FieldError(f"{key} must be a Discord id (digits)", "value")
    elif key == "bot.name" or (default is None and not key.startswith("voice.")):
        value = None if value in (None, "") else str(value).strip() or None
    elif isinstance(default, str):
        value = str(value if value is not None else "").strip()
    elif key == "voice.default":
        value = _voice_ref(value, store, guild_id, "value")  # guild 0: global voices only
    elif isinstance(default, bool):
        if not isinstance(value, bool):
            raise FieldError(f"{key} must be true or false", "value")
    elif isinstance(default, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise FieldError(f"{key} must be a number", "value") from None
        if value < 0:
            raise FieldError(f"{key} can't be negative", "value")
        if key == "sounds.max_volume" and value > 400:
            raise FieldError("sounds.max_volume goes up to 400 (%)", "value")
        if isinstance(default, int) and key.startswith("quota."):
            if value != int(value):
                raise FieldError(f"{key} must be a whole number", "value")
            value = int(value)
    elif isinstance(default, list):
        value = _str_list(value, "value")
    return value


@api
async def setting_put(request: web.Request) -> web.Response:
    store, gid, key = request.app["store"], _gid(request), request.match_info["key"]
    body = await _body(request)
    if "value" not in body:
        raise ValueError('expected {"value": ...}')
    value = clean_setting(key, body["value"], store, gid)
    store.set_setting(gid, key, value)
    return json_response({"key": key, "value": value, "effective": store.get_setting(gid, key)})


@api
async def setting_delete(request: web.Request) -> web.Response:
    from store import DEFAULT_SETTINGS

    store, gid, key = request.app["store"], _gid(request), request.match_info["key"]
    if key not in DEFAULT_SETTINGS:
        raise KeyError(f"unknown setting {key!r}")
    store.delete_setting(gid, key)
    return json_response({"key": key, "effective": store.get_setting(gid, key)})


# ───────────────────────────── quotas ─────────────────────────────

@api
async def quota_put(request: web.Request) -> web.Response:
    from store import QUOTA_RESOURCES

    store, gid, uid = request.app["store"], _gid(request), _int(request, "uid")
    resource = request.match_info["resource"]
    if resource not in QUOTA_RESOURCES:
        raise KeyError(f"unknown quota {resource!r}")
    body = await _body(request)
    limit = body.get("limit")
    if limit in (None, ""):
        limit = None
    else:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise FieldError("the limit must be a whole number", "limit") from None
        if limit < 0:
            raise FieldError("the limit can't be negative", "limit")
    store.set_quota_override(gid, uid, resource, limit)
    used, effective = store.quota(gid, uid, resource)
    return json_response({"used": used, "limit": effective, "override": limit})


@api
async def quota_decide(request: web.Request) -> web.Response:
    store, gid, qid = request.app["store"], _gid(request), _int(request, "qid")
    q = store.get_quota_request(qid)
    if q is None or q["guild_id"] != gid:
        raise KeyError(f"no quota request {qid} in this server")
    body = await _body(request)
    if not isinstance(body.get("approve"), bool):
        raise ValueError('expected {"approve": true|false}')
    amount = body.get("amount")
    if amount not in (None, ""):
        try:
            amount = int(amount)
        except (TypeError, ValueError):
            raise FieldError("amount must be a whole number", "amount") from None
        if amount < 1:
            raise FieldError("amount must be at least 1", "amount")
    else:
        amount = None
    decided = store.decide_quota_request(qid, body["approve"], decided_by="dashboard", amount=amount)
    return json_response({"request": decided})


# ───────────────────────────── test bench ─────────────────────────────

@api
async def test_bench(request: web.Request) -> web.Response:
    store, engine, gid = request.app["store"], request.app.get("engine"), _gid(request)
    if engine is None or not hasattr(engine, "explain"):
        return error("the test bench needs the reaction engine (with explain())", 501)
    body = await _body(request)
    text = str(body.get("text") or "").strip()
    if not text:
        raise FieldError("type a sentence", "text")
    user_id = to_user_id(body.get("user_id")) if body.get("user_id") not in (None, "") else None
    display_name = name = ""
    if user_id is not None:
        person = store.get_person(gid, user_id) or {}
        display_name = person.get("display_name") or ""
        name = store.nickname_for(gid, user_id, display_name)
    name = str(body.get("name") or name or "")
    results = engine.explain(gid, text, user_id=user_id, display_name=display_name, name=name)
    return json_response({"text": text, "user_id": user_id, "name": name, "results": results})


# ───────────────────────────── packs ─────────────────────────────

def _tmp_dir(app: web.Application) -> Path:
    path = app["base_dir"] / "data" / "tmp" / f"pack-{secrets.token_hex(6)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_export_zip(store, guild_id: int, work: Path, base_dir: Path, *, include_personal: bool) -> Path:
    """packs.export_pack, then one zip for the browser: pack.yaml at the top
    and the audio at the paths the YAML names (sounds/..., voices/...)."""
    import packs

    yaml_path = packs.export_pack(store, guild_id, work / "pack.yaml", include_personal=include_personal,
                                  base_dir=base_dir, header=f"# Pack exported from server {guild_id}\n")
    media = yaml_path.with_suffix(".zip")
    out = work / "download.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(yaml_path, "pack.yaml")
        if media.exists():
            with zipfile.ZipFile(media) as m:
                for info in m.infolist():
                    with m.open(info) as src, z.open(info.filename, "w") as dst:
                        shutil.copyfileobj(src, dst)
    return out


def unpack_upload(upload: Path, work: Path) -> Path:
    """An uploaded .yaml, or a .zip (ours: pack.yaml + audio; or the CLI's
    x.yaml + x.zip pair zipped together) -> work/pack.yaml (+ work/pack.zip
    with the audio), the layout packs.import_pack reads."""
    yaml_path = work / "pack.yaml"
    if not zipfile.is_zipfile(upload):
        shutil.move(upload, yaml_path)
        return yaml_path
    with zipfile.ZipFile(upload) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > MAX_ZIP_ENTRIES or sum(i.file_size for i in infos) > MAX_UNZIPPED:
            raise ValueError("the zip is too big")
        yamls = sorted((i for i in infos if i.filename.lower().endswith((".yaml", ".yml"))),
                       key=lambda i: (i.filename.count("/"), i.filename))
        if not yamls:
            raise ValueError("no .yaml pack in the zip")
        main = yamls[0]
        with z.open(main) as src, open(yaml_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        prefix = main.filename.rpartition("/")[0]
        prefix = prefix + "/" if prefix else ""
        stem = main.filename[: -len(Path(main.filename).suffix)]
        inner = next((i for i in infos if i.filename == stem + ".zip"), None)
        with zipfile.ZipFile(work / "pack.zip", "w", zipfile.ZIP_DEFLATED) as media:
            if inner is not None:  # the CLI's pair: x.yaml + x.zip
                with z.open(inner) as src, open(work / "inner.zip", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                with zipfile.ZipFile(work / "inner.zip") as iz:
                    entries = [i for i in iz.infolist() if not i.is_dir()]
                    if len(entries) > MAX_ZIP_ENTRIES or sum(i.file_size for i in entries) > MAX_UNZIPPED:
                        raise ValueError("the zip is too big")
                    for info in entries:
                        with iz.open(info) as src, media.open(info.filename, "w") as dst:
                            shutil.copyfileobj(src, dst)
            else:
                for info in infos:
                    if info is main or not info.filename.startswith(prefix):
                        continue
                    with z.open(info) as src, media.open(info.filename[len(prefix):], "w") as dst:
                        shutil.copyfileobj(src, dst)
    return yaml_path


@api
async def pack_export(request: web.Request) -> web.StreamResponse:
    store, gid, base_dir = request.app["store"], _gid(request), request.app["base_dir"]
    include_personal = request.query.get("personal", "1") not in ("0", "false", "no")
    work = _tmp_dir(request.app)
    try:
        path = await asyncio.to_thread(build_export_zip, store, gid, work, base_dir,
                                       include_personal=include_personal)
        response = web.StreamResponse(headers={
            "Content-Type": "application/zip",
            "Content-Disposition": f'attachment; filename="pack-{gid}.zip"',
            "Content-Length": str(path.stat().st_size),
        })
        await response.prepare(request)
        with open(path, "rb") as f:
            while chunk := await asyncio.to_thread(f.read, 256 * 1024):
                await response.write(chunk)
        await response.write_eof()
        return response
    finally:
        await asyncio.to_thread(shutil.rmtree, work, True)


@api
async def pack_import(request: web.Request) -> web.Response:
    import packs
    import yaml

    store, gid, base_dir = request.app["store"], _gid(request), request.app["base_dir"]
    if request.content_length and request.content_length > MAX_UPLOAD + 64 * 1024:
        return error(f"the pack is over {MAX_UPLOAD // 2**20} MB", 413)
    try:
        reader = await request.multipart()
    except (AssertionError, ValueError, KeyError):
        raise ValueError("send the pack as multipart/form-data") from None
    work = _tmp_dir(request.app)
    try:
        mode, upload = "merge", None
        while (part := await reader.next()) is not None:
            if part.name == "mode":
                mode = (await part.text()).strip()
            elif part.name == "file":
                upload = work / "upload"
                size = 0
                with open(upload, "wb") as f:
                    while chunk := await part.read_chunk(256 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD:
                            return error(f"the pack is over {MAX_UPLOAD // 2**20} MB", 413)
                        f.write(chunk)
        if upload is None or size == 0:
            raise FieldError("choose a pack file (.yaml or .zip)", "file")
        if mode not in ("merge", "replace"):
            raise FieldError("mode must be merge or replace", "mode")

        def run() -> dict:
            yaml_path = unpack_upload(upload, work)
            return packs.import_pack(store, gid, yaml_path, mode=mode, base_dir=base_dir,
                                     media_dir=base_dir / "data" / "media" / str(gid))

        try:
            result = await asyncio.to_thread(run)
        except (yaml.YAMLError, zipfile.BadZipFile, UnicodeDecodeError) as e:
            raise ValueError(f"not a valid pack: {e}") from None
        return json_response({"ok": True, "mode": mode, **result})
    finally:
        await asyncio.to_thread(shutil.rmtree, work, True)
