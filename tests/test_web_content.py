"""Dashboard editor routes against a real Store + ReactionEngine.

    .venv/bin/python -m pytest tests/test_web_content.py -q
"""
import asyncio
import io
import sys
import zipfile
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events import EventBus  # noqa: E402
from reactions import ReactionEngine  # noqa: E402
from web.dev_server import FakeController  # noqa: E402
from web.server import create_app  # noqa: E402

pytestmark = pytest.mark.asyncio

TOKEN = "content-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GID = 1100000000000000001
BIG_USER = 111111111111111111  # a real-looking snowflake: too big for a JS number
G = f"/api/g/{GID}"

GAG = {
    "kind": "gag", "name": "pelo", "enabled": True,
    "triggers": [{"type": "swap", "word": "pelo", "to": "calne", "also": ["pelos"]}],
    "options": [[{"type": "say", "text": "No, {subject} {connector} {to}."}]],
}


@pytest.fixture
def engine(store):
    return ReactionEngine(store)


@pytest_asyncio.fixture
async def client(aiohttp_client, store, engine, tmp_path):
    bus = EventBus()
    app = create_app(FakeController(bus), bus, token=TOKEN, store=store, engine=engine,
                     status_interval=60, base_dir=tmp_path)
    c = await aiohttp_client(app, headers=AUTH)
    return c


async def ok(response, status=200):
    body = await response.json()
    assert response.status == status, body
    return body


# ---------------------------------------------------------------- auth

ROUTES = [
    ("GET", f"{G}/content"), ("POST", f"{G}/reactions"), ("PATCH", f"{G}/reactions/1"),
    ("DELETE", f"{G}/reactions/1"), ("POST", f"{G}/reactions/order"), ("POST", f"{G}/reactions/1/phrase"),
    ("PATCH", f"{G}/people/1"), ("POST", f"{G}/people/1/revoke-consent"), ("POST", f"{G}/people/1/delete-text"),
    ("GET", "/api/languages"), ("PUT", f"{G}/settings/quota.gags"),
    ("DELETE", f"{G}/settings/quota.gags"), ("PUT", f"{G}/quota/1/gags"), ("POST", f"{G}/quota-requests/1/decide"),
    ("POST", f"{G}/test"), ("GET", f"{G}/pack/export"), ("POST", f"{G}/pack/import"),
]


@pytest.mark.parametrize("method,path", ROUTES)
async def test_every_route_needs_auth(client, method, path):
    r = await client.request(method, path, headers={"Authorization": ""}, json={})
    assert r.status == 401


async def test_cross_origin_write_refused(client):
    r = await client.post(f"{G}/reactions", json=GAG, headers={"Origin": "https://evil.example"})
    assert r.status == 403


async def test_no_store_no_routes(aiohttp_client):
    bus = EventBus()
    c = await aiohttp_client(create_app(FakeController(bus), bus, token=TOKEN), headers=AUTH)
    assert (await c.get(f"{G}/content")).status == 404


# ---------------------------------------------------------------- reactions

async def test_reaction_crud_round_trip(client, store):
    created = await ok(await client.post(f"{G}/reactions", json=GAG))
    rid = created["id"]
    assert created["reaction"]["creator"] == "admin"
    assert store.get_reaction(rid)["triggers"][0]["connectors"] == ["de", "e", "y"]

    content = await ok(await client.get(f"{G}/content"))
    assert [r["name"] for r in content["reactions"]] == ["pelo"]
    assert content["meta"]["test_bench"] in (True, False)

    patched = await ok(await client.patch(f"{G}/reactions/{rid}", json={"enabled": False, "chance": 0.5,
                                                                        "by_users": [str(BIG_USER), "Robin"]}))
    row = store.get_reaction(rid)
    assert row["enabled"] is False and row["chance"] == 0.5
    assert row["by_users"] == [BIG_USER, "Robin"]          # stored as an int
    assert patched["reaction"]["by_users"] == [str(BIG_USER), "Robin"]  # sent as a string

    await ok(await client.delete(f"{G}/reactions/{rid}"))
    assert store.get_reaction(rid) is None
    assert (await client.delete(f"{G}/reactions/{rid}")).status == 404


async def test_reaction_validation_is_400_with_field(client):
    bad = {**GAG, "triggers": [GAG["triggers"][0], {"type": "phrase", "phrases": []}]}
    r = await client.post(f"{G}/reactions", json=bad)
    body = await r.json()
    assert r.status == 400 and body["field"] == "triggers.1" and "phrases" in body["error"]

    bad = {**GAG, "options": [[{"type": "say", "text": "ok"}], [{"type": "builtin", "action": "dance"}]]}
    body = await (await client.post(f"{G}/reactions", json=bad)).json()
    assert body["field"] == "options.1"

    bad = {**GAG, "options": [[{"type": "sound", "sound_id": "999"}]]}
    body = await (await client.post(f"{G}/reactions", json=bad)).json()
    assert body["field"] == "options.0.0"

    r = await client.post(f"{G}/reactions", json={**GAG, "chance": 3})
    assert r.status == 400 and (await r.json())["field"] == "chance"
    r = await client.post(f"{G}/reactions", json={"name": "x"})
    assert r.status == 400


async def test_event_trigger_person_and_steps(client, store, tmp_path):
    sound_id = store.add_sound(GID, "bruh", "bruh.ogg")
    voice_id = store.add_voice("narrator", guild_id=GID, status="ready")
    body = {"kind": "response", "name": "hola robin",
            "triggers": [{"type": "event", "event": "hello", "user_id": str(BIG_USER)}],
            "options": [[{"type": "say", "text": "Llegó {name}", "voice_id": str(voice_id)},
                         {"type": "sound", "sound_id": str(sound_id)}]],
            "voice_id": "@speaker"}
    rid = (await ok(await client.post(f"{G}/reactions", json=body)))["id"]
    row = store.get_reaction(rid)
    assert row["triggers"][0]["user_id"] == BIG_USER
    assert row["options"][0][0]["voice_id"] == voice_id and row["options"][0][1]["sound_id"] == sound_id
    assert row["voice_id"] == "@speaker"
    # A voice from another server is refused.
    other = store.add_voice("other", guild_id=42)
    r = await client.patch(f"{G}/reactions/{rid}", json={"voice_id": str(other)})
    assert r.status == 400 and (await r.json())["field"] == "voice_id"


async def test_reorder(client, store):
    ids = [(await ok(await client.post(f"{G}/reactions", json={**GAG, "name": n})))["id"] for n in "abc"]
    await ok(await client.post(f"{G}/reactions/order", json={"ids": [str(i) for i in reversed(ids)]}))
    assert [r["name"] for r in store.list_reactions(GID)] == ["c", "b", "a"]
    assert (await client.post(f"{G}/reactions/order", json={"ids": [999]})).status == 400


async def test_other_guilds_reaction_is_404(client, store):
    rid = store.add_reaction(42, "gag", "x", [{"type": "phrase", "phrases": ["x"]}], [[{"type": "say", "text": "y"}]])
    assert (await client.patch(f"{G}/reactions/{rid}", json={"enabled": False})).status == 404
    assert (await client.delete(f"{G}/reactions/{rid}")).status == 404


async def test_add_phrase(client, store):
    rid = store.add_reaction(GID, "gag", "chamba", [{"type": "phrase", "phrases": ["chamba"]}],
                             [[{"type": "say", "text": "¿CHAMBA?"}]])
    await ok(await client.post(f"{G}/reactions/{rid}/phrase", json={"trigger": 0, "phrase": "chambita"}))
    await ok(await client.post(f"{G}/reactions/{rid}/phrase",
                               json={"trigger": {"type": "phrase", "phrases": ["chamba", "chambita"]}, "phrase": "Chambita"}))
    assert store.get_reaction(rid)["triggers"][0]["phrases"] == ["chamba", "chambita"]


# ---------------------------------------------------------------- approvals, people

async def test_approve_flow(client, store, engine):
    rid = store.add_reaction(GID, "gag", "pizza", [{"type": "phrase", "phrases": ["pizza"]}],
                             [[{"type": "say", "text": "pizza!"}]], status="pending", created_by=BIG_USER)
    store.set_person(GID, BIG_USER, display_name="Robin")
    content = await ok(await client.get(f"{G}/content"))
    pending = [r for r in content["reactions"] if r["status"] == "pending"]
    assert [r["creator"] for r in pending] == ["Robin"]
    assert pending[0]["created_by"] == str(BIG_USER)
    assert engine.match(GID, "quiero pizza", 1, "x", "x") is None  # pending: never fires

    await ok(await client.patch(f"{G}/reactions/{rid}", json={"status": "approved"}))
    assert store.get_reaction(rid)["status"] == "approved"
    assert engine.match(GID, "quiero pizza", 1, "x", "x") is not None  # engine recompiled on change

    r = await client.patch(f"{G}/reactions/{rid}", json={"status": "maybe"})
    assert r.status == 400


async def test_people_nickname_and_consent(client, store):
    store.set_person(GID, BIG_USER, display_name="Robin 🎮")
    store.set_consent(BIG_USER, "accepted", purpose="voice")
    store.set_consent(BIG_USER, "accepted", purpose="transcripts")
    await ok(await client.patch(f"{G}/people/{BIG_USER}", json={"nickname": "Robbie"}))
    assert store.get_person(GID, BIG_USER)["nickname"] == "Robbie"
    content = await ok(await client.get(f"{G}/content"))
    person = content["people"][0]
    assert person["user_id"] == str(BIG_USER)
    assert person["consent"] == {"voice": "accepted", "transcripts": "accepted"}
    assert set(content["meta"]["consent_purposes"]) == {"voice", "transcripts"}

    await ok(await client.post(f"{G}/people/{BIG_USER}/revoke-consent", json={"purpose": "transcripts"}))
    assert store.get_consent(BIG_USER, "transcripts")["status"] == "revoked"
    assert store.get_consent(BIG_USER, "voice")["status"] == "accepted"  # only that purpose
    await ok(await client.post(f"{G}/people/{BIG_USER}/revoke-consent"))  # default: voice
    assert store.get_consent(BIG_USER, "voice")["status"] == "revoked"
    r = await client.post(f"{G}/people/{BIG_USER}/revoke-consent", json={"purpose": "soul"})
    assert r.status == 400 and (await r.json())["field"] == "purpose"

    await ok(await client.patch(f"{G}/people/{BIG_USER}", json={"nickname": ""}))
    assert store.get_person(GID, BIG_USER)["nickname"] is None


async def test_delete_a_persons_text(client, store):
    store.log_event(GID, "no match", user_id=BIG_USER, text="secret words")
    store.log_event(GID, "no match", user_id=BIG_USER + 1, text="someone else")
    store.log_event(42, "no match", user_id=BIG_USER, text="another server")
    out = await ok(await client.post(f"{G}/people/{BIG_USER}/delete-text"))
    assert out["cleared"] == 1
    texts = {(r["guild_id"], r["user_id"]): r["text"] for r in store.list_events(limit=10)}
    assert texts[(GID, BIG_USER)] is None and texts[(GID, BIG_USER + 1)] == "someone else"
    assert texts[(42, BIG_USER)] == "another server"  # only this server


async def test_languages_and_language_setting(client, store):
    out = await ok(await client.get("/api/languages"))
    assert "en" in out["languages"] and out["default"] in out["languages"]
    content = await ok(await client.get(f"{G}/content"))
    assert content["meta"]["languages"] == out["languages"]
    await ok(await client.put(f"{G}/settings/language", json={"value": out["languages"][0]}))
    assert store.get_setting(GID, "language") == out["languages"][0]
    r = await client.put(f"{G}/settings/language", json={"value": "klingon"})
    assert r.status == 400


async def test_transcripts_setting_only_through_the_bot(client, store):
    r = await client.put(f"{G}/settings/transcripts.enabled", json={"value": True})
    assert r.status == 400 and "toggle" in (await r.json())["error"]
    assert not store.get_setting(GID, "transcripts.enabled")
    # the bot's toggle does it (FakeController with a store writes the setting, like bot.py)
    client.server.app["controller"].store = store
    await ok(await client.post("/api/control", json={"action": "toggle", "guild_id": str(GID),
                                                      "name": "transcripts", "value": True}))
    assert store.get_setting(GID, "transcripts.enabled") is True


async def test_starter_pack_setting(client, store):
    content = await ok(await client.get(f"{G}/content"))
    packs = content["meta"]["starter_packs"]
    assert "base-en" in packs
    await ok(await client.put(f"{G}/settings/content.starter_pack", json={"value": "base-en"}))
    await ok(await client.put(f"{G}/settings/content.starter_pack", json={"value": "none"}))
    assert (await client.put(f"{G}/settings/content.starter_pack", json={"value": "nope"})).status == 400


async def test_new_settings(client, store):
    await ok(await client.put(f"{G}/settings/time.zone", json={"value": "Europe/Madrid"}))
    assert store.get_setting(GID, "time.zone") == "Europe/Madrid"
    assert (await client.put(f"{G}/settings/time.zone", json={"value": "Mars/Olympus"})).status == 400
    await ok(await client.put(f"{G}/settings/time.zone", json={"value": ""}))
    assert store.get_setting(GID, "time.zone") is None
    await ok(await client.put(f"{G}/settings/notices.channel_id", json={"value": "123456789012345678"}))
    assert store.get_setting(GID, "notices.channel_id") == 123456789012345678
    assert (await client.put(f"{G}/settings/notices.channel_id", json={"value": "general"})).status == 400
    await ok(await client.put(f"{G}/settings/sounds.max_volume", json={"value": 150}))
    assert (await client.put(f"{G}/settings/sounds.max_volume", json={"value": 900})).status == 400
    await ok(await client.put(f"{G}/settings/llm.persona", json={"value": "  a grumpy pirate "}))
    assert store.get_setting(GID, "llm.persona") == "a grumpy pirate"
    await ok(await client.put(f"{G}/settings/llm.enabled", json={"value": False}))
    assert (await client.put(f"{G}/settings/llm.enabled", json={"value": "yes"})).status == 400


# ---------------------------------------------------------------- quotas, settings

async def test_quota_request_decide(client, store):
    qid = store.request_quota(GID, BIG_USER, "gags", 5, "más")
    content = await ok(await client.get(f"{G}/content"))
    assert content["requests"][0]["id"] == qid and content["requests"][0]["limit"] == 15
    assert content["quotas"]["users"][0]["gags"] == {"used": 0, "limit": 15, "override": None}

    out = await ok(await client.post(f"{G}/quota-requests/{qid}/decide", json={"approve": True, "amount": 3}))
    assert out["request"]["status"] == "approved"
    assert store.quota_limit(GID, BIG_USER, "gags") == 18
    r = await client.post(f"{G}/quota-requests/{qid}/decide", json={"approve": False})
    assert r.status == 400  # already decided

    qid2 = store.request_quota(GID, BIG_USER, "sounds", 2)
    await ok(await client.post(f"{G}/quota-requests/{qid2}/decide", json={"approve": False}))
    assert store.get_quota_request(qid2)["status"] == "denied"
    assert store.quota_limit(GID, BIG_USER, "sounds") == 15


async def test_quota_override(client, store):
    out = await ok(await client.put(f"{G}/quota/{BIG_USER}/voices", json={"limit": 9}))
    assert out == {"used": 0, "limit": 9, "override": 9}
    await ok(await client.put(f"{G}/quota/{BIG_USER}/voices", json={"limit": None}))
    assert store.quota_limit(GID, BIG_USER, "voices") == 5
    assert (await client.put(f"{G}/quota/{BIG_USER}/hats", json={"limit": 1})).status == 404
    assert (await client.put(f"{G}/quota/{BIG_USER}/gags", json={"limit": -1})).status == 400


async def test_settings_inherit_and_reset(client, store):
    await ok(await client.put("/api/g/0/settings/gags.cooldown_s", json={"value": 7}))   # global
    content = await ok(await client.get(f"{G}/content"))
    assert content["settings"]["effective"]["gags.cooldown_s"] == 7
    assert "gags.cooldown_s" not in content["settings"]["own"]

    out = await ok(await client.put(f"{G}/settings/gags.cooldown_s", json={"value": 1.5}))
    assert out["effective"] == 1.5
    out = await ok(await client.delete(f"{G}/settings/gags.cooldown_s"))
    assert out["effective"] == 7   # back to the global value
    await ok(await client.delete("/api/g/0/settings/gags.cooldown_s"))
    assert store.get_setting(GID, "gags.cooldown_s") == 3

    await ok(await client.put(f"{G}/settings/bot.wake_words", json={"value": ["oye bot", " "]}))
    assert store.get_setting(GID, "bot.wake_words") == ["oye bot"]
    await ok(await client.put(f"{G}/settings/quota.gags", json={"value": 20}))
    assert store.get_setting(GID, "quota.gags") == 20
    assert (await client.put(f"{G}/settings/quota.gags", json={"value": 2.5})).status == 400
    assert (await client.put(f"{G}/settings/user_content.needs_approval", json={"value": "yes"})).status == 400
    assert (await client.put(f"{G}/settings/no.such", json={"value": 1})).status == 404

    # global-only settings
    assert (await client.put(f"{G}/settings/cache.tts_mb", json={"value": 800})).status == 400
    await ok(await client.put("/api/g/0/settings/cache.tts_mb", json={"value": 800}))
    bot_voice = store.add_voice("bot", status="ready")  # global
    server_voice = store.add_voice("mine", guild_id=GID)
    await ok(await client.put("/api/g/0/settings/voice.bot", json={"value": str(bot_voice)}))
    assert store.get_setting(0, "voice.bot") == bot_voice
    assert (await client.put("/api/g/0/settings/voice.bot", json={"value": str(server_voice)})).status == 400
    await ok(await client.put(f"{G}/settings/voice.default", json={"value": str(server_voice)}))
    assert store.get_setting(GID, "voice.default") == server_voice


# ---------------------------------------------------------------- packs

async def test_pack_export_import_round_trip(client, store, tmp_path):
    (tmp_path / "bruh.ogg").write_bytes(b"OggS fake audio")
    sound_id = store.add_sound(GID, "bruh", "bruh.ogg")
    store.add_reaction(GID, "gag", "bruh", [{"type": "phrase", "phrases": ["bruh"]}],
                       [[{"type": "sound", "sound_id": sound_id}]])
    store.add_reaction(GID, "gag", "chamba", [{"type": "phrase", "phrases": ["chamba"]}],
                       [[{"type": "say", "text": "¿CHAMBA?"}]])
    store.set_setting(GID, "gags.cooldown_s", 5)

    r = await client.get(f"{G}/pack/export")
    assert r.status == 200 and r.headers["Content-Type"] == "application/zip"
    data = await r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        assert "pack.yaml" in names and "sounds/bruh.ogg" in names
        assert "chamba" in z.read("pack.yaml").decode()
    for _ in range(50):  # temp files are removed right after the last byte is sent
        if not any((tmp_path / "data" / "tmp").iterdir()):
            break
        await asyncio.sleep(0.02)
    assert not any((tmp_path / "data" / "tmp").iterdir())

    other = 2200000000000000002
    form = {"mode": "merge", "file": io.BytesIO(data)}
    out = await ok(await client.post(f"/api/g/{other}/pack/import", data=form))
    assert out["reactions"] == 2 and out["sounds"] == 1 and out["warnings"] == []
    imported = store.find_sound(other, "bruh")
    assert (tmp_path / imported["path"]).read_bytes() == b"OggS fake audio"
    assert store.get_setting(other, "gags.cooldown_s") == 5
    bruh = next(r for r in store.list_reactions(other) if r["name"] == "bruh")
    assert bruh["options"][0][0]["sound_id"] == imported["id"]

    # replace with a plain YAML that only has one reaction
    yaml_text = "format: 1\nreactions:\n- name: hola\n  phrase: hola\n  say: Hola.\n  voice: nobody\n"
    form = {"mode": "replace", "file": io.BytesIO(yaml_text.encode())}
    out = await ok(await client.post(f"/api/g/{other}/pack/import", data=form))
    assert [r["name"] for r in store.list_reactions(other)] == ["hola"]
    assert any("nobody" in w for w in out["warnings"])


async def test_pack_import_errors(client):
    r = await client.post(f"{G}/pack/import", data={"mode": "merge", "file": io.BytesIO(b"- not: [a pack")})
    assert r.status == 400
    r = await client.post(f"{G}/pack/import", data={"mode": "sideways", "file": io.BytesIO(b"format: 1\n")})
    assert r.status == 400 and (await r.json())["field"] == "mode"
    r = await client.post(f"{G}/pack/import", data={"mode": "merge"})
    assert r.status == 400


async def test_pack_upload_cap(client, monkeypatch):
    import web.routes_content as rc
    monkeypatch.setattr(rc, "MAX_UPLOAD", 1000)
    r = await client.post(f"{G}/pack/import", data={"mode": "merge", "file": io.BytesIO(b"x" * 5000)})
    assert r.status == 413


# ---------------------------------------------------------------- test bench

async def test_bench(client, store, engine):
    store.add_reaction(GID, "gag", "chamba", [{"type": "phrase", "phrases": ["chamba"]}],
                       [[{"type": "say", "text": "¿CHAMBA?"}]], cooldown_s=60)
    if not hasattr(engine, "explain"):
        r = await client.post(f"{G}/test", json={"text": "hay chamba"})
        assert r.status == 501
        pytest.skip("reactions.ReactionEngine.explain() isn't there yet")
    out = await ok(await client.post(f"{G}/test", json={"text": "hay chamba", "user_id": str(BIG_USER)}))
    assert any(x.get("would_fire") and x["name"] == "chamba" for x in out["results"])
    # dry run: running it again still fires (no cooldown started)
    out = await ok(await client.post(f"{G}/test", json={"text": "hay chamba", "user_id": str(BIG_USER)}))
    assert any(x.get("would_fire") for x in out["results"])
    assert (await client.post(f"{G}/test", json={"text": " "})).status == 400


# ---------------------------------------------------------------- live updates

async def test_content_changed_pushed(client, store):
    async with client.ws_connect("/ws") as ws:
        assert (await ws.receive_json(timeout=2))["type"] == "hello"
        await ok(await client.post(f"{G}/reactions", json=GAG))
        # a change made off the loop (like the bot's threads) arrives too
        await asyncio.to_thread(store.set_setting, GID, "gags.intensity", 0.5)
        seen = {}
        while len(seen) < 1 or "settings" not in seen.get(str(GID), set()):
            msg = await ws.receive_json(timeout=3)
            if msg["type"] == "content_changed":
                seen.setdefault(msg["guild_id"], set()).update(msg["tables"])
        assert "reactions" in seen[str(GID)]
