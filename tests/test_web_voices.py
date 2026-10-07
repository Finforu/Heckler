"""Voices & sounds routes: a real Store, VoiceLibrary (on the fake model from
web/dev_server.py: no torch, no GPU) and SoundLibrary, all under tmp_path.

    .venv/bin/python -m pytest tests/test_web_voices.py -q
"""
import asyncio
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events import EventBus  # noqa: E402
from sound_library import SoundLibrary  # noqa: E402
from web.dev_server import FakeController, make_library  # noqa: E402
from web.server import create_app  # noqa: E402

pytestmark = pytest.mark.asyncio

TOKEN = "voices-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GID = 1100000000000000001          # FakeController's first server (in a call)
G = f"/api/g/{GID}"
USER = 111111111111111111


def wav_bytes(seconds=12.0, freq=220.0, rate=24000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    audio = (0.3 * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    audio[(t % 3.0) > 2.2] = 0  # pauses, so speech detection finds lines
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def library(store, tmp_path):
    return make_library(store, tmp_path / "data")


@pytest.fixture
def sounds(store, tmp_path):
    return SoundLibrary(store, tmp_path / "data")


@pytest.fixture
def controller(library):
    return FakeController(EventBus(), library=library)


@pytest_asyncio.fixture
async def client(aiohttp_client, store, library, sounds, controller, tmp_path):
    bus = controller.bus
    app = create_app(controller, bus, token=TOKEN, store=store, library=library, sounds=sounds,
                     status_interval=60, base_dir=tmp_path)
    return await aiohttp_client(app, headers=AUTH)


async def ok(response, status=200):
    body = await response.json()
    assert response.status == status, body
    return body


async def builds_done(controller):
    await asyncio.gather(*controller.builds)
    controller.builds.clear()


async def new_clone(client, name="pirate", seconds=12.0) -> dict:
    form = {"name": name, "file": io.BytesIO(wav_bytes(seconds))}
    return (await ok(await client.post(f"{G}/voices", data=form)))["voice"]


# ---------------------------------------------------------------- auth

ROUTES = [
    ("GET", f"{G}/voices"), ("POST", f"{G}/voices"), ("PATCH", f"{G}/voices/1"), ("DELETE", f"{G}/voices/1"),
    ("POST", f"{G}/voices/1/ingest"), ("GET", f"{G}/voices/1/audio/ref"), ("GET", f"{G}/voices/1/peaks"),
    ("POST", f"{G}/voices/1/transcribe"), ("POST", f"{G}/voices/1/build"), ("POST", f"{G}/voices/1/preview"),
    ("POST", f"{G}/voices/cache/clear"), ("POST", f"{G}/speakers/1/rebuild"), ("POST", f"{G}/speakers/1/promote"),
    ("DELETE", f"{G}/speakers/1"), ("POST", f"{G}/sounds"), ("PATCH", f"{G}/sounds/1"), ("DELETE", f"{G}/sounds/1"),
    ("GET", f"{G}/sounds/1/audio"), ("POST", f"{G}/sounds/1/reaction"), ("POST", f"{G}/sounds/1/play"),
    ("GET", f"{G}/board"),
]


@pytest.mark.parametrize("method,path", ROUTES)
async def test_every_route_needs_auth(client, method, path):
    assert (await client.request(method, path, headers={"Authorization": ""})).status == 401


@pytest.mark.parametrize("method,path", [r for r in ROUTES if r[0] != "GET"])
async def test_writes_refuse_other_origins(client, method, path):
    r = await client.request(method, path, headers={"Origin": "https://evil.example"})
    assert r.status == 403


async def test_no_libraries_no_routes(aiohttp_client, store):
    bus = EventBus()
    c = await aiohttp_client(create_app(FakeController(bus), bus, token=TOKEN, store=store), headers=AUTH)
    assert (await c.get(f"{G}/voices")).status == 404
    assert (await c.get(f"{G}/board")).status == 404


# ---------------------------------------------------------------- clone voices

async def test_clone_upload_waveform_transcript_build_preview(client, store, controller):
    voice = await new_clone(client)
    vid = voice["id"]
    assert voice["status"] == "draft" and voice["guild_id"] == str(GID)

    listing = await ok(await client.get(f"{G}/voices"))
    row = next(v for v in listing["voices"] if v["id"] == vid)
    assert row["needs_build"] and not row["has_prompt"] and 0.5 < row["ref_seconds"] <= 12
    assert row["source_name"] == "source.wav" and listing["bot_voice_id"] is None
    assert any(c["category"] == "accent" for c in listing["design"])

    peaks = await ok(await client.get(f"{G}/voices/{vid}/peaks?which=source&n=300"))
    assert peaks["duration"] == pytest.approx(12, abs=0.1) and 250 <= len(peaks["peaks"]) <= 300
    assert max(peaks["peaks"]) == 1.0
    # a tone isn't speech to the VAD: the automatic choice is the first 12 s, as in select_speech
    assert peaks["segments"] == [] and peaks["auto"] == [0.0, 12.0]
    ref_peaks = await ok(await client.get(f"{G}/voices/{vid}/peaks?which=ref"))
    assert ref_peaks["segments"] == [] and ref_peaks["auto"] is None

    r = await client.get(f"{G}/voices/{vid}/audio/ref")
    assert r.status == 200 and r.headers["Content-Type"] == "audio/wav" and (await r.read())[:4] == b"RIFF"
    assert (await client.get(f"{G}/voices/{vid}/audio/source")).status == 200

    # no transcript yet: build refuses
    assert (await client.post(f"{G}/voices/{vid}/build")).status == 400
    out = await ok(await client.post(f"{G}/voices/{vid}/transcribe"))
    assert out["ref_text"] == "hello this is a voice test"
    await ok(await client.patch(f"{G}/voices/{vid}", json={"ref_text": "Hello, this is a voice test."}))

    out = await ok(await client.post(f"{G}/voices/{vid}/build"))
    assert out["voice"]["status"] == "queued"
    await builds_done(controller)
    assert store.get_voice(vid)["status"] == "ready"

    r = await client.post(f"{G}/voices/{vid}/preview", json={"text": "hola qué tal"})
    assert r.status == 200 and r.headers["Content-Type"] == "audio/wav"
    assert (await r.read())[:4] == b"RIFF"
    assert ("preview", {"text": "hola qué tal", "voice_id": vid}) in controller.calls

    # a new transcript changes the voice: rebuild needed; gain doesn't
    out = await ok(await client.patch(f"{G}/voices/{vid}", json={"gain_db": -3}))
    assert out["rebuild"] is False and out["voice"]["status"] == "ready"
    out = await ok(await client.patch(f"{G}/voices/{vid}", json={"ref_text": "otra cosa"}))
    assert out["rebuild"] is True and out["voice"]["status"] == "draft"

    # a new selection of the same source: a 3 s reference, transcript cleared
    out = await ok(await client.post(f"{G}/voices/{vid}/ingest", data={"start_s": "1", "end_s": "4"}))
    assert out["voice"]["ref_text"] is None
    listing = await ok(await client.get(f"{G}/voices"))
    assert next(v for v in listing["voices"] if v["id"] == vid)["ref_seconds"] == pytest.approx(3, abs=0.05)
    r = await client.post(f"{G}/voices/{vid}/ingest", data={"start_s": "4", "end_s": "1"})
    assert r.status == 400 and (await r.json())["field"] == "end_s"


async def test_voice_edits_validated(client):
    vid = (await new_clone(client))["id"]
    for body, field in (({"speed": 9}, "speed"), ({"num_step": 1}, "num_step"), ({"language": "spanish!"}, "language"),
                        ({"name": ""}, "name"), ({"instruct": "male"}, "instruct")):
        r = await client.patch(f"{G}/voices/{vid}", json=body)
        assert r.status == 400 and (await r.json())["field"] == field, body
    r = await client.patch(f"{G}/voices/{vid}", json={"status": "ready"})
    assert r.status == 400
    out = await ok(await client.patch(f"{G}/voices/{vid}", json={"name": "Capitán", "tags": "pirate, deep",
                                                                  "language": "es", "num_step": 24}))
    assert out["voice"]["tags"] == ["pirate", "deep"] and out["voice"]["num_step"] == 24


async def test_bad_upload(client, store):
    r = await client.post(f"{G}/voices", data={"name": "x", "file": io.BytesIO(b"not audio at all" * 100)})
    assert r.status == 400
    assert store.find_voice(GID, "x") is None  # cleaned up
    r = await client.post(f"{G}/voices", data={"name": "x"})
    assert r.status == 400 and (await r.json())["field"] == "file"
    store.set_setting(GID, "voices.max_mb", 0.01)
    r = await client.post(f"{G}/voices", data={"name": "big", "file": io.BytesIO(wav_bytes(5))})
    assert r.status == 413


async def test_file_routes_stay_inside_the_voice_folder(client, store, library, tmp_path):
    vid = (await new_clone(client))["id"]
    secret = tmp_path / "secret.wav"
    secret.write_bytes(wav_bytes(1))
    for which in ("prompt", "..", "..%2F..%2Fbot.db", "ref.wav", "%2E%2E"):
        assert (await client.get(f"{G}/voices/{vid}/audio/{which}")).status == 404, which
    assert (await client.get(f"{G}/voices/{vid}/audio/../../../bot.db")).status == 404
    # a source that is a symlink pointing outside the folder is refused
    folder = library.folder(vid)
    for old in folder.glob("source.*"):
        old.unlink()
    os.symlink(secret, folder / "source.wav")
    assert (await client.get(f"{G}/voices/{vid}/audio/source")).status == 404
    # another server's voice
    other = store.add_voice("elsewhere", guild_id=42)
    assert (await client.get(f"{G}/voices/{other}/audio/ref")).status == 404
    assert (await client.patch(f"{G}/voices/{other}", json={"name": "mine"})).status == 404


async def test_delete_goes_through_the_library(client, store, library, monkeypatch):
    vid = (await new_clone(client))["id"]
    calls = []
    real = library.delete_voice
    monkeypatch.setattr(library, "delete_voice", lambda v: (calls.append(v), real(v)))
    await ok(await client.delete(f"{G}/voices/{vid}"))
    assert calls == [vid] and store.get_voice(vid) is None and not library.folder(vid).exists()


async def test_bot_voice_is_marked_and_not_deletable(client, store, library, tmp_path):
    src = tmp_path / "bot.wav"
    src.write_bytes(wav_bytes(6))
    bot_id = library.ensure_bot_voice(src, name="Robo")
    listing = await ok(await client.get(f"{G}/voices"))
    assert listing["bot_voice_id"] == str(bot_id)
    assert next(v for v in listing["voices"] if v["id"] == bot_id)["is_bot"] is True
    assert (await client.delete(f"{G}/voices/{bot_id}")).status == 400


# ---------------------------------------------------------------- designed voices

async def test_designed_voice(client, store, controller):
    for instruct in ("deep voice", "male, female", "british accent, 四川话", ""):
        r = await client.post(f"{G}/voices", json={"kind": "designed", "name": "n", "instruct": instruct})
        assert r.status == 400 and (await r.json())["field"] == "instruct", instruct
    voice = (await ok(await client.post(f"{G}/voices", json={
        "kind": "designed", "name": "narrator", "instruct": "Female, High Pitch，british accent", "speed": 1.1})))["voice"]
    assert voice["instruct"] == "female, high pitch, british accent" and voice["status"] == "draft"
    await ok(await client.post(f"{G}/voices/{voice['id']}/build"))
    await builds_done(controller)
    assert store.get_voice(voice["id"])["status"] == "ready"
    r = await client.post(f"{G}/voices/{voice['id']}/preview", json={})  # default preview text
    assert r.status == 200
    out = await ok(await client.patch(f"{G}/voices/{voice['id']}", json={"instruct": "male, elderly"}))
    assert out["rebuild"] is True and out["voice"]["status"] == "draft"


# ---------------------------------------------------------------- people's own voices

async def speaker(store, library):
    store.set_person(GID, USER, display_name="Robin")
    store.set_consent(USER, "accepted")
    for k in range(5):
        t = np.arange(2 * 24000) / 24000
        library.update_speaker(USER, (0.3 * np.sin(2 * np.pi * (200 + k) * t)).astype(np.float32), f"frase numero {k}")
    vid = library.speaker_voice(None, USER, ready_only=False)["id"]
    library.build(vid)
    return vid


async def test_speaker_promote_needs_consent(client, store, library, controller):
    vid = await speaker(store, library)
    listing = await ok(await client.get(f"{G}/voices"))
    assert [s["id"] for s in listing["speakers"]] == [vid]
    assert listing["speakers"][0]["consent"] == "accepted" and listing["speakers"][0]["owner"] == "Robin"
    assert all(v["kind"] != "speaker" for v in listing["voices"])
    assert (await client.get(f"{G}/voices/{vid}/audio/ref")).status == 200

    store.set_consent(USER, "pending")
    r = await client.post(f"{G}/speakers/{USER}/promote", json={})
    assert r.status == 400 and "consent" in (await r.json())["error"]
    assert (await client.get(f"{G}/voices/{vid}/audio/ref")).status == 404  # private without consent
    assert (await client.post(f"{G}/speakers/{USER}/rebuild")).status == 400

    store.set_consent(USER, "accepted")
    out = await ok(await client.post(f"{G}/speakers/{USER}/promote", json={"name": "Robin clone"}))
    new = out["voice"]
    assert new["kind"] == "clone" and new["guild_id"] == str(GID) and new["owner_user_id"] is None
    assert new["ref_text"] == store.get_voice(vid)["ref_text"] and new["status"] == "queued"
    await builds_done(controller)
    assert store.get_voice(new["id"])["status"] == "ready"
    assert (await client.post(f"{G}/speakers/{USER}/promote", json={"name": "Robin clone"})).status == 400  # taken

    await ok(await client.post(f"{G}/speakers/{USER}/rebuild"))
    await builds_done(controller)


async def test_speaker_delete_goes_through_the_library(client, store, library, monkeypatch):
    vid = await speaker(store, library)
    calls = []
    real = library.delete_speaker
    monkeypatch.setattr(library, "delete_speaker", lambda u: (calls.append(u), real(u))[1])
    out = await ok(await client.delete(f"{G}/speakers/{USER}"))
    assert out["deleted"] == 1 and calls == [USER] and store.get_voice(vid) is None
    assert (await client.delete(f"{G}/speakers/{USER}")).status == 404
    # someone this server doesn't know
    assert (await client.post(f"{G}/speakers/123/promote", json={})).status == 404


async def test_cache_clear(client, store, library, controller):
    vid = (await new_clone(client))["id"]
    await ok(await client.post(f"{G}/voices/{vid}/transcribe"))
    await ok(await client.post(f"{G}/voices/{vid}/build"))
    await builds_done(controller)
    assert (await client.post(f"{G}/voices/{vid}/preview", json={"text": "hola"})).status == 200
    listing = await ok(await client.get(f"{G}/voices"))
    assert listing["cache"]["bytes"] > 0 and next(v for v in listing["voices"] if v["id"] == vid)["cache_bytes"] > 0
    out = await ok(await client.post(f"{G}/voices/cache/clear", json={"voice_id": str(vid)}))
    assert out["bytes"] == 0
    await ok(await client.post(f"{G}/voices/cache/clear"))


# ---------------------------------------------------------------- sounds

async def test_sound_upload_serve_edit_delete(client, store, sounds, controller):
    out = await ok(await client.post(f"{G}/sounds", data={"name": "Risa Malvada", "file": io.BytesIO(wav_bytes(2, 600))}))
    sound = out["sound"]
    sid = sound["id"]
    assert sound["name"] == "risa-malvada" and sound["created_by"] is None and 1 < sound["duration_s"] <= 2

    r = await client.get(f"{G}/sounds/{sid}/audio")
    assert r.status == 200 and r.headers["Content-Type"] == "audio/flac"
    assert (await client.get(f"/api/g/42/sounds/{sid}/audio")).status == 404

    r = await client.post(f"{G}/sounds", data={"name": "risa malvada", "file": io.BytesIO(wav_bytes(1))})
    assert r.status == 400  # name taken
    await ok(await client.patch(f"{G}/sounds/{sid}", json={"name": "Risa", "enabled": False, "gain_db": -4}))
    row = store.get_sound(sid)
    assert row["name"] == "risa" and row["enabled"] is False and row["gain_db"] == -4
    assert (await client.patch(f"{G}/sounds/{sid}", json={"name": "!!!"})).status == 400

    board = await ok(await client.get(f"{G}/board"))
    assert board["sounds"] == []  # disabled
    await ok(await client.patch(f"{G}/sounds/{sid}", json={"enabled": True}))
    await ok(await client.patch(f"{G}/sounds/{sid}", json={"volume": 50}))
    assert store.get_sound(sid)["gain_db"] == pytest.approx(-6.02, abs=0.01)
    await ok(await client.patch(f"{G}/sounds/{sid}", json={"volume": 0}))
    assert store.get_sound(sid)["gain_db"] == -60
    assert (await client.patch(f"{G}/sounds/{sid}", json={"volume": 500})).status == 400
    board = await ok(await client.get(f"{G}/board"))
    assert [s["name"] for s in board["sounds"]] == ["risa"]

    out = await ok(await client.post(f"{G}/sounds/{sid}/reaction", json={"trigger": {"type": "phrase", "phrases": "jaja"}}))
    reaction = store.get_reaction(out["id"])
    assert reaction["kind"] == "sound" and reaction["options"] == [[{"type": "sound", "sound_id": sid}]]
    out = await ok(await client.post(f"{G}/sounds/{sid}/reaction", json={}))
    assert store.get_reaction(out["id"])["triggers"] == [{"type": "slash", "name": "risa"}]
    assert (await client.post(f"{G}/sounds/{sid}/reaction", json={"trigger": {"type": "event"}})).status == 400

    out = await ok(await client.post(f"{G}/sounds/{sid}/play"))
    assert out["message"] == "Playing" and controller.calls[-1][0] == "play_sound"
    r = await client.post(f"/api/g/1100000000000000002/sounds/{sid}/play")
    assert r.status == 404  # not that server's sound

    path = Path(row["path"])
    await ok(await client.delete(f"{G}/sounds/{sid}"))
    assert store.get_sound(sid) is None and not path.exists()
    assert all(not r["options"] or r["enabled"] is False for r in store.list_reactions(GID) if r["kind"] == "sound")


async def test_sound_upload_limits(client, store):
    r = await client.post(f"{G}/sounds", data={"name": "x", "file": io.BytesIO(b"nope" * 100)})
    assert r.status == 400 and (await r.json())["field"] == "file"
    store.set_setting(GID, "sounds.max_mb", 0.01)
    r = await client.post(f"{G}/sounds", data={"name": "big", "file": io.BytesIO(wav_bytes(3))})
    assert r.status == 413


async def test_play_when_not_in_a_call(client, sounds, store):
    other = 1100000000000000002  # FakeController: not connected there
    row = await asyncio.to_thread(sounds.add, other, "boop", wav_bytes(1, 500), created_by=None)
    r = await client.post(f"/api/g/{other}/sounds/{row['id']}/play")
    assert r.status == 400 and "voice channel" in (await r.json())["error"]


async def test_role_and_channel_settings(client, store):
    await ok(await client.put(f"{G}/settings/admin.role_id", json={"value": str(USER)}))
    assert store.get_setting(GID, "admin.role_id") == USER
    assert (await client.put(f"{G}/settings/admin.channel_id", json={"value": "general"})).status == 400
    await ok(await client.put(f"{G}/settings/voices.preview_text", json={"value": " Hola "}))
    assert store.get_setting(GID, "voices.preview_text") == "Hola"
