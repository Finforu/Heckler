"""Dashboard development server with a fake bot (no Discord, no GPU).

    .venv/bin/python -m web.dev_server [--port 8799] [--token dev] [--data DIR]

Then open the printed login link. FakeController keeps a made-up bot state,
logs every control and publishes fake feed events into its own EventBus.
The editors run on a real Store + ReactionEngine in a throwaway directory
(--data, default a new temp dir), seeded with packs/base-en.yaml, a few demo gags and some
made-up people, a pending gag, a quota request, a sound and a voice.
Voices run on a VoiceLibrary with FakeVoiceModel (tones instead of speech,
no torch) and sounds on a real SoundLibrary.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import itertools
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events import EventBus  # noqa: E402
from web.server import BusLogHandler, login_url, start_dashboard  # noqa: E402

log = logging.getLogger("fakebot")

TOGGLES = ("autojoin", "userclone", "record", "transcripts")


class FakePrompt:
    def __init__(self, ref_text, samples):
        self.ref_text, self.samples = ref_text, samples

    def save(self, path):
        Path(path).write_text(json.dumps({"ref_text": self.ref_text, "samples": self.samples}))


def fake_load_prompt(path, map_location="cpu"):
    data = json.loads(Path(path).read_text())
    return FakePrompt(data["ref_text"], data["samples"])


class FakeVoiceModel:
    """Stands in for OmniVoice: "speech" is a short tone, pitched per voice."""
    device = "cpu"
    name_or_path = "fake/omnivoice"

    def create_voice_clone_prompt(self, ref_audio, ref_text=None):
        wave_, rate = ref_audio
        return FakePrompt(ref_text, int(wave_.shape[1]))

    def generate(self, text, language=None, num_step=None, voice_clone_prompt=None, instruct=None, speed=None):
        import numpy as np

        seed = voice_clone_prompt.samples if voice_clone_prompt is not None else len(instruct or "")
        freq = 160 + seed % 240
        t = np.arange(int(24000 * min(4.0, 0.3 + 0.06 * len(text)))) / 24000
        return [(0.2 * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)]


def fake_transcribe(audio16) -> str:
    return "hello this is a voice test"


def make_library(store, root: Path):
    """A VoiceLibrary on the fake model, files under root/voices and root/cache."""
    from voice_library import VoiceLibrary

    library = VoiceLibrary(store, root, language="en")
    library.attach(FakeVoiceModel(), fake_transcribe, load_prompt=fake_load_prompt)
    return library


class FakeController:
    """Implements the Controller protocol from docs/PLAN.md with in-memory state.
    With a VoiceLibrary, also the voice methods (preview, build_voice,
    transcribe_voice) and play_sound, like bot.py's Controller."""

    def __init__(self, bus: EventBus | None = None, bot_name: str = "Testbot", library=None, store=None) -> None:
        self.bus = bus
        self.store = store  # with a store, "transcripts" lives in its setting, like the real bot
        self.library = library
        self.builds: list[asyncio.Task] = []
        self.started = time.monotonic()
        self.bot_name = bot_name
        self.calls: list[tuple[str, dict]] = []
        self.tts_items: list[dict] = [
            {"kind": "pregen", "text": "Hi everyone", "voice": "bot"},
        ]
        self.tts_running: dict | None = {"kind": "build", "text": "voice: narrator"}
        self.guilds = {
            "1100000000000000001": {
                "id": "1100000000000000001", "name": "Game Night",
                "voice_channels": [
                    {"id": "1200000000000000001", "name": "General", "people": 3},
                    {"id": "1200000000000000002", "name": "Gaming", "people": 1},
                    {"id": "1200000000000000003", "name": "AFK", "people": 0},
                ],
                "text_channels": [
                    {"id": "1300000000000000001", "name": "general", "kind": "text"},
                    {"id": "1300000000000000002", "name": "bot-stuff", "kind": "text"},
                    {"id": "1200000000000000001", "name": "General", "kind": "voice"},
                ],
                "connected": "1200000000000000001",
                "toggles": {"autojoin": True, "userclone": False, "record": False, "transcripts": False},
                "reply_queue": 1, "timers": 2,
            },
            "1100000000000000002": {
                "id": "1100000000000000002", "name": "Test Lab",
                "voice_channels": [
                    {"id": "1200000000000000011", "name": "Lounge", "people": 0},
                    {"id": "1200000000000000012", "name": "Studio", "people": 2},
                ],
                "text_channels": [{"id": "1300000000000000011", "name": "lab", "kind": "text"}],
                "connected": None,
                "toggles": {"autojoin": False, "userclone": True, "record": False, "transcripts": False},
                "reply_queue": 0, "timers": 0,
            },
        }

    # -- Controller protocol

    def status(self) -> dict:
        guilds = []
        for g in self.guilds.values():
            channel = next((c for c in g["voice_channels"] if c["id"] == g["connected"]), None)
            guilds.append({
                "id": g["id"], "name": g["name"],
                "voice": ({"channel_id": channel["id"], "channel": channel["name"],
                           "people": channel["people"]} if channel else None),
                "voice_channels": [dict(c) for c in g["voice_channels"]],
                "text_channels": [dict(c) for c in g["text_channels"]],
                "toggles": {**g["toggles"], **({"transcripts": bool(self.store.get_setting(int(g["id"]), "transcripts.enabled"))}
                                                if self.store is not None else {})},
                "reply_queue": g["reply_queue"], "timers": g["timers"],
            })
        used = 4.4 + 0.3 * len(self.tts_items) + random.uniform(-0.05, 0.05)
        return {
            "bot": {"name": self.bot_name, "user": f"{self.bot_name}#1234", "connected": True,
                    "uptime_s": time.monotonic() - self.started, "stt_engine": "whisper", "time_zone": "Europe/Madrid",
                    "llm": {"provider": "ollama", "model": "llama3.2", "cloud": False, "service": "Ollama"}},
            "models": [
                {"name": "deepdml/faster-whisper-large-v3-turbo-ct2", "role": "stt",
                 "device": "cuda", "loaded": True},
                {"name": "k2-fsa/OmniVoice", "role": "tts", "device": "cuda", "loaded": True},
                {"name": "silero-vad", "role": "vad", "device": "cpu", "loaded": False},
                {"name": "llama3.2", "role": "llm", "device": "ollama", "loaded": True},
            ],
            "gpu": {"name": "RTX 3070 Ti", "used_gb": round(used, 2), "total_gb": 8.0},
            "tts_queue": {"pending": len(self.tts_items), "running": self.tts_running,
                          "items": [dict(i) for i in self.tts_items]},
            "guilds": guilds,
        }

    async def control(self, action: str, **params) -> str:
        self.calls.append((action, params))
        log.info("control %s %s", action, params)
        if action in ("reload", "restart"):
            return "Reloaded config" if action == "reload" else "Restarting…"
        if action not in ("join", "leave", "stop", "clear_queue", "say", "toggle"):
            raise ValueError(f"unknown action: {action}")
        g = self._guild(params.get("guild_id"))
        if action == "join":
            channel = next((c for c in g["voice_channels"] if c["id"] == str(params.get("channel_id"))), None)
            if channel is None:
                raise ValueError("unknown voice channel")
            g["connected"] = channel["id"]
            return f"Joined {channel['name']}"
        if action == "leave":
            if not g["connected"]:
                raise ValueError("not in a voice channel")
            g["connected"] = None
            return "Left the call"
        if action == "stop":
            self.tts_running = None
            return "Stopped"
        if action == "clear_queue":
            n, g["reply_queue"] = g["reply_queue"], 0
            return f"Cleared {n} queued replies"
        if action == "say":
            text = str(params.get("text", "")).strip()
            if not text:
                raise ValueError("nothing to say")
            if not g["connected"]:
                raise ValueError("not in a voice channel")
            self.tts_items.append({"kind": "reply", "text": text, "voice": "bot"})
            return "Queued"
        if action == "toggle":
            name = params.get("name")
            if name not in TOGGLES:
                raise ValueError(f"unknown toggle: {name}")
            g["toggles"][name] = bool(params.get("value"))
            if name == "transcripts":
                if self.store is not None:
                    self.store.set_setting(int(g["id"]), "transcripts.enabled", g["toggles"][name])
                if g["toggles"][name]:
                    log.info("Would post the transcripts notice in %s", g["name"])
            return f"{name} {'on' if g['toggles'][name] else 'off'}"
        raise ValueError(f"unknown action: {action}")

    # -- voices and sounds (only with a library, like the real bot)

    async def preview(self, text: str, voice_id: int | None = None) -> bytes:
        import soundfile as sf

        from voice_library import RATE, VoiceNotReady

        self.calls.append(("preview", {"text": text, "voice_id": voice_id}))
        if self.library is None:
            raise ValueError("Voice is off")
        try:
            audio = await asyncio.to_thread(self.library.speak, text, voice_id)
        except VoiceNotReady as e:
            raise ValueError(str(e)) from None
        buffer = io.BytesIO()
        sf.write(buffer, audio, RATE, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def build_voice(self, voice_id: int) -> str:
        self.calls.append(("build_voice", {"voice_id": voice_id}))
        if self.library is None:
            raise ValueError("Voice is off")
        if not self.library.request_rebuild(voice_id):
            raise ValueError("This voice has nothing to build from yet")

        async def run():
            await asyncio.sleep(0.3)  # "queued" for a moment, like behind live replies
            try:
                await asyncio.to_thread(self.library.build, voice_id)
            except Exception as e:
                log.warning("Fake build of voice %s failed: %s", voice_id, e)

        self.builds.append(asyncio.get_running_loop().create_task(run()))
        return "Build queued"

    async def transcribe_voice(self, voice_id: int) -> str:
        self.calls.append(("transcribe_voice", {"voice_id": voice_id}))
        if self.library is None:
            raise ValueError("Voice is off")
        return await asyncio.to_thread(self.library.transcribe_ref, voice_id, force=True)

    async def play_sound(self, guild_id: int, sound_id: int) -> str:
        self.calls.append(("play_sound", {"guild_id": guild_id, "sound_id": sound_id}))
        g = self._guild(guild_id)
        if not g["connected"]:
            raise ValueError("Not in a voice channel there")
        return "Playing"

    def _guild(self, guild_id) -> dict:
        g = self.guilds.get(str(guild_id))
        if g is None:
            raise ValueError("unknown server")
        return g

    # -- fake activity

    async def run_fake_activity(self) -> None:
        """Publishes made-up utterances (and their outcomes) forever."""
        users = [("Alice", 301), ("Bruno", 302), ("Carla", 303), ("Dani", 304)]
        lines = [
            ("hi everyone", None, "no match"),
            ("hey bot, play something", {"kind": "command", "name": "play"}, "queued"),
            ("ugh, it's monday again", {"kind": "gag", "name": "monday"}, "queued"),
            ("I need more coffee", {"kind": "gag", "name": "coffee"}, "queued"),
            ("drumroll please", {"kind": "gag", "name": "drumroll"}, "queued"),
            ("did anyone watch the match yesterday", None, "no match"),
            ("hey bot, set a timer for five minutes", {"kind": "command", "name": "timer"}, "queued"),
        ]
        for n in itertools.count():
            await asyncio.sleep(random.uniform(1.0, 3.5))
            g = random.choice(list(self.guilds.values()))
            user, uid = random.choice(users)
            text, matched, outcome = random.choice(lines)
            event = {
                "type": "feed", "id": self.bus.next_id(), "guild_id": int(g["id"]), "guild": g["name"],
                "user_id": uid, "user": user, "text": text, "lang": "es",
                "duration": round(random.uniform(0.6, 3.0), 1), "matched": matched,
                "voice": (random.choice(["bot", "narrator", "@speaker"]) if matched else None),
                "outcome": outcome,
            }
            self.bus.publish(event)
            if matched:
                asyncio.get_running_loop().call_later(
                    random.uniform(0.5, 2.5), self.bus.publish,
                    {"type": "feed_update", "id": event["id"],
                     "outcome": random.choice(["played", "played", "played", "skipped: stale"])})
            if self.tts_items and random.random() < 0.4:
                self.tts_items.pop(0)
            if random.random() < 0.3:
                self.tts_items.append({"kind": random.choice(["reply", "preview", "build"]),
                                       "text": text, "voice": "bot"})
            g["reply_queue"] = random.randint(0, 3)
            if n % 7 == 6:
                log.warning("Fake warning #%d: reply skipped, queue was stale", n)
            if n % 23 == 22:
                try:
                    raise RuntimeError("fake failure")
                except RuntimeError:
                    log.exception("Fake error while playing")


def tone_file(path: Path, seconds: float, freq: float = 220.0) -> Path:
    """A synthetic "voice": a warbling tone with pauses, so speech detection finds lines."""
    import numpy as np
    import soundfile as sf

    rate = 24000
    t = np.arange(int(seconds * rate)) / rate
    audio = 0.3 * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    audio[(t % 3.0) > 2.2] = 0  # a pause every 3 s
    sf.write(path, audio.astype(np.float32), rate)
    return path


def seed_history(store, guild_id: int, days: int = 30, count: int = 900) -> None:
    """Made-up transcribed sentences for the Stats and History tabs."""
    from datetime import datetime, timedelta

    rng = random.Random(5)
    users = [(301, "Alice"), (302, "Bruno"), (303, "Carla"), (304, "Dani")]
    word = [r for r in store.list_reactions(guild_id)
            if r["status"] == "approved" and any(t["type"] in ("phrase", "swap", "command") for t in r["triggers"])]
    favorites = word[:8]  # a few popular ones, so "never fired" has something to show
    now = datetime.now().astimezone()
    stamps = sorted(now - timedelta(seconds=rng.uniform(0, days * 86400)) for _ in range(count))
    with store.batch():
        for t in stamps:
            if rng.random() < 0.55 and 2 <= t.hour < 15:
                continue  # quieter at night and in the morning
            uid, name = rng.choice(users)
            roll = rng.random()
            reaction = rng.choice(favorites) if roll < 0.45 else None
            if reaction is None:
                outcome, text = ("ignored (echo)", "Yes?") if roll > 0.95 else ("no match", "did anyone watch the match")
            else:
                outcome = rng.choice(["matched"] * 6 + ["skipped (cooldown)"] * 2 + ["skipped (chance)"])
                text = f"something about {reaction['name']}"
            store.log_event(guild_id, outcome, user_id=uid, user_name=name, text=text,
                            reaction_id=reaction["id"] if reaction else None,
                            details={"matched": {"kind": reaction["kind"], "name": reaction["name"]} if reaction else None,
                                     "lang": "en", "duration": round(rng.uniform(0.5, 3), 1)},
                            time=t.isoformat(timespec="seconds"))


DEMO_GAGS = [  # made up for the dev server; nothing personal
    ("monday", ["monday", "mondays"], ["Mondays should be illegal.", "Not Monday again."]),
    ("coffee", ["coffee"], ["Coffee? Make it a double, {name}."]),
    ("bug", ["bug", "it's a bug"], ["It's not a bug, it's a feature.", "Did you try turning it off and on again?"]),
    ("lag", ["lag", "lagging"], ["Blame the router.", "Lag is just suspense, {name}."]),
]


def seed_store(data_dir: Path, guild_ids: list[str]):
    """A Store + ReactionEngine + VoiceLibrary + SoundLibrary in data_dir with
    demo content (synthetic only)."""
    import packs
    from reactions import ReactionEngine
    from sound_library import SoundLibrary
    from store import Store

    store = Store(data_dir / "bot.db")
    engine = ReactionEngine(store)
    library = make_library(store, data_dir / "data")
    sounds = SoundLibrary(store, data_dir / "data")
    if store.list_reactions(int(guild_ids[0])):
        return store, engine, library, sounds  # already seeded (--data reused)
    for i, gid in enumerate(map(int, guild_ids)):
        packs.import_pack(store, gid, packs.ROOT / "packs" / "base-en.yaml", base_dir=data_dir)
        store.set_setting(gid, "language", "en")
        for uid, display, nick in ((301, "Alice", None), (302, "Bruno 🎮", "Bruno"), (303, "Carla", None),
                                   (304, "Dani", None)):
            store.set_person(gid, uid, display_name=display, nickname=nick, last_seen="2026-10-06T19:00:00-03:00")
        beep = sounds.add(gid, "rimshot", tone_file(data_dir / "rimshot-src.wav", 0.6, 880), created_by=302)
        sounds.add(gid, "airhorn", tone_file(data_dir / "airhorn-src.wav", 1.2, 520), created_by=None)
        store.add_voice("narrator", guild_id=gid, kind="designed", instruct="male, low pitch", status="ready")
        if i == 0:
            vid = store.add_voice("pirate", guild_id=gid, kind="clone")
            library.ingest(vid, tone_file(data_dir / "pirate-src.wav", 20, 180))
            for name, phrases, replies in DEMO_GAGS:
                store.add_reaction(gid, "gag", name, [{"type": "phrase", "phrases": phrases}],
                                   [[{"type": "say", "text": r}] for r in replies])
            store.add_reaction(gid, "gag", "drumroll", [{"type": "phrase", "phrases": ["drumroll", "drum roll"]}],
                               [[{"type": "say", "text": "And the winner is... {name}!"},
                                 {"type": "sound", "sound_id": beep["id"]}]], chance=0.5)
            store.add_reaction(gid, "gag", "pizza", [{"type": "phrase", "phrases": ["pizza"]}],
                               [[{"type": "say", "text": "Did someone say pizza, {name}?"}]],
                               status="pending", created_by=302)
            store.request_quota(gid, 302, "gags", 5, "I have so many more ideas")
    bot_src = tone_file(data_dir / "bot-voice.wav", 9, 260)
    bot_id = library.ensure_bot_voice(bot_src, name="bot voice")
    library.transcribe_ref(bot_id)
    library.build(bot_id)
    store.set_consent(301, "accepted")
    store.set_consent(302, "pending")
    store.set_consent(301, "accepted", purpose="transcripts")
    store.set_consent(303, "declined", purpose="transcripts")
    seed_history(store, int(guild_ids[0]))
    import numpy as np  # Alice's own voice, from "calls"
    for k in range(5):
        t = np.arange(2 * 24000) / 24000
        library.update_speaker(301, (0.3 * np.sin(2 * np.pi * (200 + k * 7) * t)).astype(np.float32),
                               f"this is sentence number {k}", name="Alice")
    return store, engine, library, sounds


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--token", default="dev")
    parser.add_argument("--data", help="directory for the throwaway database and media (default: a temp dir)")
    parser.add_argument("--bot-name", default="Testbot", help="the instance's bot name shown in the dashboard")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bus = EventBus()
    logging.getLogger().addHandler(BusLogHandler(bus, asyncio.get_running_loop()))
    controller = FakeController(bus, bot_name=args.bot_name)
    data_dir = Path(args.data) if args.data else Path(tempfile.mkdtemp(prefix="dash-dev-"))
    data_dir.mkdir(parents=True, exist_ok=True)
    store, engine, library, sounds = seed_store(data_dir, list(controller.guilds))
    controller.library = library
    controller.store = store
    for voice_id in library.pending_builds():
        controller.build_voice(voice_id)
    print("Data:", data_dir, flush=True)
    runner = await start_dashboard(controller, bus, host=args.host, port=args.port, token=args.token,
                                   store=store, engine=engine, library=library, sounds=sounds, base_dir=data_dir)
    print("Login:", login_url(args.host, args.port, args.token), flush=True)
    try:
        await controller.run_fake_activity()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
