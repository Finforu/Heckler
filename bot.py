import asyncio
import contextvars
import io
import logging
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

# Must be set before torch touches CUDA: less fragmentation, so the memory
# OmniVoice frees can actually be reused (by Whisper, when STT_ENGINE=whisper).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import discord
import soundfile as sf
from discord.ext import commands, voice_recv
from dotenv import load_dotenv

import content_commands
import dave
import i18n
import packs
import timers
from events import bus
from listener import Utterance, UtteranceSink
from reactions import Match, ReactionEngine
from recorder import SessionRecorder
from store import Store
from transcripts import Transcripts
from tts_queue import TTSQueue
from transcriber import (DEFAULT_MODEL, HybridTranscriber, ParakeetTranscriber, TranscriptionBatcher,
                         WhisperTranscriber, normalize, pcm_to_wav)
from voice_commands import CommandListener, bot_name
from sound_library import SoundLibrary
from voice_library import RATE as VOICE_RATE, VoiceLibrary, VoiceNotReady, to_discord_pcm
from voices import discord_pcm_to_mono

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

transcript_files = Transcripts(ROOT / "transcripts")
RECORDINGS_DIR = ROOT / "recordings"
TESTDATA_DIR = ROOT / "testdata"
# The bot's own voice: cloned from a recording (~10 s of clear speech), or,
# without one, made from a description (OmniVoice voice design: gender, age,
# pitch, accent... see omnivoice/utils/voice_design.py).
VOICE_FILE = ROOT / os.environ["BOT_VOICE_FILE"] if os.getenv("BOT_VOICE_FILE", "").strip() else None
BOT_VOICE_DESCRIPTION = os.getenv("BOT_VOICE_DESCRIPTION", "").strip() or "male, young adult, moderate pitch"
POST_TRANSCRIPTS = os.getenv("POST_TRANSCRIPTS", "0") == "1"
SAVE_AUDIO = os.getenv("SAVE_AUDIO", "0") == "1"
# The bot talking needs OmniVoice, ~2.3 GB VRAM. VOICE=0 skips loading it:
# the bot only listens.
VOICE = os.getenv("VOICE", "1") == "1"
# Pause that ends an utterance. Shorter answers sooner but may split a
# sentence at a hesitation.
END_SILENCE_S = float(os.getenv("END_SILENCE_S", "0.6"))
# Speech-to-text (all forced to the main language, the first of WHISPER_LANGUAGES):
#   "whisper"  Whisper on the GPU (~2.3 GB), ~205 ms a clip. Default: on a real
#              recorded session it was both the most accurate and the fastest
#              for short commands and gags, and the hotword helps with the bot's name.
#   "hybrid"   Parakeet (CPU, ~130 ms, batches people talking at once), with
#              doubtful or empty results re-done by Whisper.
#   "parakeet" Parakeet only: no GPU memory, but drifts into other languages
#              on short or noisy clips.
STT_ENGINE = os.getenv("STT_ENGINE", "whisper").lower()
# Where Parakeet runs. "gpu" is ~3x faster (46 vs 135 ms a sentence) but takes
# 3.4 GB; with the voice model that leaves no room for Whisper, so "gpu" turns
# the hybrid's Whisper re-check off (and the language drift comes back).
PARAKEET_DEVICE = os.getenv("PARAKEET_DEVICE", "cpu").lower()
# The bot's replies wait their turn instead of being dropped, but one that
# couldn't start within this long is skipped: a late joke is worse than none.
REPLY_MAX_AGE_S = 5.0
# Something heard this soon after the bot said something similar is its own
# voice leaking through someone's mic; never answer it (it would loop).
ECHO_WINDOW_S = 4.0
ECHO_SIMILARITY = 0.6
# What was heard and what fired, kept for the dashboard's stats.
EVENT_HISTORY_DAYS = 90

log = logging.getLogger("dc-bot")
# voice-recv logs every voice gateway payload and RTCP report at INFO.
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
# "N packets were lost": ordinary network loss, already concealed by the decoder.
logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.ERROR)

dave.install()

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

LANGUAGES = [lang.strip() for lang in os.getenv("WHISPER_LANGUAGES", "es,en").split(",") if lang.strip()]

# The hotword nudges Whisper toward spelling the bot's name consistently.
def make_whisper() -> WhisperTranscriber:
    return WhisperTranscriber(os.getenv("WHISPER_MODEL", DEFAULT_MODEL), LANGUAGES[:1],
                              hotwords=bot_name())


if STT_ENGINE == "parakeet" or (STT_ENGINE == "hybrid" and PARAKEET_DEVICE == "gpu"):
    transcriber = ParakeetTranscriber(device=PARAKEET_DEVICE)
elif STT_ENGINE == "hybrid":
    transcriber = HybridTranscriber(ParakeetTranscriber(), make_whisper())
else:
    transcriber = make_whisper()
stt = TranscriptionBatcher(transcriber)
# Each speaker's own utterances are handled in order ("Hey Heckler..." must
# come before "...vete"); different speakers are handled concurrently.
speaker_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
voice_commands = CommandListener()

voice_model = None  # OmniVoice, once loaded (VOICE=1)
tts = TTSQueue()


async def generate(kind: str, fn, *args, text: str = "", voice: str = ""):
    """Run a speech-generation call on the TTS thread (see tts_queue.PRIORITY)."""
    return await asyncio.wrap_future(tts.submit(kind, fn, *args, text=text, voice=voice))


# Everything the bot says and reacts to (gags, commands, greetings, stock
# replies...) lives in data/bot.db, edited from the dashboard. Cooldowns and
# the like are settings there too.
store = Store(ROOT / "data" / "bot.db")
engine = ReactionEngine(store)
# Saved voices (the bot's own, uploaded/designed ones, people's own): each
# built once, and every line generated once (cached on disk, data/).
library = VoiceLibrary(store, ROOT / "data", language=LANGUAGES[0] if LANGUAGES else None)
sounds = SoundLibrary(store, ROOT / "data")  # soundboard clips
# !userclone: per guild, gags answer in the speaker's own voice instead of
# the bot's (once they've consented and their voice is built). Off by default.
userclone: dict[int, bool] = {}
# !record: per guild, the test recording in progress (one person's speech).
recorders: dict[int, SessionRecorder] = {}
restart_requested = False


class Bot(commands.Bot):
    async def close(self) -> None:
        if dashboard_runner is not None:
            await dashboard_runner.cleanup()  # closes open dashboard tabs' connections
        await super().close()

    async def setup_hook(self) -> None:
        global voice_model
        await start_dashboard()  # first, so the models can be watched loading
        # The voice model goes on the GPU before Whisper: building it needs a
        # burst of ~2.2 GB that no longer fits once Whisper is there.
        if isinstance(transcriber, HybridTranscriber):
            parakeet, whisper = transcriber.parakeet, transcriber.whisper
        elif isinstance(transcriber, ParakeetTranscriber):
            parakeet, whisper = transcriber, None
        else:
            parakeet, whisper = None, transcriber
        if parakeet is not None:
            await asyncio.to_thread(parakeet.load)
        if VOICE:
            if VOICE_FILE is not None and not VOICE_FILE.is_file():
                log.warning("BOT_VOICE_FILE %s not found: using a designed voice instead", VOICE_FILE)
            voice_model = await asyncio.to_thread(load_voice_model)
            transcriber.on_oom = free_gpu_memory  # only used by Whisper
        if whisper is not None:
            await asyncio.to_thread(whisper.load)
        for voice_id in library.pending_builds():  # interrupted or waiting since last run
            queue_build(voice_id)
        self.add_dynamic_items(ConsentButton)
        # /gag, /sound, /voices: what people make themselves, within their quotas.
        await content_commands.setup(self, SimpleNamespace(
            store=store, engine=engine, library=library, sounds=sounds,
            voice_on=lambda: voice_model is not None, queue_build=queue_build,
            say_text=say_text, play_sound=play_sound))
        # Edited content (dashboard, /gag...) gets its new fixed lines generated.
        # The store may call this from another thread.
        loop = asyncio.get_running_loop()
        store.on_change(lambda table, guild_id: table == "reactions" and loop.call_soon_threadsafe(schedule_pregenerate))


bot = Bot(command_prefix="!", intents=intents)


VOICE_MODEL = "k2-fsa/OmniVoice"


def load_voice_model():
    """Load OmniVoice and make sure the bot's own voice is built. Blocking.
    Runs before Whisper loads: building a voice needs a burst of ~2.2 GB
    that no longer fits once Whisper is there."""
    import torch  # only when the voice is on
    from omnivoice import OmniVoice

    cuda = torch.cuda.is_available()
    model = OmniVoice.from_pretrained(VOICE_MODEL, device_map="cuda:0" if cuda else "cpu",
                                      dtype=torch.float16 if cuda else torch.float32)
    library.attach(model, transcribe_reference, model_name=VOICE_MODEL, free_memory=free_gpu_memory)
    if VOICE_FILE is not None and VOICE_FILE.is_file():
        voice_id = library.ensure_bot_voice(VOICE_FILE, name=bot_name())
    else:
        voice_id = ensure_designed_bot_voice()
    if library.build_due(voice_id):
        library.build(voice_id)
    log.info("Loaded %s; the bot's voice is ready", VOICE_MODEL)
    if cuda:
        free, total = torch.cuda.mem_get_info()
        log.info("GPU memory in use: %.1f of %.1f GB", (total - free) / 1e9, total / 1e9)
    return model


def ensure_designed_bot_voice() -> int:
    """With no BOT_VOICE_FILE, the bot speaks in a voice made from a
    description (BOT_VOICE_DESCRIPTION), so a fresh install talks without
    any recording. Changing the description remakes it."""
    instruct = BOT_VOICE_DESCRIPTION
    voice_id = library.bot_voice_id()
    row = store.get_voice(voice_id) if voice_id is not None else None
    if row is None or row["kind"] != "designed":
        voice_id = store.add_voice(bot_name(), guild_id=None, kind="designed", instruct=instruct, status="draft")
        store.set_setting(0, "voice.bot", voice_id)
    elif row["instruct"] != instruct:
        store.update_voice(voice_id, instruct=instruct, status="draft")
        library.clear_cache(voice_id)
    return voice_id


def free_gpu_memory() -> None:
    """Hand PyTorch's cached GPU memory back so Whisper (a separate CUDA
    allocator) can use it. Both share one card."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_reference(audio) -> str:
    """Transcript of a voice sample, only needed when a voice is (re)built
    (e.g. after BOT_VOICE_FILE changes). It has to be accurate: a
    transcript that doesn't match the audio makes every cloned line start with
    leftovers of the sample ("¡Hombre!..."). So Whisper, on the CPU (a few
    seconds) since the GPU is needed for building the voice right then."""
    engine = WhisperTranscriber(os.getenv("WHISPER_MODEL", DEFAULT_MODEL), LANGUAGES[:1], cpu_only=True)
    engine.load()
    return engine.transcribe_audio(audio)


SLASH_INVITE = "https://discord.com/oauth2/authorize?client_id={}&scope=bot+applications.commands&permissions=3214336"


async def sync_slash_commands(guild) -> None:
    """Per-server sync: the commands show up in the / picker right away
    (a global sync can take up to an hour)."""
    bot.tree.copy_global_to(guild=guild)
    try:
        await bot.tree.sync(guild=guild)
    except discord.Forbidden:
        log.warning("No slash commands in %s: re-invite the bot with %s", guild, SLASH_INVITE.format(bot.user.id))


@bot.event
async def on_ready():
    print(f'Bot connected as {bot.user} (ID: {bot.user.id})')
    dave.ignore_user = ignore_bots
    pruned = store.prune_events((datetime.now() - timedelta(days=EVENT_HISTORY_DAYS)).astimezone().isoformat(timespec="seconds"))
    if pruned:
        log.info("Dropped %d history events older than %d days", pruned, EVENT_HISTORY_DAYS)
    import_content(bot.guilds)
    for guild in bot.guilds:
        await sync_slash_commands(guild)
        await auto_join(guild)
    schedule_pregenerate()


@bot.event
async def on_guild_join(guild):
    import_content([guild])
    await sync_slash_commands(guild)
    schedule_pregenerate()


def import_content(guilds) -> None:
    """A server seen for the first time gets its starter content: the pack in
    setting content.starter_pack, or this install's old gags.py & co. if they
    still exist (servers that already have content are left alone)."""
    for guild in guilds:
        result = packs.seed_guild(store, guild.id)
        if not result.startswith("kept"):
            log.info("Server %s: %s", guild.name, result)


def can_post(channel) -> bool:
    return channel.permissions_for(channel.guild.me).send_messages


def lang_of(where) -> str:
    """The language of a server (a guild, a ctx, an interaction, or an id)."""
    guild = getattr(where, "guild", where)
    guild_id = guild if isinstance(guild, int) else getattr(guild, "id", None)
    return i18n.guild_language(store, guild_id)


def tr(where, key: str, **values) -> str:
    """A message in that server's language (locales/<lang>/bot.json)."""
    return i18n.t(key, lang_of(where), bot=bot_name(), **values)


async def reply(ctx, key: str, **values) -> None:
    """Answer a command with the message `key` (see tr). Slash commands
    answer privately (only the person who used it sees it); ! commands log
    instead of failing where the bot can't post."""
    text = tr(ctx, key, **values)
    if ctx.interaction is not None:
        await ctx.send(text, ephemeral=True)
        return
    if not can_post(ctx.channel):
        log.warning("Can't reply in #%s (missing Send Messages): %s", ctx.channel, text)
        return
    await ctx.send(text)


async def author_is_admin(ctx) -> bool:
    """The bot owner, anyone with Manage Server, or the admin.role_id role."""
    member = ctx.author
    return content_commands.is_admin(
        store, ctx.guild.id if ctx.guild else None, is_owner=await bot.is_owner(member),
        manage_guild=bool(getattr(getattr(member, "guild_permissions", None), "manage_guild", False)),
        role_ids=[r.id for r in getattr(member, "roles", [])])


async def handle_utterance(u: Utterance, voice_channel, text_channel) -> None:
    async with speaker_locks[u.user_id]:
        await _handle_utterance(u, voice_channel, text_channel)


async def _handle_utterance(u: Utterance, voice_channel, text_channel) -> None:
    guild = voice_channel.guild
    try:
        heard = await asyncio.wrap_future(stt.submit(u.pcm, language=lang_of(guild)))
    except Exception:
        log.exception("Transcription failed for %s", u.user_name)
        return
    text, lang = heard
    confidence = getattr(heard, "confidence", None)
    recorder = recorders.get(voice_channel.guild.id)
    if recorder is None or recorder.user_id != u.user_id:
        recorder = None
    if not text:
        if recorder:
            recorder.save(u, "", lang, outcome="nothing heard")
        if getattr(heard, "dropped", None) == "hallucination":
            # Shown live (not saved) so the filter can be seen working.
            bus.publish({"type": "feed", "id": bus.next_id(), "time": u.started_at.isoformat(timespec="seconds"),
                         "guild_id": str(guild.id), "guild": guild.name, "user_id": str(u.user_id),
                         "user": u.user_name, "text": heard.heard, "lang": lang, "duration": round(u.duration, 2),
                         "confidence": confidence, "matched": None, "reply": None, "voice": None,
                         "outcome": "ignored (made up)"})
        return

    log.info("[%s] %s (%s, %.1fs): %s", u.started_at.strftime("%H:%M:%S"), u.user_name, lang, u.duration, text)
    entry = {
        "type": "feed", "id": bus.next_id(), "time": u.started_at.isoformat(timespec="seconds"),
        "guild_id": str(voice_channel.guild.id), "guild": voice_channel.guild.name,
        "user_id": str(u.user_id), "user": u.user_name, "text": text, "lang": lang,
        "duration": round(u.duration, 2), "confidence": confidence,
        "matched": None, "reply": None, "voice": None, "outcome": "no match",
    }
    current_feed.set(entry)

    # Words are only written down (transcripts, history text, the debug
    # options) in servers with transcripts on, for people who agreed.
    keep_text = await may_keep_text(u.user_id, voice_channel)
    if keep_text:
        transcript_files.append({
            "time": u.started_at.isoformat(timespec="seconds"),
            "guild_id": guild.id, "guild": guild.name, "channel": voice_channel.name,
            "user_id": u.user_id, "user": u.user_name, "language": lang,
            "duration": round(u.duration, 2), "text": text,
        })

    if SAVE_AUDIO and keep_text:
        RECORDINGS_DIR.mkdir(exist_ok=True)
        (RECORDINGS_DIR / f"{u.started_at:%Y%m%d-%H%M%S}-{u.user_id}.wav").write_bytes(pcm_to_wav(u.pcm))

    if POST_TRANSCRIPTS and keep_text and text_channel is not None:
        try:
            await text_channel.send(f"**{u.user_name}**: {text}", allowed_mentions=discord.AllowedMentions.none())
        except discord.HTTPException as e:
            log.warning("Couldn't post transcript in #%s: %s", text_channel, e)

    if voice_model is not None:
        try:
            await collect_voice(u, text, voice_channel)
        except Exception:  # a side job: never in the way of answering
            log.exception("Collecting %s's voice failed", u.user_name)

    if is_echo(voice_channel.guild.id, text):
        log.info("Ignoring %s: sounds like the bot's own voice (echo)", u.user_name)
        if recorder:
            recorder.save(u, text, lang, outcome="ignored as echo")
        bus.publish({**entry, "outcome": "ignored (echo)"})
        return

    # A sentence the speech-to-text was unsure of (mumbled, noisy, cut off)
    # sets nothing off: a misheard word firing a gag is worse than a missed one.
    min_confidence = store.get_setting(guild.id, "stt.min_confidence")
    if confidence is not None and min_confidence is not None and confidence < float(min_confidence):
        log.info("Not reacting to %s: unsure transcript (confidence %.2f < %.2f): %s",
                 u.user_name, confidence, float(min_confidence), text)
        if recorder:
            recorder.save(u, text, lang, outcome="unsure")
        bus.publish({**entry, "outcome": "ignored (unsure)"})
        store.log_event(guild.id, "ignored (unsure)", user_id=u.user_id, user_name=u.user_name,
                        text=text if keep_text else None, details={"lang": lang, "confidence": round(confidence, 3)})
        return

    name = speaker_name(guild.id, u.user_id, u.user_name)
    gag = engine.match(guild.id, text, u.user_id, u.user_name, name) if voice_model is not None else None
    command = voice_commands.feed(u.user_id, text)  # the wake word, and what was said after it
    order = None  # the command reaction for that request
    if command and command.action != "wake":
        order = engine.for_command(guild.id, command.request, u.user_id, u.user_name, name)
    if recorder:
        label = "wake" if command and command.action == "wake" else (order.reaction["name"] if order else None)
        recorder.save(u, text, lang, command=label, command_request=command.request if command else None,
                      gag=gag.text if gag else None)

    # A gag that starts with the bot's name ("Heckler, nice coffee") is a
    # gag, not a command the bot didn't understand.
    if command and not order and gag:
        command = None
    if command:
        entry["matched"] = {"kind": "command", "name": order.reaction["name"] if order else
                            ("wake" if command.action == "wake" else "not understood")}
    elif gag:
        entry["matched"] = {"kind": gag.reaction["kind"], "name": gag.reaction["name"]}
        entry["reply"] = gag.text or None
    chosen = order if command else gag
    blocked = chosen.blocked if chosen else None
    entry["outcome"] = f"skipped ({blocked})" if blocked else ("matched" if entry["matched"] else "no match")
    bus.publish(entry)
    # History for the dashboard's stats (the text only if events.save_text).
    store.log_event(guild.id, entry["outcome"], user_id=u.user_id, user_name=u.user_name,
                    text=text if keep_text else None,
                    reaction_id=chosen.reaction["id"] if chosen else None,
                    details={"matched": entry["matched"], "lang": lang, "duration": entry["duration"],
                             "confidence": round(confidence, 3) if confidence is not None else None})

    who = {"user_id": u.user_id, "display_name": u.user_name}
    if command and command.action == "wake":
        await respond(guild, "wake", **who)  # "¿Más trabajo?"
    elif command and order is None:
        await respond(guild, "unknown", **who)  # "¿Qué? No entendí."
    elif command:
        await perform(order, guild, text=text, **who)
    elif gag:
        await perform(gag, guild, text=text, **who)


# Which of a person's display names / nicknames we've stored this run.
seen_people: set[tuple[int, int]] = set()


def speaker_name(guild_id: int, user_id: int, display_name: str) -> str:
    """What the bot calls them (their nickname). The first time they're heard
    this run, their display name is saved too, for the dashboard."""
    if (guild_id, user_id) not in seen_people:
        seen_people.add((guild_id, user_id))
        store.set_person(guild_id, user_id, display_name=display_name, last_seen=datetime.now().isoformat(timespec="seconds"))
    return store.nickname_for(guild_id, user_id, display_name)


async def respond(guild, event: str, *, user_id: int | None = None, display_name: str = "",
                  values: dict | None = None) -> list[asyncio.Future]:
    """React to an event (wake, ack, hello, timer_ring...): the person's own
    reaction if they have one, otherwise the server's."""
    name = speaker_name(guild.id, user_id, display_name) if user_id else ""
    m = engine.for_event(guild.id, event, user_id, display_name, name, values)
    return await perform(m, guild, user_id=user_id, display_name=display_name)


async def perform(m: Match | None, guild, *, user_id: int | None = None, display_name: str = "",
                  text: str = "") -> list[asyncio.Future]:
    """Carry out a reaction's steps in order: say each line in its voice,
    play sounds, run built-in actions. Returns the queued lines' "played"
    futures."""
    if m is None or m.blocked or guild.voice_client is None:
        return []
    store.record_use(m.reaction["id"])
    played = []
    for step in m.steps:
        if step["type"] == "say":
            started = time.monotonic()
            line = await speech(step["text"], step.get("voice_id"), guild, user_id, display_name,
                                as_speaker=m.reaction["kind"] == "gag")
            if line is None:
                feed_update(outcome="failed")
                continue
            pcm, voice = line
            log.info("%s %r in %s's voice (%.2fs): %s", m.reaction["kind"], m.reaction["name"], voice,
                     time.monotonic() - started, step["text"])
            feed_update(voice=voice)
            played.append(say(guild, step["text"], pcm))
        elif step["type"] == "builtin":
            await run_builtin(step["action"], guild, user_id, display_name, text)
        elif step["type"] == "sound":
            try:
                played.append(await play_sound(guild, step["sound_id"]))
            except Exception as e:
                log.warning("Couldn't play sound %s (reaction %r): %s", step.get("sound_id"), m.reaction["name"], e)
        else:
            log.warning("%s steps aren't supported (reaction %r)", step["type"], m.reaction["name"])
    return played


async def play_sound(guild, sound_id: int) -> asyncio.Future:
    """Queue a soundboard clip like a reply, so skits keep their order."""
    if sounds is None:
        raise ValueError("Sounds aren't available")
    row = store.get_sound(int(sound_id))
    if row is None or not row["enabled"]:
        raise ValueError(f"No sound {sound_id}")
    pcm = await asyncio.to_thread(sounds.pcm, row["id"])
    return say(guild, f"[{row['name']}]", pcm)


async def speech(text: str, voice_id, guild, user_id: int | None, display_name: str,
                 *, as_speaker: bool = False) -> tuple[bytes, str] | None:
    """`text` as Discord PCM, and whose voice it's in. voice_id: a saved
    voice; None = the bot's; "@speaker" = the person's own (once they've
    consented and their voice is built). With as_speaker (gags), lines with no
    voice of their own are in the person's voice too while !userclone is on.
    Anything unavailable falls back to the bot's voice."""
    if voice_model is None:
        return None
    if voice_id == "@speaker" or (as_speaker and voice_id is None and userclone.get(guild.id)):
        own = library.speaker_voice(guild.id, user_id) if user_id else None
        voice_id = own["id"] if own else None
    elif isinstance(voice_id, str):  # a digit string from the dashboard
        voice_id = int(voice_id) if voice_id.isdigit() else None
    row = store.get_voice(voice_id) if voice_id is not None else None
    if row is None:
        voice_id = None
    label = row["name"] if row else bot_name()
    try:
        language = lang_of(guild)
        audio = library.cached(text, voice_id, language=language)  # most lines were said before: no queue, no GPU
        if audio is None:
            audio = await generate("reply", partial(library.speak, language=language), text, voice_id,
                                   text=text, voice=label)
        return to_discord_pcm(audio), label
    except VoiceNotReady as e:
        if voice_id is None:
            log.warning("The bot's voice isn't ready: %s", e)
            return None
        log.info("Voice %s isn't ready (%s); using the bot's", label, e)
        return await speech(text, None, guild, user_id, display_name)
    except Exception:
        log.exception("Speech generation failed for %r in %s's voice", text, label)
        return None


async def collect_voice(u: Utterance, text: str, voice_channel) -> None:
    """Keep this sentence for the speaker's own voice, only if they agreed.
    Where !userclone is on, people who never decided are asked (once)."""
    consent = store.get_consent(u.user_id, purpose="voice")
    if consent is None:
        if userclone.get(voice_channel.guild.id):
            await ask_consent(u.user_id, voice_channel, "voice")
        return
    if consent["status"] != "accepted":
        return
    audio = await asyncio.to_thread(discord_pcm_to_mono, u.pcm)
    if await asyncio.to_thread(library.update_speaker, u.user_id, audio, text, name=u.user_name):
        own = library.speaker_voice(None, u.user_id, ready_only=False)
        if own:
            log.info("Enough of %s's voice heard: building it", u.user_name)
            queue_build(own["id"])


def queue_build(voice_id: int) -> None:
    """Build (encode) a voice on the TTS thread, behind live replies."""
    row = store.get_voice(voice_id)
    name = row["name"] if row else str(voice_id)
    future = tts.submit("build", library.build, voice_id, text="(build)", voice=name)

    def done(f) -> None:
        if f.exception():
            log.warning("Building voice %s failed: %s", name, f.exception())
        else:
            log.info("Voice %s is ready", name)

    future.add_done_callback(done)


async def run_builtin(action: str, guild, user_id: int | None, display_name: str, text: str) -> None:
    """The actions only code can do, set off by command reactions."""
    vc = guild.voice_client
    if vc is None:
        return
    who = {"user_id": user_id, "display_name": display_name}
    if action == "leave":
        clear_replies(guild)
        played = await respond(guild, "leave", **who)  # "Entonces, allá voy."
        if played:
            try:
                await asyncio.wait_for(asyncio.gather(*played), timeout=10)
            except asyncio.TimeoutError:
                pass
        await disconnect(guild, pause_autojoin=True)
    elif action == "timer":
        parsed = timers.parse_timer(text, lang_of(guild))
        member = guild.get_member(user_id) if user_id else None
        if parsed is None or member is None:
            await respond(guild, "unknown", **who)
            return
        seconds, said, message = parsed
        start_timer(guild, seconds, said, message, member)
        await respond(guild, "timer_set", **who, values={"said": said})  # "OK, I'll remind you in 5 minutes."
    elif action == "stop":
        clear_replies(guild)
        if is_busy(vc):
            stop_sound(vc)
            await respond(guild, "ack", **who)
    else:
        log.warning("Unknown built-in action %r", action)


_pregen_task: asyncio.Task | None = None
_pregen_again = False


def schedule_pregenerate() -> None:
    """Re-generate the fixed lines soon; a burst of edits only triggers one run."""
    global _pregen_task, _pregen_again
    if _pregen_task is not None and not _pregen_task.done():
        _pregen_again = True
        return
    _pregen_task = asyncio.create_task(_pregenerate())


async def _pregenerate() -> None:
    global _pregen_again
    while True:
        _pregen_again = False
        await asyncio.sleep(2)
        await pregenerate_fixed_lines()
        if not _pregen_again:
            return


async def pregenerate_fixed_lines() -> None:
    """Generate every server's fixed lines (no {name}...) ahead of time so they play instantly."""
    if voice_model is None:
        return
    # Each line in each server's language: the same voice is generated per language.
    wanted = {(text, voice, lang_of(guild)) for guild in bot.guilds for text, voice in engine.fixed_texts(guild.id)}
    missing = []
    for text, voice, language in sorted(wanted, key=str):
        voice = int(voice) if isinstance(voice, str) and voice.isdigit() else voice
        if voice is not None and (not isinstance(voice, int) or store.get_voice(voice) is None):
            continue  # a voice that's gone: those lines use the bot's voice, generated below
        try:
            if not library.cache_path(text, voice, language=language).is_file():
                missing.append((text, voice, language))
        except VoiceNotReady:
            pass  # not built yet: generated on first use
    if not missing:
        return
    log.info("Generating %d new fixed lines", len(missing))
    results = await asyncio.gather(*(generate("pregen", partial(library.speak, language=language), text, voice,
                                              text=text, voice=str(voice or "bot"))
                                     for text, voice, language in missing), return_exceptions=True)
    for (text, _, _), result in zip(missing, results):
        if isinstance(result, Exception):
            log.warning("Pre-generating %r failed: %s", text, result)


def play_audio(vc, source, done=None) -> None:
    """Play a sound (the bot talking, a clip). `done()` runs (on the event
    loop) when it ends."""
    loop = asyncio.get_running_loop()

    def after(error) -> None:
        if error:
            log.warning("Playback error: %s", error)
        if done is not None:
            loop.call_soon_threadsafe(done)

    vc.play(source, after=after)


@dataclass
class Reply:
    text: str  # what the bot says, for echo detection
    pcm: bytes
    queued_at: float = field(default_factory=time.monotonic)
    played: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


# Per guild: the bot's pending replies, the task playing them, and what it
# said lately [normalized text, when he finished (None = still talking)].
reply_queues: dict[int, asyncio.Queue] = {}
reply_workers: dict[int, asyncio.Task] = {}
recent_lines: defaultdict[int, deque] = defaultdict(lambda: deque(maxlen=8))


# The live-feed entry (events.py) of the utterance being handled, so the
# replies it causes can report what became of them. Each utterance is
# handled in its own task, so each sees its own entry.
current_feed: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_feed", default=None)


def feed_update(**fields) -> None:
    entry = current_feed.get()
    if entry is not None:
        bus.publish({"type": "feed_update", "id": entry["id"], **fields})


def say(guild, text: str, pcm: bytes) -> asyncio.Future:
    """Queue a reply. The future resolves True once played, False if skipped."""
    queue = reply_queues.setdefault(guild.id, asyncio.Queue())
    worker = reply_workers.get(guild.id)
    if worker is None or worker.done():
        reply_workers[guild.id] = asyncio.create_task(_play_replies(guild, queue))
    reply = Reply(text, pcm)
    queue.put_nowait(reply)
    entry = current_feed.get()
    if entry is not None:
        reply.played.add_done_callback(lambda played: bus.publish(
            {"type": "feed_update", "id": entry["id"], "outcome": "played" if played.result() else "skipped"}))
    return reply.played


async def _play_replies(guild, queue: asyncio.Queue) -> None:
    while True:
        reply = await queue.get()
        vc = guild.voice_client
        # Wait out a sound that isn't a reply (the test sound), within reason.
        while vc is not None and vc.is_connected() and is_busy(vc) \
                and time.monotonic() - reply.queued_at < REPLY_MAX_AGE_S:
            await asyncio.sleep(0.05)
        if vc is None or not vc.is_connected() or is_busy(vc) \
                or time.monotonic() - reply.queued_at > REPLY_MAX_AGE_S:
            log.info("Skipping stale reply: %r", reply.text)
            reply.played.set_result(False)
            continue
        line = [normalize(reply.text), None]
        recent_lines[guild.id].append(line)
        finished = asyncio.Event()
        play_audio(vc, discord.PCMAudio(io.BytesIO(reply.pcm)), done=finished.set)
        await finished.wait()
        line[1] = time.monotonic()
        reply.played.set_result(True)


def clear_replies(guild) -> None:
    """Drop everything the bot was about to say (stop / leave / reset)."""
    queue = reply_queues.get(guild.id)
    while queue is not None and not queue.empty():
        queue.get_nowait().played.set_result(False)


def is_echo(guild_id: int, text: str) -> bool:
    heard = normalize(text)
    now = time.monotonic()
    for said, ended in recent_lines[guild_id]:
        if not said or (ended is not None and now - ended > ECHO_WINDOW_S):
            continue
        if said in heard or SequenceMatcher(None, heard, said).ratio() >= ECHO_SIMILARITY:
            return True
    return False


def is_busy(vc) -> bool:
    """Is a sound playing?"""
    return vc.is_playing()


def stop_sound(vc) -> None:
    vc.stop_playing()  # VoiceRecvClient.stop() would also stop listening




async def connect_and_listen(channel, text_channel):
    """Join `channel` (or move there) and make sure we're listening. Everyone
    in it is told the bot is listening (announce_join)."""
    vc = channel.guild.voice_client
    if vc is None:
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
        await announce_join(channel)
    elif vc.channel != channel:
        await vc.move_to(channel)
        await announce_join(channel)

    if not vc.is_listening():
        loop = asyncio.get_running_loop()

        def on_utterance(u: Utterance) -> None:
            # Called from the sink's watcher thread.
            asyncio.run_coroutine_threadsafe(handle_utterance(u, vc.channel, text_channel), loop)

        vc.listen(UtteranceSink(on_utterance, silence_s=END_SILENCE_S))
    return vc


async def disconnect(guild, *, pause_autojoin: bool = False) -> None:
    """Leave voice (!leave, "vete", auto-leave). pause_autojoin: someone sent
    him away, so don't auto-join that channel again until it empties out."""
    vc = guild.voice_client
    if vc is None:
        return
    if pause_autojoin:
        autojoin_paused[guild.id] = vc.channel.id
    clear_replies(guild)
    if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
        vc.stop_listening()  # flushes whatever is still being said
    await vc.disconnect()


async def say_text(guild, text: str | None, voice_id: int | None = None) -> None:
    """Say any text, in the bot's voice or a saved one (~0.5 s unless cached)."""
    if voice_model is None or not text or guild.voice_client is None:
        return
    line = await speech(text, voice_id, guild, None, "")
    if line is not None:
        say(guild, text, line[0])


# ------------------------------------------------------------------ presence
# Auto-join: the bot joins a voice channel once this many people (bots don't
# count) are in it, and leaves when nobody's left.
AUTO_JOIN_MIN_PEOPLE = 2
# Grace period before auto-leaving: someone may just be reconnecting.
AUTO_LEAVE_DELAY_S = 10
# No greeting/goodbye storm when someone's connection keeps dropping.
GREET_COOLDOWN_S = 60

auto_join_enabled: dict[int, bool] = defaultdict(lambda: os.getenv("AUTO_JOIN", "1") == "1")
# guild -> channel the bot was sent away from; no auto-join there until it
# drops below AUTO_JOIN_MIN_PEOPLE.
autojoin_paused: dict[int, int] = {}
last_greeting: dict[tuple[int, int, str], float] = {}
leave_tasks: dict[int, asyncio.Task] = {}


def people_in(channel) -> list:
    return [m for m in channel.members if not m.bot]


def ignore_bots(vc, user_id: int) -> bool:
    """Other bots (Lunabot & co.) are never listened to: their audio is
    dropped before it's even decrypted (see dave.ignore_user)."""
    member = vc.guild.get_member(user_id)
    return member is not None and member.bot


async def greet(member, event: str) -> None:
    key = (member.guild.id, member.id, event)
    now = time.monotonic()
    if now - last_greeting.get(key, -GREET_COOLDOWN_S) < GREET_COOLDOWN_S:
        return
    last_greeting[key] = now
    await respond(member.guild, event, user_id=member.id, display_name=member.display_name)


async def auto_join(guild) -> None:
    if not auto_join_enabled[guild.id] or guild.voice_client is not None:
        return
    candidates = [
        c for c in guild.voice_channels
        if len(people_in(c)) >= AUTO_JOIN_MIN_PEOPLE
        and autojoin_paused.get(guild.id) != c.id
        and c.permissions_for(guild.me).connect
    ]
    if not candidates:
        return
    channel = max(candidates, key=lambda c: len(people_in(c)))
    try:
        await connect_and_listen(channel, None)
    except Exception as e:
        log.warning("Auto-join of %s failed: %s", channel, e)
        return
    log.info("Auto-joined %s (%d people)", channel, len(people_in(channel)))
    await respond(guild, "arrival")  # "A sapear."


async def auto_leave_later(guild) -> None:
    await asyncio.sleep(AUTO_LEAVE_DELAY_S)
    vc = guild.voice_client
    if vc is not None and not people_in(vc.channel):
        log.info("Nobody left in %s, leaving", vc.channel)
        await disconnect(guild)


@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot or before.channel == after.channel:
        return  # bots, or just mute/deafen/stream changes
    guild = member.guild
    paused = autojoin_paused.get(guild.id)
    if paused and before.channel and before.channel.id == paused \
            and len(people_in(before.channel)) < AUTO_JOIN_MIN_PEOPLE:
        autojoin_paused.pop(guild.id, None)

    vc = guild.voice_client
    if vc is None:
        await auto_join(guild)
        return
    here = vc.channel
    if after.channel == here:
        task = leave_tasks.pop(guild.id, None)
        if task:
            task.cancel()
        await greet(member, "hello")
    elif before.channel == here:
        if people_in(here):
            await greet(member, "bye")
        elif auto_join_enabled[guild.id]:
            leave_tasks[guild.id] = asyncio.create_task(auto_leave_later(guild))


# -------------------------------------------------------------------- timers
@dataclass
class Timer:
    ends_at: float
    who: str
    said: str
    message: str
    task: asyncio.Task | None = None


guild_timers: defaultdict[int, list] = defaultdict(list)


def start_timer(guild, seconds: float, said: str, message: str, member) -> Timer:
    name = speaker_name(guild.id, member.id, member.display_name)
    timer = Timer(time.monotonic() + seconds, name, said, message)

    async def ring() -> None:
        try:
            await asyncio.sleep(seconds)
            log.info("Timer for %s: %s (%s)", name, said, message or "-")
            # "{name}, {message}." or, without a message, "{name}, ya pasaron {said}."
            await respond(guild, "timer_ring", user_id=member.id, display_name=member.display_name,
                          values={"said": said, "message": message})
        finally:
            if timer in guild_timers[guild.id]:
                guild_timers[guild.id].remove(timer)

    timer.task = asyncio.create_task(ring())
    guild_timers[guild.id].append(timer)
    log.info("Timer set by %s: %s (%s)", name, said, message or "-")
    return timer


# ------------------------------------------------------------------ commands
# Every command works both as !command and as a /slash command.
OnOff = Literal["on", "off"]


@bot.hybrid_command(name="join", description="Join your voice channel and start listening")
async def join(ctx):
    if not ctx.author.voice:
        return await reply(ctx, "common.not_in_voice")
    await ctx.defer(ephemeral=True)
    channel = ctx.author.voice.channel
    autojoin_paused.pop(ctx.guild.id, None)
    posting = ctx.interaction is None and can_post(ctx.channel)
    try:
        await connect_and_listen(channel, ctx.channel if posting else None)
    except Exception as e:
        log.warning("Voice connection error: %s", e)
        return await reply(ctx, "common.failed")
    await reply(ctx, "join.joined", channel=channel.name)


@bot.hybrid_command(name="leave", description="Leave the voice channel")
async def leave(ctx):
    if ctx.voice_client:
        await disconnect(ctx.guild, pause_autojoin=True)
    await reply(ctx, "join.left")


@bot.hybrid_command(name="say", description="The bot says this out loud in the call")
async def say_command(ctx, *, text: str):
    if ctx.voice_client is None:
        return await reply(ctx, "common.not_in_voice")
    if voice_model is None:
        return await reply(ctx, "common.voice_off")
    await ctx.defer(ephemeral=True)
    await say_text(ctx.guild, text[:300])
    await reply(ctx, "common.done")


@bot.hybrid_command(name="timer", description="The bot reminds you in N minutes (optionally about something)")
async def timer_command(ctx, minutes: float, *, text: str = ""):
    if not 0 < minutes <= 720:
        return await reply(ctx, "timer.range")
    said = timers.describe(minutes * 60, lang_of(ctx))
    start_timer(ctx.guild, minutes * 60, said, text, ctx.author)
    await reply(ctx, "timer.set", said=said)


@bot.hybrid_command(name="timers", description="List the running timers, or clear them all")
async def timers_command(ctx, action: Literal["clear"] | None = None):
    running = guild_timers[ctx.guild.id]
    if action == "clear":
        for timer in list(running):
            timer.task.cancel()
        return await reply(ctx, "timer.cleared")
    now = time.monotonic()
    rows = [
        tr(ctx, "timer.row", who=t.who, what=t.message or t.said,
           left=f"{int((t.ends_at - now) // 60)}:{int((t.ends_at - now) % 60):02d}")
        for t in sorted(running, key=lambda t: t.ends_at)
    ]
    await reply(ctx, "common.raw", text="\n".join(rows) or tr(ctx, "timer.none"))


@bot.hybrid_command(name="autojoin", description="Join calls by itself when 2+ people are in one")
async def autojoin_command(ctx, mode: OnOff | None = None):
    if mode is not None:
        auto_join_enabled[ctx.guild.id] = mode == "on"
        if mode == "on":
            await auto_join(ctx.guild)
    await reply(ctx, "autojoin.on" if auto_join_enabled[ctx.guild.id] else "autojoin.off")


@bot.hybrid_command(name="userclone", description="Gags answer in the speaker's own cloned voice")
async def userclone_command(ctx, mode: OnOff | None = None):
    if mode is not None:
        if mode == "on" and voice_model is None:
            return await reply(ctx, "common.voice_off")
        userclone[ctx.guild.id] = mode == "on"
    await reply(ctx, "userclone.on" if userclone.get(ctx.guild.id) else "userclone.off")


# ------------------------------------------------------------------ consent
# Two things need people's explicit "yes", each asked separately:
#   voice        keep their voice to imitate it (/voice, !userclone)
#   transcripts  write down what they say (only where /transcripts is on)
# Bump the version when the text of a question changes what people agree to.
CONSENT_VERSION = 1


class ConsentButton(discord.ui.DynamicItem[discord.ui.Button],
                    template=r"consent:(?:(?P<purpose>voice|transcripts):)?(?P<choice>yes|no)"
                             r"(?::(?P<lang>[a-z]{2}))?(?::(?P<user_id>\d+))?"):
    """Accept / No thanks. Asked of one person (their id is in the button),
    or open to whoever clicks (the transcripts notice). Keeps working after
    a restart: everything it needs is in the button's id."""

    def __init__(self, purpose: str, choice: str, lang: str, user_id: int | None = None):
        yes = choice == "yes"
        custom_id = f"consent:{purpose}:{choice}:{lang}" + (f":{user_id}" if user_id else "")
        super().__init__(discord.ui.Button(
            label=i18n.t("consent.accept" if yes else "consent.decline", lang),
            style=discord.ButtonStyle.success if yes else discord.ButtonStyle.secondary,
            custom_id=custom_id))
        self.purpose, self.choice, self.lang, self.user_id = purpose, choice, lang, user_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        user_id = int(match["user_id"]) if match["user_id"] else None
        return cls(match["purpose"] or "voice", match["choice"], match["lang"] or i18n.default_language(), user_id)

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.user_id is not None and interaction.user.id != self.user_id:
            await interaction.response.send_message(i18n.t("consent.not_for_you", self.lang), ephemeral=True)
            return
        answer = record_consent(interaction.user.id, self.purpose, self.choice == "yes", self.lang)
        log.info("%s consent from %s: %s", self.purpose, interaction.user, self.choice)
        if self.user_id is None:  # a shared notice: answer just this person, keep the buttons for others
            await interaction.response.send_message(answer, ephemeral=True)
        else:
            await interaction.response.edit_message(content=answer, view=None)


def record_consent(user_id: int, purpose: str, accepted: bool, lang: str) -> str:
    """Store someone's answer and act on it. Returns the confirmation to show them."""
    store.set_consent(user_id, "accepted" if accepted else "declined", purpose=purpose, text_version=CONSENT_VERSION)
    if purpose == "voice":
        if accepted:
            voice_id = library.request_speaker_rebuild(user_id)
            if voice_id is not None and voice_model is not None:
                queue_build(voice_id)
        else:
            library.delete_speaker(user_id)
    elif not accepted:
        forget_text(user_id)
    return i18n.t(f"consent.{purpose}.{'accepted' if accepted else 'declined'}", lang, bot=bot_name())


def forget_text(user_id: int) -> int:
    """Delete everything written down of what this person said."""
    return store.delete_user_text(user_id) + transcript_files.delete_user(user_id)


def consent_view(purpose: str, lang: str, user_id: int | None = None) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(ConsentButton(purpose, "yes", lang, user_id))
    view.add_item(ConsentButton(purpose, "no", lang, user_id))
    return view


async def ask_consent(user_id: int, voice_channel, purpose: str) -> None:
    """Ask once: by DM, or in the voice channel's chat if their DMs are closed.
    Marked "pending" so they aren't asked again (/voice optin and
    /transcripts optin ask anew)."""
    member = voice_channel.guild.get_member(user_id)
    if member is None or member.bot:
        return
    store.set_consent(user_id, "pending", purpose=purpose, text_version=CONSENT_VERSION)
    lang = lang_of(voice_channel.guild)
    text = tr(voice_channel.guild, f"consent.{purpose}.ask")
    try:
        await member.send(text, view=consent_view(purpose, lang, user_id))
        return
    except discord.HTTPException:
        pass
    try:
        await voice_channel.send(f"{member.mention} {text}", view=consent_view(purpose, lang, user_id),
                                 allowed_mentions=discord.AllowedMentions(users=[member]))
    except discord.HTTPException as e:
        log.warning("Couldn't ask %s for %s consent: %s", member, purpose, e)


@bot.hybrid_group(name="voice", fallback="status", description="Your own cloned voice")
async def voice_group(ctx):
    consent = store.get_consent(ctx.author.id, purpose="voice")
    own = library.speaker_voice(None, ctx.author.id, ready_only=False)
    state = consent["status"] if consent else "never_asked"
    built = (own["status"] if own else "no_samples") if consent and consent["status"] == "accepted" else None
    await reply(ctx, "voice.status", consent=tr(ctx, f"consent.state.{state}"),
                voice=tr(ctx, f"voice.state.{built}") if built else "-")


@voice_group.command(name="optin", description="Agree to have your voice saved and cloned")
async def voice_optin(ctx):
    await ctx.send(tr(ctx, "consent.voice.ask"), view=consent_view("voice", lang_of(ctx), ctx.author.id),
                   ephemeral=True)


@voice_group.command(name="delete", description="Delete your saved voice and stop collecting it")
async def voice_delete(ctx):
    removed = library.delete_speaker(ctx.author.id)
    store.set_consent(ctx.author.id, "revoked", purpose="voice", text_version=CONSENT_VERSION)
    await reply(ctx, "voice.deleted" if removed else "voice.nothing_saved")


@voice_group.command(name="refresh", description="Rebuild your voice from your latest sentences")
async def voice_refresh(ctx):
    voice_id = library.request_speaker_rebuild(ctx.author.id)
    if voice_id is None or voice_model is None:
        return await reply(ctx, "voice.nothing_to_rebuild")
    queue_build(voice_id)
    await reply(ctx, "voice.rebuilding")


# --------------------------------------------------------------- transcripts
def transcripts_on(guild_id: int) -> bool:
    return bool(store.get_setting(guild_id, "transcripts.enabled"))


async def may_keep_text(user_id: int, voice_channel) -> bool:
    """May what this person just said be written down? Only where transcripts
    are on, and only if they said yes; people who never answered are asked."""
    if not transcripts_on(voice_channel.guild.id):
        return False
    consent = store.get_consent(user_id, purpose="transcripts")
    if consent is None:
        try:
            await ask_consent(user_id, voice_channel, "transcripts")
        except Exception:
            log.exception("Asking %s about transcripts failed", user_id)
        return False
    return consent["status"] == "accepted"


def notice_channel(guild):
    """Where the bot announces things: the chat of the call it's in, else
    the server's system channel (if it may post there)."""
    vc = guild.voice_client
    for channel in (vc.channel if vc else None, guild.system_channel):
        if channel is not None and can_post(channel):
            return channel
    return None


async def set_transcripts(guild, on: bool) -> str:
    """Turn transcripts on/off for a server and tell everyone. When turned on,
    the notice carries the Accept / No thanks buttons for anyone to answer.
    Returns the message key for whoever did it."""
    store.set_setting(guild.id, "transcripts.enabled", on)
    log.info("Transcripts %s in %s", "on" if on else "off", guild.name)
    channel = notice_channel(guild)
    if channel is None:
        log.warning("Transcripts turned %s in %s, but there's nowhere to announce it", "on" if on else "off", guild)
        return "transcripts.on_unannounced" if on else "transcripts.off"
    try:
        if on:
            await channel.send(tr(guild, "transcripts.broadcast_on"), view=consent_view("transcripts", lang_of(guild)))
        else:
            await channel.send(tr(guild, "transcripts.broadcast_off"))
    except discord.HTTPException as e:
        log.warning("Couldn't announce transcripts in %s: %s", channel, e)
    return "transcripts.on" if on else "transcripts.off"


@bot.hybrid_group(name="transcripts", fallback="status",
                  description="Whether what's said in calls is written down, and your choice")
async def transcripts_group(ctx):
    consent = store.get_consent(ctx.author.id, purpose="transcripts")
    await reply(ctx, "transcripts.status",
                server=tr(ctx, "common.on" if transcripts_on(ctx.guild.id) else "common.off"),
                consent=tr(ctx, f"consent.state.{consent['status'] if consent else 'never_asked'}"))


@transcripts_group.command(name="on", description="Admins: start writing down calls (everyone is asked first)")
async def transcripts_enable(ctx):
    if not await author_is_admin(ctx):
        return await reply(ctx, "common.admins_only")
    await reply(ctx, await set_transcripts(ctx.guild, True))


@transcripts_group.command(name="off", description="Admins: stop writing down calls")
async def transcripts_disable(ctx):
    if not await author_is_admin(ctx):
        return await reply(ctx, "common.admins_only")
    await reply(ctx, await set_transcripts(ctx.guild, False))


@transcripts_group.command(name="optin", description="Agree to have your words written down")
async def transcripts_optin(ctx):
    await ctx.send(tr(ctx, "consent.transcripts.ask"),
                   view=consent_view("transcripts", lang_of(ctx), ctx.author.id), ephemeral=True)


@transcripts_group.command(name="delete", description="Delete everything written down of what you said")
async def transcripts_delete(ctx):
    removed = forget_text(ctx.author.id)
    store.set_consent(ctx.author.id, "revoked", purpose="transcripts", text_version=CONSENT_VERSION)
    await reply(ctx, "transcripts.deleted", count=removed)


# ------------------------------------------------------------------ language
@bot.hybrid_command(name="language", description="The bot's language in this server (admins can change it)")
async def language_command(ctx, code: str | None = None):
    available = ", ".join(i18n.languages())
    if code is None:
        return await reply(ctx, "language.current", language=lang_of(ctx), available=available)
    if not await author_is_admin(ctx):
        return await reply(ctx, "common.admins_only")
    code = code.strip().lower()
    if code not in i18n.languages():
        return await reply(ctx, "language.unknown", language=code, available=available)
    store.set_setting(ctx.guild.id, "language", code)
    await reply(ctx, "language.set", language=code)


# --------------------------------------------------------------- join notice
# Everyone in a call should know the bot is listening. Not repeated when it
# just reconnects to the same channel.
JOIN_NOTICE_EVERY_S = 3 * 3600
last_join_notice: dict[int, float] = {}


async def announce_join(channel) -> None:
    now = time.monotonic()
    if now - last_join_notice.get(channel.id, -JOIN_NOTICE_EVERY_S) < JOIN_NOTICE_EVERY_S:
        return
    last_join_notice[channel.id] = now
    if not can_post(channel):
        return
    guild = channel.guild
    text = tr(guild, "join.notice",
              transcripts=tr(guild, "join.transcripts_on" if transcripts_on(guild.id) else "join.transcripts_off"))
    try:
        await channel.send(text)
    except discord.HTTPException as e:
        log.warning("Couldn't post the join notice in %s: %s", channel, e)


def reload_content() -> None:
    """Recompile every server's reactions from the database and re-generate
    the fixed lines. Edits from the dashboard do this by themselves."""
    engine.invalidate()
    schedule_pregenerate()


@bot.hybrid_command(name="reload", description="Re-read the gags and replies from the database")
async def reload_command(ctx):
    reload_content()
    await reply(ctx, "common.done")


def fit(ctx, rows: list[str], limit: int = 1900) -> str:
    """As many rows as fit in one Discord message."""
    out, size = [], 0
    for row in rows:
        if size + len(row) + 1 > limit:
            out.append(tr(ctx, "common.more", count=len(rows) - len(out)))
            break
        out.append(row)
        size += len(row) + 1
    return "\n".join(out)


@bot.hybrid_command(name="nicknames", description="Who's in the call, their IDs and nicknames")
async def nicknames_command(ctx):
    vc = ctx.voice_client
    channel = vc.channel if vc else (ctx.author.voice.channel if ctx.author.voice else None)
    if channel is None:
        return await reply(ctx, "common.not_in_voice")
    rows = []
    for member in people_in(channel):
        person = store.get_person(ctx.guild.id, member.id)
        nick = person["nickname"] if person else None
        said = nick or tr(ctx, "nicknames.none", name=store.nickname_for(ctx.guild.id, member.id, member.display_name))
        rows.append(tr(ctx, "nicknames.row", display=member.display_name, id=member.id, said=said))
    await reply(ctx, "common.raw", text=fit(ctx, rows) or tr(ctx, "nicknames.nobody"))


@bot.hybrid_command(name="record", description="Record YOUR speech (only yours) to testdata/ for testing")
async def record_command(ctx, mode: OnOff | None = None):
    if mode == "on":
        recorders[ctx.guild.id] = SessionRecorder(TESTDATA_DIR, ctx.author.id, ctx.author.display_name)
        log.info("Test recording of %s -> %s", ctx.author.display_name, recorders[ctx.guild.id].dir)
    elif mode == "off":
        recorder = recorders.pop(ctx.guild.id, None)
        if recorder:
            log.info("Test recording stopped: %d clips in %s", recorder.count, recorder.dir)
    await reply(ctx, "record.on" if ctx.guild.id in recorders else "record.off")


@bot.hybrid_command(name="reset", description="Rejoin and clear stuck state ('full' restarts the bot, owner only)")
async def reset(ctx, mode: Literal["full"] | None = None):
    global restart_requested
    if mode == "full":
        if not await bot.is_owner(ctx.author):
            return await reply(ctx, "common.owner_only")
        await reply(ctx, "reset.restarting")
        restart_requested = True
        await bot.close()  # bot.run() returns, then __main__ re-executes the process
        return

    await ctx.defer(ephemeral=True)
    vc = ctx.voice_client
    channel = vc.channel if vc else (ctx.author.voice.channel if ctx.author.voice else None)
    if vc:
        try:
            if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
                vc.stop_listening()
            await vc.disconnect(force=True)
        except Exception:
            log.exception("Error disconnecting during reset")

    voice_commands.reset()
    speaker_locks.clear()
    clear_replies(ctx.guild)
    worker = reply_workers.pop(ctx.guild.id, None)
    if worker is not None:
        worker.cancel()
    reply_queues.pop(ctx.guild.id, None)
    recent_lines.pop(ctx.guild.id, None)
    engine.invalidate(ctx.guild.id)

    if channel is None:
        return await reply(ctx, "common.done")
    try:
        await connect_and_listen(channel, ctx.channel if ctx.interaction is None and can_post(ctx.channel) else None)
    except Exception as e:
        log.warning("Rejoining %s after reset failed: %s", channel, e)
        return await reply(ctx, "common.failed")
    await reply(ctx, "common.done")


# ----------------------------------------------------------------- dashboard
# The admin web dashboard (web/) reads the bot's state and drives it through
# this controller; the contract is in docs/PLAN.md.
STARTED_AT = time.monotonic()
DASHBOARD = os.getenv("DASHBOARD", "1") == "1"
dashboard_runner = None


def _model_rows() -> list[dict]:
    rows = []
    if isinstance(transcriber, HybridTranscriber):
        engines = (transcriber.parakeet, transcriber.whisper)
    else:
        engines = (transcriber,)
    for engine in engines:
        if isinstance(engine, WhisperTranscriber):
            rows.append({"name": engine.model_name, "role": "stt", "device": engine.device or "-",
                         "loaded": engine.model is not None})
        elif isinstance(engine, ParakeetTranscriber):
            rows.append({"name": "parakeet-tdt-0.6b-v3", "role": "stt", "device": engine.device,
                         "loaded": engine.recognizer is not None})
    if VOICE:
        model = voice_model
        rows.append({"name": "k2-fsa/OmniVoice", "role": "tts",
                     "device": str(model.device) if model is not None else "-", "loaded": model is not None})
    return rows


def _gpu_status() -> dict | None:
    torch = sys.modules.get("torch")  # only imported when the voice is on
    if torch is None or not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {"name": torch.cuda.get_device_name(0), "used_gb": round((total - free) / 1e9, 2),
            "total_gb": round(total / 1e9, 2)}


class Controller:
    def status(self) -> dict:
        guilds = []
        for guild in bot.guilds:
            vc = guild.voice_client
            guilds.append({
                "id": str(guild.id), "name": guild.name,
                "voice": {"channel_id": str(vc.channel.id), "channel": vc.channel.name,
                          "people": len(people_in(vc.channel))} if vc is not None else None,
                "voice_channels": [{"id": str(c.id), "name": c.name, "people": len(people_in(c))}
                                   for c in guild.voice_channels if c.permissions_for(guild.me).connect],
                "toggles": {"autojoin": auto_join_enabled[guild.id], "userclone": bool(userclone.get(guild.id)),
                            "record": guild.id in recorders, "transcripts": transcripts_on(guild.id)},
                "language": lang_of(guild),
                "reply_queue": reply_queues[guild.id].qsize() if guild.id in reply_queues else 0,
                "timers": len(guild_timers[guild.id]),
            })
        return {
            "bot": {"name": bot_name(), "user": str(bot.user) if bot.user else None,
                    "connected": bot.is_ready() and not bot.is_closed(),
                    "uptime_s": round(time.monotonic() - STARTED_AT), "stt_engine": STT_ENGINE},
            "models": _model_rows(),
            "gpu": _gpu_status(),
            "tts_queue": tts.snapshot(),
            "guilds": guilds,
        }

    # Voices (dashboard "Voices & sounds"). Blocking model work goes through the
    # TTS queue, behind live replies.
    async def preview(self, text: str, voice_id: int | None = None) -> bytes:
        """`text` in this voice (None: the bot's) as a WAV file, for the browser."""
        if voice_model is None:
            raise ValueError("Voice is off")
        text = (text or "").strip()[:300]
        if not text:
            raise ValueError("Nothing to say")
        try:
            audio = library.cached(text, voice_id)
            if audio is None:
                audio = await generate("preview", library.speak, text, voice_id, text=text, voice=str(voice_id or "bot"))
        except VoiceNotReady as e:
            raise ValueError(str(e)) from None
        buffer = io.BytesIO()
        sf.write(buffer, audio, VOICE_RATE, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def build_voice(self, voice_id: int) -> str:
        """Queue (re)building a voice; its status in the store shows progress."""
        if voice_model is None:
            raise ValueError("Voice is off")
        if not library.request_rebuild(voice_id):
            raise ValueError("This voice has nothing to build from yet")
        queue_build(voice_id)
        return "Build queued"

    async def transcribe_voice(self, voice_id: int) -> str:
        """Fill in what's said in the voice's reference (CPU Whisper, a few seconds)."""
        return await asyncio.to_thread(library.transcribe_ref, voice_id, force=True)

    async def play_sound(self, guild_id: int, sound_id: int) -> str:
        guild = bot.get_guild(int(guild_id))
        if guild is None or guild.voice_client is None:
            raise ValueError("Not in a voice channel there")
        await play_sound(guild, int(sound_id))
        return "Playing"

    async def control(self, action: str, **params) -> str:
        global restart_requested
        if action == "reload":
            reload_content()
            return "Reloaded"
        if action == "restart":
            restart_requested = True
            asyncio.get_running_loop().call_later(0.5, lambda: asyncio.create_task(bot.close()))
            return "Restarting"

        if action not in ("join", "leave", "stop", "clear_queue", "say", "toggle"):
            raise ValueError(f"Unknown action {action!r}")
        guild = bot.get_guild(int(params.get("guild_id") or 0))
        if guild is None:
            raise ValueError("Unknown server")
        vc = guild.voice_client
        if action == "join":
            channel = guild.get_channel(int(params.get("channel_id") or 0))
            if not isinstance(channel, discord.VoiceChannel):
                raise ValueError("Unknown voice channel")
            autojoin_paused.pop(guild.id, None)
            await connect_and_listen(channel, None)
            return f"Joined {channel.name}"
        if action == "leave":
            await disconnect(guild, pause_autojoin=True)
            return "Left"
        if action == "stop":
            clear_replies(guild)
            if vc is not None and is_busy(vc):
                stop_sound(vc)
            return "Stopped"
        if action == "clear_queue":
            clear_replies(guild)
            return "Queue cleared"
        if action == "say":
            text = str(params.get("text") or "").strip()[:300]
            if not text:
                raise ValueError("Nothing to say")
            if vc is None:
                raise ValueError("Not in a voice channel")
            if voice_model is None:
                raise ValueError("Voice is off")
            voice_id = params.get("voice_id")
            await say_text(guild, text, int(voice_id) if voice_id not in (None, "") else None)
            return "Said"
        if action == "toggle":
            name, value = params.get("name"), bool(params.get("value"))
            if name == "autojoin":
                auto_join_enabled[guild.id] = value
                if value:
                    await auto_join(guild)
            elif name == "userclone":
                if value and voice_model is None:
                    raise ValueError("Voice is off")
                userclone[guild.id] = value
            elif name == "transcripts":
                # Through set_transcripts, never the bare setting: turning
                # them on must announce it and ask people.
                await set_transcripts(guild, value)
            elif name == "record":
                if value:
                    member = guild.get_member(int(params.get("user_id") or 0))
                    if member is None:
                        raise ValueError("Recording needs a person: start it with /record in Discord")
                    recorders[guild.id] = SessionRecorder(TESTDATA_DIR, member.id, member.display_name)
                else:
                    recorders.pop(guild.id, None)
            else:
                raise ValueError(f"Unknown toggle {name!r}")
            return f"{name} {'on' if value else 'off'}"
        raise ValueError(f"Unknown action {action!r}")


async def start_dashboard() -> None:
    global dashboard_runner
    if not DASHBOARD or dashboard_runner is not None:
        return
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("DASHBOARD_PORT", "8765"))
    # A broken dashboard must never keep the bot itself from running.
    try:
        from web.server import BusLogHandler, load_or_create_token, login_url, start_dashboard as start

        token = os.getenv("DASHBOARD_TOKEN", "").strip() or load_or_create_token(ROOT / "data" / "dashboard_token")
        dashboard_runner = await start(Controller(), bus, host=host, port=port, token=token,
                                       store=store, engine=engine, base_dir=ROOT, library=library, sounds=sounds)
    except Exception as e:
        log.warning("Dashboard couldn't start on %s:%d: %s", host, port, e)
        return
    logging.getLogger().addHandler(BusLogHandler(bus, asyncio.get_running_loop()))
    log.info("Dashboard: %s", login_url(host, port, token))


if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        sys.exit("Error: DISCORD_BOT_TOKEN is not set (put it in .env).")
    if token.lower().startswith("bot "):
        token = token.split(" ", 1)[1]
    bot.run(token, root_logger=True)
    if restart_requested:
        os.execv(sys.executable, [sys.executable, *sys.argv])
