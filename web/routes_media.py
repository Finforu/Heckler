"""Voices and sounds: the admin side of voice_library.VoiceLibrary and
sound_library.SoundLibrary, plus the soundboard.

Registered only when the bot passes a library (voices) or sounds (sounds).
Every write goes through the libraries, never straight to the store, so files
and cached lines go with the rows. Blocking work (decoding, ingest, waveform
peaks) runs in a thread; model work goes through the Controller, which queues
it behind live replies.

Voices, under /api/g/{gid}:
    GET    voices                              the Voices tab: voices here + global, people's own voices, cache
    POST   voices                              multipart {name, file} (clone) | JSON {kind: "designed", name, instruct, speed?}
    PATCH  voices/{vid}                        {name, gain_db, speed, num_step, tags, language, ref_text, instruct}
    DELETE voices/{vid}                        library.delete_voice (not the bot's own voice)
    POST   voices/{vid}/ingest                 multipart {file?, start_s?, end_s?}: new audio and/or a new selection
    GET    voices/{vid}/audio/{ref|source}     the file, for an <audio> player
    GET    voices/{vid}/peaks?which=&n=        waveform peaks (+ the automatic speech segments of the source)
    POST   voices/{vid}/transcribe             controller.transcribe_voice
    POST   voices/{vid}/build                  controller.build_voice
    POST   voices/{vid}/preview                {text} -> audio/wav (controller.preview)
    POST   voices/cache/clear                  {voice_id?}
    POST   speakers/{uid}/rebuild | promote    promote: copy a consented person's voice into a clone voice here
    DELETE speakers/{uid}                      library.delete_speaker
Sounds:
    GET    sounds/{sid}/audio                  the stored file
    POST   sounds                              multipart {name, file}
    PATCH  sounds/{sid}                        {name, enabled, gain_db}
    DELETE sounds/{sid}                        sounds.delete
    POST   sounds/{sid}/reaction               {trigger: {type: phrase|command|slash, ...}}: a kind="sound" reaction
    POST   sounds/{sid}/play                   controller.play_sound
    GET    board                               enabled sounds + usable voices, for the soundboard
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import secrets
import shutil
from collections import OrderedDict
from pathlib import Path

from aiohttp import web

from .routes_content import FieldError, _body, _gid, _int, _names, api, to_user_id
from .util import error, json_response

log = logging.getLogger(__name__)

VOICE_UPLOAD_HARD_CAP = 50 * 1024 * 1024
PEAK_BUCKETS = 1200
PEAKS_CACHE = 32
AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".opus": "audio/ogg",
               ".flac": "audio/flac", ".m4a": "audio/mp4", ".aac": "audio/aac", ".webm": "audio/webm"}
VOICE_EDITABLE = ("name", "gain_db", "speed", "num_step", "tags", "language", "ref_text", "instruct")
# Changing these changes how the voice sounds: it has to be built again.
IDENTITY = {"clone": ("ref_text",), "speaker": ("ref_text",), "designed": ("instruct", "speed")}


# ───────────────────────────── voice design attributes ─────────────────────────────

def _load_voice_design():
    """omnivoice/utils/voice_design.py on its own: importing the omnivoice
    package would pull in torch."""
    import voice_library

    path = Path(voice_library.ROOT) / "omnivoice" / "utils" / "voice_design.py"
    spec = importlib.util.spec_from_file_location("_dash_voice_design", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DESIGN = None


def design_attributes() -> list[dict]:
    """The instruct items OmniVoice accepts, by category (English names;
    dialects are Chinese-only)."""
    global _DESIGN
    if _DESIGN is None:
        _DESIGN = _load_voice_design()
    names = ["gender", "age", "pitch", "style", "accent", "dialect"]
    out = []
    for name, cat in zip(names, _DESIGN._INSTRUCT_CATEGORIES):
        values = list(cat) if isinstance(cat, dict) else sorted(cat)
        out.append({"category": name, "values": values})
    return out


def check_instruct(instruct: str) -> str:
    """Normalise and validate like OmniVoice's _resolve_instruct: known items
    only, at most one per category, no accent with a dialect."""
    if _DESIGN is None:
        design_attributes()
    items = [x.strip().lower() for x in re.split(r"\s*[,，]\s*", instruct or "") if x.strip()]
    if not items:
        raise FieldError("describe the voice: pick at least one attribute", "instruct")
    unknown = [x for x in items if x not in _DESIGN._INSTRUCT_ALL_VALID]
    if unknown:
        raise FieldError(f"not supported: {', '.join(unknown)}", "instruct")
    if any(x.endswith("话") for x in items) and any(" accent" in x for x in items):
        raise FieldError("an accent (English) and a dialect (Chinese) can't be mixed", "instruct")
    en = [_DESIGN._INSTRUCT_ZH_TO_EN.get(x, x) for x in items]
    for cat in _DESIGN._INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [x for x in en if x in cat]
        if len(hits) > 1:
            raise FieldError(f"pick one of: {', '.join(hits)}", "instruct")
    return ", ".join(dict.fromkeys(items))


# ───────────────────────────── setup ─────────────────────────────

def setup_media_routes(app: web.Application, store, library=None, sounds=None) -> None:
    if store is None or (library is None and sounds is None):
        return
    app["library"] = library
    app["sounds"] = sounds
    app["peaks_cache"] = OrderedDict()
    r = app.router
    g = "/api/g/{gid:\\d+}"
    r.add_get(f"{g}/board", board)
    if library is not None:
        r.add_get(f"{g}/voices", voices_list)
        r.add_post(f"{g}/voices", voice_create)
        r.add_post(f"{g}/voices/cache/clear", cache_clear)
        v = f"{g}/voices/{{vid:\\d+}}"
        r.add_patch(v, voice_update)
        r.add_delete(v, voice_delete)
        r.add_post(f"{v}/ingest", voice_ingest)
        r.add_get(f"{v}/audio/{{which}}", voice_audio)
        r.add_get(f"{v}/peaks", voice_peaks)
        r.add_post(f"{v}/transcribe", voice_transcribe)
        r.add_post(f"{v}/build", voice_build)
        r.add_post(f"{v}/preview", voice_preview)
        s = f"{g}/speakers/{{uid:\\d+}}"
        r.add_post(f"{s}/rebuild", speaker_rebuild)
        r.add_post(f"{s}/promote", speaker_promote)
        r.add_delete(s, speaker_delete)
    if sounds is not None:
        r.add_post(f"{g}/sounds", sound_create)
        s = f"{g}/sounds/{{sid:\\d+}}"
        r.add_patch(s, sound_update)
        r.add_delete(s, sound_delete)
        r.add_get(f"{s}/audio", sound_audio)
        r.add_post(f"{s}/reaction", sound_reaction)
        r.add_post(f"{s}/play", sound_play)


# ───────────────────────────── helpers ─────────────────────────────

def _work_dir(app: web.Application) -> Path:
    path = app["base_dir"] / "data" / "tmp" / f"upload-{secrets.token_hex(6)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


async def _read_form(request: web.Request, work: Path, cap: int) -> tuple[dict, Path | None, str | None]:
    """Multipart fields (as strings) and the uploaded `file`, streamed to work/
    and capped at `cap` bytes (413 above it)."""
    if request.content_length and request.content_length > cap + 64 * 1024:
        raise web.HTTPRequestEntityTooLarge(max_size=cap, actual_size=request.content_length,
                                            text=f'{{"error": "the file is over {cap // 2**20} MB"}}',
                                            content_type="application/json")
    if request.content_type == "application/json":
        body = await _body(request)
        return {k: str(v) for k, v in body.items() if v is not None}, None, None
    if not request.content_type.startswith("multipart/"):
        if request.can_read_body:
            form = await request.post()  # urlencoded fields only; no file this way
            return {k: str(v)[:2000] for k, v in form.items() if isinstance(v, str)}, None, None
        return {}, None, None
    try:
        reader = await request.multipart()
    except (AssertionError, ValueError, KeyError):
        raise FieldError("choose an audio file (sent as multipart/form-data)", "file") from None
    fields, upload, filename = {}, None, None
    while (part := await reader.next()) is not None:
        if part.name == "file" and part.filename is not None:
            filename = Path(part.filename).name
            upload = work / ("upload" + Path(filename).suffix.lower()[:8])
            size = 0
            with open(upload, "wb") as f:
                while chunk := await part.read_chunk(256 * 1024):
                    size += len(chunk)
                    if size > cap:
                        raise web.HTTPRequestEntityTooLarge(
                            max_size=cap, actual_size=size, content_type="application/json",
                            text=f'{{"error": "the file is over {cap // 2**20} MB"}}')
                    f.write(chunk)
            if size == 0:
                upload = None
            elif upload.suffix not in AUDIO_TYPES:
                upload = upload.rename(upload.with_suffix(_sniff(upload)))
        elif part.name:
            fields[part.name] = (await part.text())[:2000]
    return fields, upload, filename


def _sniff(path: Path) -> str:
    """A file extension from the first bytes, for uploads without a usable name."""
    head = path.read_bytes()[:12] if path.stat().st_size < 64 else open(path, "rb").read(12)
    if head[:4] == b"RIFF":
        return ".wav"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return ".audio"


def _float(value, field: str, *, low: float | None = None, high: float | None = None, allow_none=True):
    if value in (None, ""):
        if allow_none:
            return None
        raise FieldError(f"{field} is required", field)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise FieldError(f"{field} must be a number", field) from None
    if (low is not None and value < low) or (high is not None and value > high):
        raise FieldError(f"{field} must be between {low:g} and {high:g}", field)
    return value


def _voice_here(store, gid: int, vid: int, *, allow_speaker: bool = False) -> dict:
    row = store.get_voice(vid)
    if row is None or row["guild_id"] not in (None, gid):
        raise KeyError(f"no voice {vid} here")
    if row["kind"] == "speaker" and not allow_speaker:
        raise KeyError(f"voice {vid} is someone's own voice")
    return row


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _source_file(library, vid: int) -> Path | None:
    files = sorted(library.folder(vid).glob("source.*"))
    return files[0] if files else None


def _audio_seconds(path: Path) -> float | None:
    import soundfile as sf

    try:
        return round(sf.info(str(path)).duration, 2)
    except Exception:
        return None


def _voice_out(library, row: dict, names: dict, bot_id) -> dict:
    folder = library.folder(row["id"])
    ref = folder / "ref.wav"
    source = _source_file(library, row["id"])
    status = row["status"]
    has_prompt = (folder / "prompt.pt").is_file() or (row["kind"] == "designed" and status == "ready")

    def who(uid):
        return None if uid is None else names.get(uid, str(uid))

    return {
        **{k: row[k] for k in ("id", "guild_id", "name", "kind", "owner_user_id", "ref_text", "instruct", "speed",
                               "num_step", "gain_db", "tags", "language", "status", "error", "created_by",
                               "created_at", "updated_at")},
        "is_bot": row["id"] == bot_id,
        "scope": "global" if row["guild_id"] is None else "server",
        "creator": who(row["created_by"]) or "admin",
        "owner": who(row["owner_user_id"]),
        "ref_seconds": _audio_seconds(ref) if ref.is_file() else None,
        "source_name": source.name if source else None,
        "source_seconds": _audio_seconds(source) if source and source.suffix in (".wav", ".flac", ".ogg") else None,
        "has_prompt": has_prompt,
        "needs_build": status in ("draft", "failed"),
        "cache_bytes": _dir_size(library.cache_dir / str(row["id"])),
    }


# ───────────────────────────── voices ─────────────────────────────

@api
async def voices_list(request: web.Request) -> web.Response:
    store, library, gid = request.app["store"], request.app["library"], _gid(request)
    names = _names(store, gid)
    bot_id = library.bot_voice_id()

    def collect():
        voices = [_voice_out(library, v, names, bot_id) for v in store.list_voices(gid) if v["kind"] != "speaker"]
        # People's own voices: only for people this server knows.
        people = {p["user_id"]: p for p in store.list_people(gid)}
        speakers = []
        for v in store.find_voices(kind="speaker"):
            if v["owner_user_id"] not in people or v["guild_id"] not in (None, gid):
                continue
            consent = (store.get_consent(v["owner_user_id"]) or {}).get("status")
            speakers.append({**_voice_out(library, v, names, bot_id), "consent": consent,
                             "clip_seconds": round(library._clip_seconds(v["id"]), 1)})
        return voices, speakers, library.cache_size()

    voices, speakers, cache_size = await asyncio.to_thread(collect)
    return json_response({
        "voices": voices, "speakers": speakers, "bot_voice_id": bot_id,
        "cache": {"bytes": cache_size, "limit": library.cache_limit()},
        "design": design_attributes(),
        "upload_mb": _voice_cap(store, gid) // 2**20,
        "preview_text": store.get_setting(gid, "voices.preview_text", ""),
        "controls": {k: hasattr(request.app["controller"], k)
                     for k in ("preview", "build_voice", "transcribe_voice", "play_sound")},
    })


def _voice_cap(store, gid: int) -> int:
    mb = float(store.get_setting(gid, "voices.max_mb", 50) or 50)
    return min(int(mb * 2**20), VOICE_UPLOAD_HARD_CAP)


def _clean_name(value) -> str:
    name = str(value or "").strip()
    if not name:
        raise FieldError("the voice needs a name", "name")
    if len(name) > 60:
        raise FieldError("60 characters at most", "name")
    return name


@api
async def voice_create(request: web.Request) -> web.Response:
    store, library, gid = request.app["store"], request.app["library"], _gid(request)
    if request.content_type == "application/json":
        body = await _body(request)
        if body.get("kind") != "designed":
            raise FieldError("a clone voice is made from an uploaded file (multipart)", "kind")
        name = _clean_name(body.get("name"))
        if store.find_voice(gid, name):
            raise FieldError(f"there's already a voice called {name}", "name")
        instruct = check_instruct(str(body.get("instruct") or ""))
        speed = _float(body.get("speed"), "speed", low=0.5, high=2.0)
        vid = store.add_voice(name, guild_id=gid, kind="designed", instruct=instruct, speed=speed,
                              status="draft", created_by=None)
        return json_response({"voice": store.get_voice(vid)})

    work = _work_dir(request.app)
    try:
        fields, upload, filename = await _read_form(request, work, _voice_cap(store, gid))
        name = _clean_name(fields.get("name"))
        if upload is None:
            raise FieldError("choose an audio file", "file")
        if store.find_voice(gid, name):
            raise FieldError(f"there's already a voice called {name}", "name")
        vid = store.add_voice(name, guild_id=gid, kind="clone", status="draft", created_by=None)
        try:
            row = await asyncio.to_thread(library.ingest, vid, upload)
        except Exception as e:
            await asyncio.to_thread(library.delete_voice, vid)
            raise ValueError(f"couldn't use that audio: {e}") from None
        return json_response({"voice": row})
    finally:
        await asyncio.to_thread(shutil.rmtree, work, True)


@api
async def voice_update(request: web.Request) -> web.Response:
    store, gid, vid = request.app["store"], _gid(request), _int(request, "vid")
    row = _voice_here(store, gid, vid)
    body = await _body(request)
    unknown = set(body) - set(VOICE_EDITABLE)
    if unknown:
        raise FieldError(f"can't change: {', '.join(sorted(unknown))}")
    fields = {}
    if "name" in body:
        fields["name"] = _clean_name(body["name"])
        other = store.find_voice(gid, fields["name"])
        if other and other["id"] != vid:
            raise FieldError(f"there's already a voice called {fields['name']}", "name")
    if "gain_db" in body:
        fields["gain_db"] = _float(body["gain_db"], "gain_db", low=-30, high=20) or 0.0
    if "speed" in body:
        fields["speed"] = _float(body["speed"], "speed", low=0.5, high=2.0)
    if "num_step" in body:
        steps = _float(body["num_step"], "num_step", low=4, high=64)
        fields["num_step"] = None if steps is None else int(steps)
    if "tags" in body:
        tags = body["tags"]
        if isinstance(tags, str):
            tags = [t for t in re.split(r"\s*,\s*", tags) if t]
        if not isinstance(tags, list):
            raise FieldError("tags must be a list", "tags")
        fields["tags"] = [str(t).strip()[:30] for t in tags if str(t).strip()][:20]
    if "language" in body:
        language = str(body["language"] or "").strip().lower()
        if language and not re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]{2,8})?", language):
            raise FieldError("a language code such as es or en", "language")
        fields["language"] = language or None
    if "ref_text" in body:
        if row["kind"] == "designed":
            raise FieldError("a designed voice has no transcript", "ref_text")
        fields["ref_text"] = str(body["ref_text"] or "").strip() or None
    if "instruct" in body:
        if row["kind"] != "designed":
            raise FieldError("only designed voices have instruct", "instruct")
        fields["instruct"] = check_instruct(str(body["instruct"] or ""))
    changed = [k for k in IDENTITY.get(row["kind"], ()) if k in fields and fields[k] != row[k]]
    if changed and row["status"] in ("ready", "failed"):
        fields["status"] = "draft"  # sounds different now: build again (the old prompt works meanwhile)
    if fields:
        store.update_voice(vid, **fields)
    return json_response({"voice": store.get_voice(vid), "rebuild": bool(changed)})


@api
async def voice_delete(request: web.Request) -> web.Response:
    store, library, gid, vid = request.app["store"], request.app["library"], _gid(request), _int(request, "vid")
    _voice_here(store, gid, vid)
    if vid == library.bot_voice_id():
        raise ValueError("that's the bot's own voice: replace its file instead of deleting it")
    await asyncio.to_thread(library.delete_voice, vid)
    return json_response({"ok": True})


@api
async def voice_ingest(request: web.Request) -> web.Response:
    store, library, gid, vid = request.app["store"], request.app["library"], _gid(request), _int(request, "vid")
    row = _voice_here(store, gid, vid)
    if row["kind"] != "clone":
        raise ValueError("only clone voices are made from audio")
    work = _work_dir(request.app)
    try:
        fields, upload, _ = await _read_form(request, work, _voice_cap(store, gid))
        start = _float(fields.get("start_s"), "start_s", low=0)
        end = _float(fields.get("end_s"), "end_s", low=0)
        if start is not None and end is not None and end <= start:
            raise FieldError("the selection ends before it starts", "end_s")
        source = upload or _source_file(library, vid)
        if source is None:
            raise FieldError("upload an audio file first", "file")
        try:
            row = await asyncio.to_thread(library.ingest, vid, source, start_s=start, end_s=end)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"couldn't use that audio: {e}") from None
        if upload is not None or start is not None or end is not None:
            store.update_voice(vid, ref_text=None)  # new reference audio: its old transcript no longer matches
        request.app["peaks_cache"].clear()
        return json_response({"voice": store.get_voice(vid)})
    finally:
        await asyncio.to_thread(shutil.rmtree, work, True)


def _voice_file(library, store, gid: int, vid: int, which: str) -> Path:
    """ref.wav or source.* of a voice, resolved strictly inside its folder."""
    row = _voice_here(store, gid, vid, allow_speaker=True)
    if row["kind"] == "speaker" and not store.has_consent(row["owner_user_id"]):
        raise KeyError("no consent: their voice stays private")
    folder = library.folder(vid).resolve()
    if which == "ref":
        path = folder / "ref.wav"
    elif which == "source":
        path = _source_file(library, vid)
    else:
        raise KeyError("no such file")
    if path is None:
        raise KeyError("no such file")
    path = path.resolve()
    if not path.is_relative_to(folder) or not path.is_file():
        raise KeyError("no such file")
    return path


@api
async def voice_audio(request: web.Request) -> web.StreamResponse:
    store, library, gid, vid = request.app["store"], request.app["library"], _gid(request), _int(request, "vid")
    path = _voice_file(library, store, gid, vid, request.match_info["which"])
    return web.FileResponse(path, headers={"Content-Type": AUDIO_TYPES.get(path.suffix.lower(),
                                                                           "application/octet-stream")})


def compute_peaks(path: Path, buckets: int, with_segments: bool) -> dict:
    """Max |amplitude| per bucket (0..1), and for a source the stretches of
    speech the automatic selection (voice_library.select_speech) would take."""
    import numpy as np
    from faster_whisper.audio import decode_audio

    import voice_library as vl

    audio = np.asarray(decode_audio(str(path), sampling_rate=vl.VAD_RATE), dtype=np.float32)
    duration = len(audio) / vl.VAD_RATE
    buckets = max(10, min(int(buckets), 4000))
    if len(audio) == 0:
        return {"duration": 0, "peaks": [], "segments": [], "auto": None}
    per = max(1, len(audio) // buckets)
    usable = audio[: per * (len(audio) // per)]
    peaks = np.abs(usable.reshape(-1, per)).max(axis=1)
    top = float(peaks.max()) or 1.0
    out = {"duration": round(duration, 3), "peaks": [round(float(p) / top, 3) for p in peaks], "segments": [],
           "auto": None}
    if with_segments:
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            lines = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100,
                                                            max_speech_duration_s=6))
            total, chosen = 0.0, []
            for line in lines:  # the same walk as select_speech
                start, end = line["start"] / vl.VAD_RATE, line["end"] / vl.VAD_RATE
                if total + (end - start) > vl.REF_MAX_SECONDS:
                    continue
                chosen.append([round(start, 2), round(end, 2)])
                total += end - start
                if total >= vl.REF_SECONDS:
                    break
            out["segments"] = chosen
        except Exception:
            log.exception("speech detection for the waveform failed")
        # What ingest() takes with no selection: the speech found, or (none
        # found) the first REF_MAX_SECONDS, like select_speech's fallback.
        segs = out["segments"]
        out["auto"] = [segs[0][0], segs[-1][1]] if segs else [0.0, round(min(duration, vl.REF_MAX_SECONDS), 2)]
    return out


@api
async def voice_peaks(request: web.Request) -> web.Response:
    store, library, gid, vid = request.app["store"], request.app["library"], _gid(request), _int(request, "vid")
    which = request.query.get("which", "ref")
    path = _voice_file(library, store, gid, vid, which)
    try:
        buckets = int(request.query.get("n", PEAK_BUCKETS))
    except ValueError:
        raise FieldError("n must be a number", "n") from None
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, buckets)
    cache = request.app["peaks_cache"]
    if key not in cache:
        try:
            cache[key] = await asyncio.to_thread(compute_peaks, path, buckets, which == "source")
        except Exception as e:
            raise ValueError(f"couldn't read the audio: {e}") from None
        while len(cache) > PEAKS_CACHE:
            cache.popitem(last=False)
    return json_response({"which": which, **cache[key]})


def _controller(request: web.Request, method: str):
    fn = getattr(request.app["controller"], method, None)
    if fn is None:
        raise web.HTTPNotImplemented(text=f'{{"error": "the bot can\'t {method} here"}}',
                                     content_type="application/json")
    return fn


@api
async def voice_transcribe(request: web.Request) -> web.Response:
    store, gid, vid = request.app["store"], _gid(request), _int(request, "vid")
    row = _voice_here(store, gid, vid)
    if row["kind"] != "clone":
        raise ValueError("only clone voices have a transcript to make")
    try:
        text = await _controller(request, "transcribe_voice")(vid)
    except (RuntimeError, LookupError) as e:
        raise ValueError(str(e)) from None
    if row["status"] in ("ready", "failed") and text != row["ref_text"]:
        store.update_voice(vid, status="draft")
    return json_response({"ref_text": text, "voice": store.get_voice(vid)})


@api
async def voice_build(request: web.Request) -> web.Response:
    store, gid, vid = request.app["store"], _gid(request), _int(request, "vid")
    row = _voice_here(store, gid, vid, allow_speaker=True)
    if row["kind"] == "clone" and not row["ref_text"]:
        raise ValueError("add the transcript first (Auto-transcribe, or type it)")
    message = _controller(request, "build_voice")(vid)
    if asyncio.iscoroutine(message):
        message = await message
    return json_response({"message": message, "voice": store.get_voice(vid)})


@api
async def voice_preview(request: web.Request) -> web.Response:
    store, gid, vid = request.app["store"], _gid(request), _int(request, "vid")
    _voice_here(store, gid, vid, allow_speaker=True)
    body = await _body(request)
    text = str(body.get("text") or "").strip() or str(store.get_setting(gid, "voices.preview_text", "") or "")
    if not text:
        raise FieldError("type something to say", "text")
    wav = await _controller(request, "preview")(text[:300], vid)
    return web.Response(body=wav, content_type="audio/wav")


@api
async def cache_clear(request: web.Request) -> web.Response:
    store, library, gid = request.app["store"], request.app["library"], _gid(request)
    body = await _body(request) if request.can_read_body else {}
    voice_id = to_user_id(body.get("voice_id")) if body.get("voice_id") not in (None, "") else None
    if voice_id is not None:
        _voice_here(store, gid, voice_id, allow_speaker=True)
    await asyncio.to_thread(library.clear_cache, voice_id)
    return json_response({"bytes": await asyncio.to_thread(library.cache_size)})


# ───────────────────────────── people's own voices ─────────────────────────────

def _speaker_row(store, library, gid: int, uid: int) -> dict:
    if store.get_person(gid, uid) is None:
        raise KeyError("nobody with that id in this server")
    rows = [v for v in store.find_voices(kind="speaker", owner_user_id=uid) if v["guild_id"] in (None, gid)]
    if not rows:
        raise KeyError("they have no voice of their own")
    return rows[0]


@api
async def speaker_rebuild(request: web.Request) -> web.Response:
    store, library, gid, uid = request.app["store"], request.app["library"], _gid(request), _int(request, "uid")
    row = _speaker_row(store, library, gid, uid)
    if not store.has_consent(uid):
        raise ValueError("they haven't consented: their voice can't be built")
    message = _controller(request, "build_voice")(row["id"])
    if asyncio.iscoroutine(message):
        message = await message
    return json_response({"message": message})


@api
async def speaker_delete(request: web.Request) -> web.Response:
    store, library, gid, uid = request.app["store"], request.app["library"], _gid(request), _int(request, "uid")
    _speaker_row(store, library, gid, uid)
    count = await asyncio.to_thread(library.delete_speaker, uid)
    return json_response({"deleted": count})


@api
async def speaker_promote(request: web.Request) -> web.Response:
    """Copy a consented person's voice into a regular clone voice of this
    server (from their reference, with its transcript), then queue its build."""
    store, library, gid, uid = request.app["store"], request.app["library"], _gid(request), _int(request, "uid")
    row = _speaker_row(store, library, gid, uid)
    if (store.get_consent(uid) or {}).get("status") != "accepted":
        raise ValueError("only with their consent (accepted)")
    folder = library.folder(row["id"])
    ref, clips = folder / "ref.wav", folder / "clips.wav"
    source = ref if ref.is_file() else clips if clips.is_file() else None
    if source is None:
        raise ValueError("there's none of their audio yet")
    body = await _body(request) if request.can_read_body else {}
    person = store.get_person(gid, uid) or {}
    name = _clean_name(body.get("name") or f"{person.get('nickname') or person.get('display_name') or uid} (voice)")
    if store.find_voice(gid, name):
        raise FieldError(f"there's already a voice called {name}", "name")
    vid = store.add_voice(name, guild_id=gid, kind="clone", status="draft", created_by=None)
    try:
        # The whole reference (≤ 12 s): its transcript still matches.
        await asyncio.to_thread(library.ingest, vid, source, start_s=0.0 if source == ref else None)
    except Exception as e:
        await asyncio.to_thread(library.delete_voice, vid)
        raise ValueError(f"couldn't copy their voice: {e}") from None
    if source == ref and row["ref_text"]:
        store.update_voice(vid, ref_text=row["ref_text"])
    message = None
    build = getattr(request.app["controller"], "build_voice", None)
    if build is not None and store.get_voice(vid)["ref_text"]:
        try:
            message = build(vid)
            if asyncio.iscoroutine(message):
                message = await message
        except ValueError as e:
            message = str(e)
    return json_response({"voice": store.get_voice(vid), "message": message})


# ───────────────────────────── sounds ─────────────────────────────

def _sound_here(store, gid: int, sid: int) -> dict:
    row = store.get_sound(sid)
    if row is None or row["guild_id"] != gid:
        raise KeyError(f"no sound {sid} here")
    return row


def _sound_cap(store, gid: int) -> int:
    return int(float(store.get_setting(gid, "sounds.max_mb", 5) or 5) * 2**20)


@api
async def sound_create(request: web.Request) -> web.Response:
    store, sounds, gid = request.app["store"], request.app["sounds"], _gid(request)
    work = _work_dir(request.app)
    try:
        fields, upload, filename = await _read_form(request, work, _sound_cap(store, gid))
        name = fields.get("name") or (Path(filename).stem if filename else "")
        if not str(name).strip():
            raise FieldError("the sound needs a name", "name")
        if upload is None:
            raise FieldError("choose an audio file", "file")
        try:
            row = await asyncio.to_thread(sounds.add, gid, name, upload, created_by=None, filename=filename)
        except ValueError as e:
            raise FieldError(str(e), "name" if "name" in str(e) or "sound called" in str(e) else "file") from None
        return json_response({"sound": row})
    finally:
        await asyncio.to_thread(shutil.rmtree, work, True)


@api
async def sound_update(request: web.Request) -> web.Response:
    import sound_library

    store, gid, sid = request.app["store"], _gid(request), _int(request, "sid")
    row = _sound_here(store, gid, sid)
    body = await _body(request)
    fields = {}
    if "name" in body:
        try:
            name = sound_library.sound_name(str(body["name"] or ""))
        except ValueError as e:
            raise FieldError(str(e), "name") from None
        other = store.find_sound(gid, name)
        if other and other["id"] != sid:
            raise FieldError(f"there's already a sound called {name}", "name")
        fields["name"] = name
    if "enabled" in body:
        fields["enabled"] = bool(body["enabled"])
    if "gain_db" in body:
        fields["gain_db"] = _float(body["gain_db"], "gain_db", low=-30, high=20) or 0.0
    if fields:
        store.update_sound(row["id"], **fields)
    return json_response({"sound": store.get_sound(sid)})


@api
async def sound_delete(request: web.Request) -> web.Response:
    store, sounds, gid, sid = request.app["store"], request.app["sounds"], _gid(request), _int(request, "sid")
    _sound_here(store, gid, sid)
    await asyncio.to_thread(sounds.delete, sid)
    return json_response({"ok": True})


@api
async def sound_audio(request: web.Request) -> web.StreamResponse:
    import sound_library

    store, sounds, gid, sid = request.app["store"], request.app["sounds"], _gid(request), _int(request, "sid")
    row = _sound_here(store, gid, sid)
    if not row["path"]:
        raise KeyError("the sound has no file")
    path = Path(row["path"])
    path = (path if path.is_absolute() else Path(sound_library.ROOT) / path).resolve()
    folder = sounds.folder(gid).resolve()
    if not path.is_relative_to(folder) or not path.is_file():
        raise KeyError("no such file")
    return web.FileResponse(path, headers={"Content-Type": AUDIO_TYPES.get(path.suffix.lower(),
                                                                           "application/octet-stream")})


@api
async def sound_reaction(request: web.Request) -> web.Response:
    import reactions as rules

    store, gid, sid = request.app["store"], _gid(request), _int(request, "sid")
    row = _sound_here(store, gid, sid)
    body = await _body(request)
    trigger = body.get("trigger") or {"type": "slash", "name": row["name"]}
    kind = trigger.get("type") if isinstance(trigger, dict) else None
    if kind in ("phrase", "command"):
        phrases = trigger.get("phrases")
        phrases = [phrases] if isinstance(phrases, str) else phrases or []
        trigger = {"type": kind, "phrases": [str(p).strip() for p in phrases if str(p).strip()]}
    elif kind == "slash":
        trigger = {"type": "slash", "name": str(trigger.get("name") or row["name"]).strip().lstrip("/")}
    else:
        raise FieldError("the trigger is a phrase, a command or a slash name", "trigger")
    options = [[{"type": "sound", "sound_id": sid}]]
    try:
        rules.validate([trigger], options)
    except ValueError as e:
        raise FieldError(str(e), "trigger") from None
    name = str(body.get("name") or row["name"]).strip()
    rid = store.add_reaction(gid, "sound", name, [trigger], options)
    return json_response({"id": rid, "reaction": store.get_reaction(rid)})


@api
async def sound_play(request: web.Request) -> web.Response:
    store, gid, sid = request.app["store"], _gid(request), _int(request, "sid")
    _sound_here(store, gid, sid)
    message = await _controller(request, "play_sound")(gid, sid)
    return json_response({"message": message})


@api
async def board(request: web.Request) -> web.Response:
    """What the soundboard on the status view needs: playable sounds and voices."""
    store, gid = request.app["store"], _gid(request)
    sounds = [{"id": s["id"], "name": s["name"], "duration_s": s["duration_s"]}
              for s in store.list_sounds(gid) if s["enabled"] and request.app.get("sounds") is not None]
    voices = [{"id": v["id"], "name": v["name"], "status": v["status"], "kind": v["kind"]}
              for v in store.list_voices(gid) if v["kind"] != "speaker"]
    library = request.app.get("library")
    return json_response({"sounds": sounds, "voices": voices,
                          "bot_voice_id": library.bot_voice_id() if library is not None else None})
