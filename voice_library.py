"""Saved voices: each one is built once, and every line it says is
generated once.

A voice lives in the store (the `voices` table) plus a folder:

    data/voices/<voice_id>/
        source.<ext>    what it was cloned from (a file someone uploaded, the bot's mp3)
        ref.wav         the reference actually encoded: the best ~10 s of speech, 24 kHz mono
        prompt.pt       the encoded OmniVoice prompt (~15 KB), loaded on demand
        clips.wav/.json speaker voices only: the person's latest sentences and their text
        meta.json       bookkeeping (e.g. which bot voice file it was made from)

    data/cache/tts/<voice_id>/<key>.flac   every generated line, kept across restarts

Kinds:
  clone     from an audio file: ingest() -> transcribe_ref() -> build()
  designed  no audio: the voice is its `instruct` text (+ speed); nothing to encode
  speaker   someone's own voice, collected from what they say in calls
            (update_speaker), only after they consented. Encoded once, not on
            every line as the old !userclone did (a ~1.3 GB GPU spike each time).

The model is attached from outside (attach()), so this module never imports
torch itself and the tests run with a fake model. Everything that touches the
model is a plain blocking call, meant for the bot's single TTS thread; the
bookkeeping (files, cache, loaded prompts) is behind one lock, the model calls
are outside it.
"""
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import soxr

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RATE = 24000          # OmniVoice's native rate: references and cached lines are stored at it
VAD_RATE = 16000      # what VAD and speech-to-text take
REF_SECONDS = 10.0    # longer references sound no better and slow every generation
REF_MAX_SECONDS = 12.0  # hard cap: encoding needs ~0.22 GB of GPU memory per second
# A speaker voice is encoded while Whisper is loaded too, so it stays smaller
# (6 s: ~1.3 GB burst), like the old !userclone reference.
SPEAKER_REF_SECONDS = 6.0
REF_RMS = 0.1         # references are leveled to this (-20 dBFS), peaks kept under 1
DEFAULT_NUM_STEP = 16  # ~0.5 s for a short line on an RTX 3070 Ti
DEFAULT_MODEL_NAME = "k2-fsa/OmniVoice"

# Speaker clips: the newest ~10 s of what someone said, sentence by sentence.
CLIP_TARGET_SECONDS = 10.0
MIN_CLIP_SECONDS = 1.0
MIN_CLIP_WORDS = 2
BUILD_AT_SECONDS = 8.0  # a speaker voice is first built once this much was heard
GAP_SECONDS = 0.3


class VoiceNotReady(RuntimeError):
    """The voice has no prompt yet (never built) or is missing."""


def to_discord_pcm(audio: np.ndarray, rate: int = RATE) -> bytes:
    """Mono float audio -> Discord PCM (48 kHz stereo s16)."""
    audio = soxr.resample(np.asarray(audio, dtype=np.float32), rate, 48000) if rate != 48000 else audio
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    return np.repeat(pcm, 2).tobytes()


def _sha(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resample(audio: np.ndarray, rate: int, target: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:  # (samples, channels)
        audio = audio.mean(axis=1)
    return audio if rate == target else soxr.resample(audio, rate, target).astype(np.float32)


def _join(chunks: list[np.ndarray], rate: int) -> np.ndarray:
    gap = np.zeros(int(GAP_SECONDS * rate), dtype=np.float32)
    parts = [part for chunk in chunks for part in (chunk, gap)][:-1]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def level(audio: np.ndarray) -> np.ndarray:
    """Loudness to REF_RMS, without letting peaks clip."""
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    if rms <= 1e-6:
        return audio
    gain = REF_RMS / rms
    peak = float(np.max(np.abs(audio))) * gain
    if peak > 0.99:
        gain *= 0.99 / peak
    return (audio * gain).astype(np.float32)


def select_speech(audio: np.ndarray, *, target_s: float = REF_SECONDS, max_s: float = REF_MAX_SECONDS) -> np.ndarray:
    """The first ~target_s of speech in 24 kHz audio, found by voice activity
    detection, lines joined with short pauses. Never longer than max_s.
    When VAD finds nothing, the start of the audio."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps  # CPU only

    audio16 = _resample(audio, RATE, VAD_RATE)
    # Long stretches without a pause are split into pieces of at most 6 s.
    lines = get_speech_timestamps(
        audio16, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100, max_speech_duration_s=6)
    )
    parts, total = [], 0.0
    for line in lines:
        start, end = line["start"] / VAD_RATE, line["end"] / VAD_RATE
        if total + (end - start) > max_s:
            continue  # would make the reference too big to encode
        parts.append(audio[int(start * RATE): int(end * RATE)])
        total += end - start
        if total >= target_s:
            break
    if not parts:
        return audio[: int(max_s * RATE)]
    return _join(parts, RATE)[: int(max_s * RATE)]


class VoiceLibrary:
    def __init__(self, store, root: str | Path = ROOT / "data", *, language: str | None = None,
                 num_step: int = DEFAULT_NUM_STEP):
        """language: what lines are generated in unless a voice says
        otherwise (e.g. "es"); None lets the model guess."""
        self.store = store
        self.root = Path(root)
        self.voices_dir = self.root / "voices"
        self.cache_dir = self.root / "cache" / "tts"
        self.language = language
        self.num_step = num_step
        self.model = None
        self.model_name = DEFAULT_MODEL_NAME
        self._transcribe: Callable[[np.ndarray], str] | None = None
        self._load_prompt: Callable | None = None
        self._free_memory: Callable[[], None] | None = None
        self._lock = threading.RLock()
        self._prompts: dict[int, object] = {}
        self._cache_bytes: int | None = None  # estimate; None = not scanned yet
        store.on_change(self._changed)

    def attach(self, model, transcribe: Callable[[np.ndarray], str] | None = None, *,
               model_name: str | None = None, load_prompt: Callable | None = None,
               free_memory: Callable[[], None] | None = None) -> None:
        """Use this model from now on.
        transcribe: 16 kHz mono float -> text, for references without one.
        load_prompt(path, map_location): defaults to OmniVoice's VoiceClonePrompt.load.
        free_memory: called after every model call (e.g. torch.cuda.empty_cache,
        so Whisper can use the memory)."""
        self.model = model
        self.model_name = model_name or getattr(model, "name_or_path", None) or DEFAULT_MODEL_NAME
        self._transcribe = transcribe
        self._load_prompt = load_prompt
        self._free_memory = free_memory
        with self._lock:
            self._prompts.clear()  # loaded for another model / device

    # ------------------------------------------------------------ paths
    def folder(self, voice_id: int) -> Path:
        return self.voices_dir / str(int(voice_id))

    def _stored_path(self, path: Path) -> str:
        """Relative to the repo when inside it (portable), else absolute."""
        path = path.resolve()
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    @staticmethod
    def _disk_path(stored: str | None) -> Path | None:
        if not stored:
            return None
        path = Path(stored)
        return path if path.is_absolute() else ROOT / path

    def _meta(self, voice_id: int) -> dict:
        path = self.folder(voice_id) / "meta.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _set_meta(self, voice_id: int, **values) -> None:
        folder = self.folder(voice_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "meta.json").write_text(json.dumps({**self._meta(voice_id), **values}), encoding="utf-8")

    def _voice(self, voice_id: int) -> dict:
        row = self.store.get_voice(voice_id)
        if row is None:
            raise KeyError(f"no voice {voice_id}")
        return row

    def _changed(self, table: str, guild_id) -> None:
        if table != "voices":
            return
        with self._lock:
            for voice_id in list(self._prompts):
                if self.store.get_voice(voice_id) is None:
                    self._prompts.pop(voice_id, None)

    # ------------------------------------------------------------ references
    def ingest(self, voice_id: int, audio: str | Path | np.ndarray, rate: int | None = None, *,
               start_s: float | None = None, end_s: float | None = None) -> dict:
        """Save the voice's source and make its reference (ref.wav): the
        start_s..end_s selection if given, otherwise the best ~10 s of speech.
        Capped at 12 s and leveled. The voice goes back to draft (its old
        prompt keeps working until the next build). Returns the row."""
        self._voice(voice_id)
        folder = self.folder(voice_id)
        folder.mkdir(parents=True, exist_ok=True)
        if isinstance(audio, (str, Path)):
            from faster_whisper.audio import decode_audio  # any format PyAV reads

            source_file = Path(audio)
            source = folder / f"source{source_file.suffix.lower() or '.audio'}"
            if source_file.resolve() != source.resolve():
                for old in folder.glob("source.*"):
                    old.unlink()
                shutil.copyfile(source_file, source)
            samples = decode_audio(str(source), sampling_rate=RATE)
        else:
            if rate is None:
                raise ValueError("rate is required for raw audio")
            samples = _resample(audio, rate, RATE)
            for old in folder.glob("source.*"):
                old.unlink()
            source = folder / "source.wav"
            sf.write(source, samples, RATE, subtype="PCM_16")
        if start_s is not None or end_s is not None:
            start = int((start_s or 0) * RATE)
            end = int(end_s * RATE) if end_s is not None else len(samples)
            ref = samples[start:end][: int(REF_MAX_SECONDS * RATE)]
        else:
            ref = select_speech(samples)
        if len(ref) < int(0.5 * RATE):
            raise ValueError("the selection has less than half a second of audio")
        self._write_ref(voice_id, ref)
        self.store.update_voice(voice_id, source_path=self._stored_path(source), status="draft", error=None)
        return self._voice(voice_id)

    def _write_ref(self, voice_id: int, ref: np.ndarray) -> Path:
        path = self.folder(voice_id) / "ref.wav"
        sf.write(path, level(ref), RATE, subtype="PCM_16")
        self.store.update_voice(voice_id, ref_path=self._stored_path(path))
        return path

    def _ref_audio(self, row: dict) -> np.ndarray:
        path = self._disk_path(row["ref_path"]) or self.folder(row["id"]) / "ref.wav"
        if not path.is_file():
            raise VoiceNotReady(f"voice {row['id']} ({row['name']}) has no reference audio")
        audio, rate = sf.read(path, dtype="float32")
        return _resample(audio, rate, RATE)

    def transcribe_ref(self, voice_id: int, *, force: bool = False) -> str:
        """Fill ref_text from the reference audio, unless someone already
        typed it (force: redo it anyway). Blocking (speech-to-text)."""
        row = self._voice(voice_id)
        if row["ref_text"] and not force:
            return row["ref_text"]
        if self._transcribe is None:
            raise RuntimeError("no transcribe function attached")
        text = (self._transcribe(_resample(self._ref_audio(row), RATE, VAD_RATE)) or "").strip()
        if not text:
            raise ValueError(f"couldn't transcribe voice {voice_id}'s reference")
        self.store.update_voice(voice_id, ref_text=text)
        return text

    # ------------------------------------------------------------ building
    def identity(self, row: dict) -> str | None:
        """What makes this voice sound the way it does, as a hash: the cache key's voice part."""
        if row["kind"] == "designed":
            return _sha("designed", row["instruct"] or "", row["speed"], self.model_name)
        return row["prompt_hash"]

    def _ref_hash(self, row: dict) -> str:
        path = self._disk_path(row["ref_path"]) or self.folder(row["id"]) / "ref.wav"
        return _sha(row["kind"], _file_sha(path), row["ref_text"] or "", self.model_name)

    def build_due(self, voice_id: int) -> bool:
        """Does build() have anything to do?"""
        row = self._voice(voice_id)
        if row["kind"] == "designed":
            return row["status"] != "ready"
        prompt = self.folder(voice_id) / "prompt.pt"
        if row["status"] == "ready" and prompt.is_file():
            return False
        return True

    def build(self, voice_id: int) -> bool:
        """Encode the voice and save its prompt. Blocking, on the model.
        Returns False when nothing changed since the last build (skipped).
        A failure marks the voice failed (with the error) and raises."""
        row = self._voice(voice_id)
        if row["kind"] == "designed":
            if not (row["instruct"] or "").strip():
                self.store.update_voice(voice_id, status="failed", error="a designed voice needs instruct")
                raise ValueError("a designed voice needs instruct")
            self.store.update_voice(voice_id, status="ready", error=None, prompt_hash=self.identity(row))
            return True
        if self.model is None:
            raise RuntimeError("no model attached")
        try:
            if row["kind"] == "speaker":
                self._speaker_ref(voice_id)
            elif not (self.folder(voice_id) / "ref.wav").is_file():
                source = self._disk_path(row["source_path"])
                if source is None or not source.is_file():
                    raise VoiceNotReady(f"voice {voice_id} has no audio to clone")
                self.ingest(voice_id, source)  # e.g. a voice that came in a pack
            self.transcribe_ref(voice_id)
            row = self._voice(voice_id)
            new_hash = self._ref_hash(row)
            prompt_path = self.folder(voice_id) / "prompt.pt"
            if row["prompt_hash"] == new_hash and prompt_path.is_file():
                self.store.update_voice(voice_id, status="ready", error=None)
                return False
            self.store.update_voice(voice_id, status="building", error=None)
            ref = self._ref_audio(row)
            try:
                prompt = self.model.create_voice_clone_prompt(
                    (ref[np.newaxis, :], RATE), ref_text=row["ref_text"])
            finally:
                self._after_model()
            tmp = prompt_path.with_suffix(".tmp")
            prompt.save(str(tmp))
            os.replace(tmp, prompt_path)
        except Exception as e:
            self.store.update_voice(voice_id, status="failed", error=f"{type(e).__name__}: {e}")
            raise
        self.unload(voice_id)
        self.clear_cache(voice_id)  # lines in the old voice
        self.store.update_voice(voice_id, status="ready", error=None, prompt_hash=new_hash,
                                prompt_path=self._stored_path(prompt_path))
        return True

    def pending_builds(self) -> list[int]:
        """Voices waiting for a build (queued, or interrupted mid-build): requeue them at startup."""
        return sorted(v["id"] for v in self.store.find_voices(statuses=("queued", "building")))

    def request_rebuild(self, voice_id: int) -> bool:
        """Mark a voice for building (the admin or its owner asked, or consent
        just came in). True when there's something to build it from: queue
        build() at "build" priority then."""
        row = self._voice(voice_id)
        folder = self.folder(voice_id)
        if row["kind"] == "speaker":
            has_audio = self._clip_seconds(voice_id) > 0
        elif row["kind"] == "designed":
            has_audio = bool(row["instruct"])
        else:
            has_audio = (folder / "ref.wav").is_file() or bool(
                self._disk_path(row["source_path"]) and self._disk_path(row["source_path"]).is_file())
        if has_audio:
            self.store.update_voice(voice_id, status="queued", error=None)
        return has_audio

    # ------------------------------------------------------------ prompts
    def prompt(self, voice_id: int):
        """The voice's encoded prompt, loaded once and kept on the model's device."""
        with self._lock:
            if voice_id in self._prompts:
                return self._prompts[voice_id]
        path = self.folder(voice_id) / "prompt.pt"
        if not path.is_file():
            raise VoiceNotReady(f"voice {voice_id} hasn't been built")
        load = self._load_prompt
        if load is None:
            from omnivoice.models.omnivoice import VoiceClonePrompt  # imports torch

            load = VoiceClonePrompt.load
        prompt = load(str(path), map_location=str(getattr(self.model, "device", "cpu")))
        row = self.store.get_voice(voice_id)
        if row is not None and not row["ref_text"] and getattr(prompt, "ref_text", None):
            self.store.update_voice(voice_id, ref_text=prompt.ref_text)  # an imported prompt knows its text
        with self._lock:
            self._prompts[voice_id] = prompt
        return prompt

    def unload(self, voice_id: int) -> None:
        with self._lock:
            self._prompts.pop(voice_id, None)

    # ------------------------------------------------------------ speaking
    def bot_voice_id(self) -> int | None:
        voice_id = self.store.get_setting(0, "voice.bot")
        return voice_id if voice_id is not None and self.store.get_voice(voice_id) else None

    def _resolve(self, voice_id: int | None) -> dict:
        if voice_id is None:
            voice_id = self.bot_voice_id()
            if voice_id is None:
                raise VoiceNotReady("the bot has no voice yet (ensure_bot_voice)")
        return self._voice(voice_id)

    def _settings(self, row: dict, num_step: int | None, speed: float | None,
                  language: str | None = None) -> tuple[int, float | None, str | None]:
        """(steps, speed, language). The language: the voice's own if it has
        one, else the asked one (the server's), else the library default."""
        return (int(num_step or row["num_step"] or self.num_step),
                speed if speed is not None else row["speed"],
                row["language"] or language or self.language)

    def cache_path(self, text: str, voice_id: int | None = None, *, num_step: int | None = None,
                   speed: float | None = None, language: str | None = None) -> Path:
        row = self._resolve(voice_id)
        identity = self.identity(row)
        if identity is None:
            raise VoiceNotReady(f"voice {row['id']} ({row['name']}) hasn't been built")
        steps, speed, language = self._settings(row, num_step, speed, language)
        key = _sha(identity, text, steps, speed, language, self.model_name)[:40]
        return self.cache_dir / str(row["id"]) / f"{key}.flac"

    def cached(self, text: str, voice_id: int | None = None, *, num_step: int | None = None,
               speed: float | None = None, language: str | None = None) -> np.ndarray | None:
        """The line if it was generated before, else None. Never touches the
        model, so it's fine to call from the event loop."""
        try:
            path = self.cache_path(text, voice_id, num_step=num_step, speed=speed, language=language)
        except (VoiceNotReady, KeyError):
            return None
        row = self._resolve(voice_id)
        try:
            audio, _ = sf.read(path, dtype="float32")
            os.utime(path)  # recently used: evicted last
        except (OSError, RuntimeError):  # missing, or a half-written file from a crash
            return None
        return self._gain(audio, row)

    def speak(self, text: str, voice_id: int | None = None, *, num_step: int | None = None,
              speed: float | None = None, language: str | None = None) -> np.ndarray:
        """`text` in this voice (None: the bot's), 24 kHz mono float. From the
        disk cache when it was said before, else generated (blocking, on the
        model) and cached. language: what to speak in (the server's) unless
        the voice has its own."""
        hit = self.cached(text, voice_id, num_step=num_step, speed=speed, language=language)
        if hit is not None:
            return hit
        if self.model is None:
            raise RuntimeError("no model attached")
        row = self._resolve(voice_id)
        path = self.cache_path(text, voice_id, num_step=num_step, speed=speed, language=language)
        steps, speed, language = self._settings(row, num_step, speed, language)
        kwargs = {"text": text, "language": language, "num_step": steps}
        if speed is not None:
            kwargs["speed"] = speed
        if row["kind"] == "designed":
            kwargs["instruct"] = row["instruct"]
        else:
            kwargs["voice_clone_prompt"] = self.prompt(row["id"])
        try:
            audio = np.asarray(self.model.generate(**kwargs)[0], dtype=np.float32).reshape(-1)
        finally:
            self._after_model()
        self._store_cached(path, audio)
        return self._gain(audio, row)

    @staticmethod
    def _gain(audio: np.ndarray, row: dict) -> np.ndarray:
        # Applied on the way out, so changing a voice's gain doesn't invalidate its cache.
        if not row["gain_db"]:
            return audio
        return np.clip(audio * 10 ** (row["gain_db"] / 20), -1, 1).astype(np.float32)

    def _after_model(self) -> None:
        if self._free_memory is not None:
            try:
                self._free_memory()
            except Exception:
                log.exception("free_memory failed")

    # ------------------------------------------------------------ the cache
    def _store_cached(self, path: Path, audio: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.flac")
        os.close(fd)
        try:
            sf.write(tmp, audio, RATE, format="FLAC", subtype="PCM_16")
            os.replace(tmp, path)
        except OSError as e:
            log.warning("Couldn't cache %s: %s", path.name, e)
            Path(tmp).unlink(missing_ok=True)
            return
        with self._lock:
            if self._cache_bytes is None:
                self._cache_bytes = self._scan_cache()[1]
            else:
                self._cache_bytes += path.stat().st_size
            if self._cache_bytes > self.cache_limit():
                self.evict()

    def cache_limit(self) -> int:
        return int(float(self.store.get_setting(0, "cache.tts_mb", 500)) * 1024 * 1024)

    def _scan_cache(self) -> tuple[list[tuple[float, int, Path]], int]:
        files = []
        if self.cache_dir.is_dir():
            for path in self.cache_dir.rglob("*.flac"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                files.append((stat.st_mtime, stat.st_size, path))
        return files, sum(size for _, size, _ in files)

    def cache_size(self) -> int:
        """Bytes in the audio cache."""
        return self._scan_cache()[1]

    def evict(self, limit: int | None = None) -> int:
        """Delete the least recently used lines until the cache is under 90%
        of its limit. Returns how many files went."""
        limit = self.cache_limit() if limit is None else limit
        with self._lock:
            files, total = self._scan_cache()
            removed = 0
            for _, size, path in sorted(files, key=lambda f: f[0]):
                if total <= limit * 0.9:
                    break
                path.unlink(missing_ok=True)
                total -= size
                removed += 1
            self._cache_bytes = total
            return removed

    def clear_cache(self, voice_id: int | None = None) -> None:
        """Forget generated lines: one voice's, or all of them."""
        with self._lock:
            target = self.cache_dir if voice_id is None else self.cache_dir / str(int(voice_id))
            shutil.rmtree(target, ignore_errors=True)
            self._cache_bytes = None

    # ------------------------------------------------------------ the bot's own voice
    def ensure_bot_voice(self, path: str | Path, *, name: str | None = None,
                         prompt_file: str | Path | None = None) -> int:
        """The bot's own voice as a regular, global voice (setting voice.bot).
        Made from `path` (e.g. BOT_VOICE_FILE) and remade when that file
        changes (size or date). prompt_file: an already-built OmniVoice prompt
        for this same file, reused instead of rebuilding.
        Returns its id; then build() it if build_due()."""
        path = Path(path)
        stat = path.stat()
        signature = f"{stat.st_size}-{int(stat.st_mtime)}"
        name = name or self.store.get_setting(0, "bot.name") or "bot"
        voice_id = self.bot_voice_id()
        if voice_id is None:
            voice_id = self.store.add_voice(name, guild_id=None, kind="clone", status="draft")
            self.store.set_setting(0, "voice.bot", voice_id)
        elif self._voice(voice_id)["name"] != name:
            self.store.update_voice(voice_id, name=name)
        if self._meta(voice_id).get("source_signature") == signature and (self.folder(voice_id) / "ref.wav").is_file():
            return voice_id

        self.ingest(voice_id, path)
        self.store.update_voice(voice_id, ref_text=None)  # new audio, new transcript
        self._set_meta(voice_id, source_signature=signature, source=str(path))
        if prompt_file is not None and Path(prompt_file).is_file():
            target = self.folder(voice_id) / "prompt.pt"
            shutil.copyfile(prompt_file, target)
            self.unload(voice_id)
            self.clear_cache(voice_id)
            self.store.update_voice(voice_id, status="ready", prompt_path=self._stored_path(target),
                                    prompt_hash=_sha("imported", _file_sha(target), self.model_name))
        return voice_id

    # ------------------------------------------------------------ people's own voices
    def _speaker_rows(self, user_id: int) -> list[dict]:
        return self.store.find_voices(kind="speaker", owner_user_id=user_id)

    def speaker_voice(self, guild_id: int | None, user_id: int, *, ready_only: bool = True) -> dict | None:
        """Their own voice, only if they consented (and, by default, only once
        it's usable). A server-specific one wins over their global one."""
        if not self.store.has_consent(user_id, "voice"):
            return None
        for row in self._speaker_rows(user_id):
            if row["guild_id"] not in (None, guild_id):
                continue
            if ready_only and not (self.folder(row["id"]) / "prompt.pt").is_file():
                continue
            return row
        return None

    def _clips(self, voice_id: int) -> list[tuple[np.ndarray, str]]:
        folder = self.folder(voice_id)
        wav, meta = folder / "clips.wav", folder / "clips.json"
        if not (wav.is_file() and meta.is_file()):
            return []
        try:
            audio, rate = sf.read(wav, dtype="float32")
            audio = _resample(audio, rate, RATE)
            clips, offset = [], 0
            for length, text in json.loads(meta.read_text(encoding="utf-8")):
                clips.append((audio[offset: offset + length], text))
                offset += length
            return clips
        except (OSError, ValueError, RuntimeError) as e:
            log.warning("Couldn't read voice %s's clips: %s", voice_id, e)
            return []

    def _save_clips(self, voice_id: int, clips: list[tuple[np.ndarray, str]]) -> None:
        folder = self.folder(voice_id)
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "clips.wav", np.concatenate([a for a, _ in clips]), RATE, subtype="PCM_16")
        (folder / "clips.json").write_text(json.dumps([[len(a), t] for a, t in clips], ensure_ascii=False),
                                           encoding="utf-8")

    def _clip_seconds(self, voice_id: int) -> float:
        meta = self.folder(voice_id) / "clips.json"
        try:
            return sum(length for length, _ in json.loads(meta.read_text(encoding="utf-8"))) / RATE
        except (OSError, ValueError):
            return 0.0

    def update_speaker(self, user_id: int, audio: np.ndarray, text: str, *, rate: int = RATE,
                       name: str | None = None) -> bool:
        """Keep one more sentence of theirs (audio + its transcript), only with
        their consent: without it nothing is stored at all. Their newest
        ~10 s are kept; clips under 1 s or 2 words are dropped. Returns True
        when their voice should be built now (the first time there's enough
        audio); queue build() at "build" priority then. Not on every clip."""
        if not self.store.has_consent(user_id, "voice"):
            return False
        audio = _resample(audio, rate, RATE)
        if len(audio) < MIN_CLIP_SECONDS * RATE or len((text or "").split()) < MIN_CLIP_WORDS:
            return False
        with self._lock:
            rows = [r for r in self._speaker_rows(user_id) if r["guild_id"] is None]
            if rows:
                row = rows[0]
            else:
                voice_id = self.store.add_voice(name or f"speaker {user_id}", guild_id=None, kind="speaker",
                                                owner_user_id=user_id, created_by=user_id, status="draft")
                row = self._voice(voice_id)
            clips = self._clips(row["id"])
            clips.append((audio, text.strip()))
            # Drop the oldest clips while the rest still reach the target length.
            while len(clips) > 1 and sum(len(a) for a, _ in clips[1:]) >= CLIP_TARGET_SECONDS * RATE:
                clips.pop(0)
            self._save_clips(row["id"], clips)
            enough = sum(len(a) for a, _ in clips) >= BUILD_AT_SECONDS * RATE
            first_time = row["status"] == "draft" and not (self.folder(row["id"]) / "prompt.pt").is_file()
            if enough and first_time:
                self.store.update_voice(row["id"], status="queued")
                return True
            return False

    def _speaker_ref(self, voice_id: int) -> None:
        """A speaker's reference: their newest clips that fit SPEAKER_REF_SECONDS, and their text."""
        clips = self._clips(voice_id)
        if not clips:
            raise VoiceNotReady(f"voice {voice_id} has no clips")
        limit, gap = int(SPEAKER_REF_SECONDS * RATE), int(GAP_SECONDS * RATE)
        chosen, total = [], -gap  # the pauses between clips count too
        for audio, text in reversed(clips):
            if len(audio) > limit:
                continue
            if total + gap + len(audio) > limit:
                break
            chosen.insert(0, (audio, text))
            total += gap + len(audio)
        if not chosen:  # only long clips: the end of the newest one
            audio, text = clips[-1]
            chosen = [(audio[-limit:], text)]
        self._write_ref(voice_id, _join([a for a, _ in chosen], RATE))
        self.store.update_voice(voice_id, ref_text=" ".join(t for _, t in chosen))

    def request_speaker_rebuild(self, user_id: int) -> int | None:
        """Their voice id when a rebuild was queued (consent required)."""
        row = self.speaker_voice(None, user_id, ready_only=False)
        if row is not None and self.request_rebuild(row["id"]):
            return row["id"]
        return None

    def delete_speaker(self, user_id: int) -> int:
        """Everything of their voice: clips, prompt, cached lines and rows
        (for /voice delete and a revoked consent). Returns how many voices went."""
        rows = self._speaker_rows(user_id)
        for row in rows:
            self.delete_voice(row["id"])
        return len(rows)

    def delete_voice(self, voice_id: int) -> None:
        """The voice's files, its cached lines and its row."""
        self.unload(voice_id)
        self.clear_cache(voice_id)
        with self._lock:
            shutil.rmtree(self.folder(voice_id), ignore_errors=True)
        if self.store.get_setting(0, "voice.bot") == voice_id:
            self.store.delete_setting(0, "voice.bot")
        self.store.delete_voice(voice_id)
