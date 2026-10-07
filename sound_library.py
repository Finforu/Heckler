"""Sounds people upload (/sound add): cleaned up once on the way in, then
played as they are.

On the way in a sound is decoded (any format ffmpeg/PyAV reads), its silent
start and end are cut, it's capped in length (setting sounds.max_seconds) and
leveled to about the loudness of the bot's speech, so a meme sound doesn't
blast the call. It's stored as data/sounds/<guild_id>/<sound_id>.flac, 48 kHz
mono: Discord's rate, so playing it is just a copy to stereo.

Each sound has its own volume (gain_db in the store, shown to people as a
percentage of that even level: 100% = as uploaded), and the server's
sounds.max_volume caps every one of them when it plays. Peaks are never
pushed past full scale, so a boost can't distort.
"""
import io
import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

from transcriber import normalize
from voice_library import level

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RATE = 48000
SILENCE_DB = -45.0      # quieter than this, at the edges, is cut
EDGE_PAD_S = 0.05       # kept around what's left, so nothing starts mid-breath
FADE_S = 0.05           # fade-out when a sound is cut short
PCM_CACHE_SIZE = 32
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_VOLUME = 400        # percent: nothing, not even the server cap, goes higher
SILENT_DB = -60.0       # what 0% is stored as


class SoundError(ValueError):
    """A sound refused, for a reason to tell the person: `key` is its text in
    locales/*/commands.json (str() is the English one, for logs)."""

    def __init__(self, key: str, **values):
        import i18n

        self.key, self.values = key, values
        super().__init__(i18n.t(key, "en", **values))


def sound_name(name: str) -> str:
    """'Air Horn!' -> 'air-horn'. SoundError when nothing usable is left."""
    slug = "-".join(normalize(name).split())
    if not NAME_RE.match(slug):
        raise SoundError("sound.bad_name")
    return slug


def trim_silence(audio: np.ndarray, rate: int = RATE) -> np.ndarray:
    """Cut the silent start and end (10 ms frames under SILENCE_DB)."""
    frame = max(1, rate // 100)
    usable = len(audio) // frame * frame
    if usable == 0:
        return audio
    frames = audio[:usable].reshape(-1, frame)
    loud = np.flatnonzero(np.sqrt(np.mean(frames ** 2, axis=1)) > 10 ** (SILENCE_DB / 20))
    if len(loud) == 0:
        return audio[:0]
    pad = int(EDGE_PAD_S * rate)
    start = max(0, loud[0] * frame - pad)
    end = min(len(audio), (loud[-1] + 1) * frame + pad)
    return audio[start:end]


def decode(data: bytes | str | Path) -> np.ndarray:
    """Any audio file (or its bytes) -> 48 kHz mono float32. ValueError if it isn't audio."""
    from faster_whisper.audio import decode_audio  # PyAV, which bundles ffmpeg's decoders

    source = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else str(data)
    try:
        audio = decode_audio(source, sampling_rate=RATE)
    except Exception as e:  # PyAV raises a zoo of errors for non-audio input
        log.debug("Couldn't decode an upload: %s: %s", type(e).__name__, e)
        raise SoundError("sound.not_audio") from None
    return np.asarray(audio, dtype=np.float32)


def volume_to_db(percent: float) -> float:
    """100 (%) -> 0.0 dB, 50 -> -6.02, 0 -> SILENT_DB."""
    percent = float(percent)
    if not 0 <= percent <= MAX_VOLUME:
        raise ValueError(f"volume must be between 0 and {MAX_VOLUME}%")
    return SILENT_DB if percent == 0 else max(SILENT_DB, round(20 * float(np.log10(percent / 100)), 2))


def db_to_volume(gain_db: float | None) -> int:
    """0.0 dB -> 100 (%); SILENT_DB or lower -> 0."""
    gain_db = float(gain_db or 0)
    return 0 if gain_db <= SILENT_DB else int(round(100 * 10 ** (gain_db / 20)))


def effective_db(gain_db: float | None, max_volume) -> float:
    """A sound's gain once the server's cap (sounds.max_volume, %) is applied."""
    gain_db = float(gain_db or 0)
    if max_volume is None:
        return gain_db
    return min(gain_db, volume_to_db(min(max(float(max_volume), 0), MAX_VOLUME)))


def to_discord_pcm(audio: np.ndarray) -> bytes:
    """48 kHz mono float -> Discord PCM (48 kHz stereo s16)."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    return np.repeat(pcm, 2).tobytes()


class SoundLibrary:
    def __init__(self, store, root: str | Path = ROOT / "data"):
        self.store = store
        self.root = Path(root)
        self._lock = threading.Lock()
        self._pcm: OrderedDict[tuple[int, float], bytes] = OrderedDict()

    def folder(self, guild_id: int) -> Path:
        return self.root / "sounds" / str(int(guild_id))

    def _stored_path(self, path: Path) -> str:
        path = path.resolve()
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    @staticmethod
    def _disk_path(stored: str) -> Path:
        path = Path(stored)
        return path if path.is_absolute() else ROOT / path

    def prepare(self, guild_id: int, data: bytes | str | Path) -> np.ndarray:
        """Decode, trim, cap and level. SoundError when it can't be used."""
        max_mb = float(self.store.get_setting(guild_id, "sounds.max_mb"))
        size = len(data) if isinstance(data, (bytes, bytearray)) else Path(data).stat().st_size
        if size > max_mb * 1024 * 1024:
            raise SoundError("common.file_too_big", mb=f"{max_mb:g}")
        audio = trim_silence(decode(data))
        if len(audio) < int(0.1 * RATE):
            raise SoundError("sound.silent")
        max_samples = int(float(self.store.get_setting(guild_id, "sounds.max_seconds")) * RATE)
        if len(audio) > max_samples:
            audio = audio[:max_samples].copy()
            fade = min(len(audio), int(FADE_S * RATE))
            audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        return level(audio)

    def add(self, guild_id: int, name: str, data: bytes | str | Path, *, created_by: int | None,
            filename: str | None = None) -> dict:
        """Clean up and save a new sound. Returns its row. ValueError (with a
        message for the user: SoundError) on a bad name, a taken name or bad audio."""
        name = sound_name(name)
        if self.store.find_sound(guild_id, name):
            raise SoundError("sound.taken", name=name)
        audio = self.prepare(guild_id, data)
        folder = self.folder(guild_id)
        folder.mkdir(parents=True, exist_ok=True)
        sound_id = self.store.add_sound(guild_id, name, "", duration_s=round(len(audio) / RATE, 2),
                                        created_by=created_by)
        path = folder / f"{sound_id}.flac"
        try:
            sf.write(path, audio, RATE, format="FLAC", subtype="PCM_16")
            self.store.update_sound(sound_id, path=self._stored_path(path))
        except Exception:
            path.unlink(missing_ok=True)
            self.store.delete_sound(sound_id)
            raise
        log.info("Sound %s added to %s (%.1fs, from %s)", name, guild_id, len(audio) / RATE, filename or "upload")
        return self.store.get_sound(sound_id)

    def gain(self, row: dict) -> float:
        """The gain (dB) a sound plays with: its own, capped by the server's sounds.max_volume."""
        return effective_db(row["gain_db"], self.store.get_setting(row["guild_id"], "sounds.max_volume"))

    def pcm(self, sound_id: int) -> bytes:
        """The sound as Discord PCM, at its volume (see gain). KeyError if it's gone."""
        row = self.store.get_sound(sound_id)
        if row is None:
            raise KeyError(f"no sound {sound_id}")
        key = (sound_id, self.gain(row))
        with self._lock:
            if key in self._pcm:
                self._pcm.move_to_end(key)
                return self._pcm[key]
        audio, rate = sf.read(self._disk_path(row["path"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != RATE:
            import soxr

            audio = soxr.resample(audio, rate, RATE).astype(np.float32)
        if key[1]:
            gain = 10 ** (key[1] / 20)
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak * gain > 0.99:  # a boost stops where it would clip
                gain = 0.99 / peak
            audio = audio * gain
        pcm = to_discord_pcm(audio)
        with self._lock:
            self._pcm[key] = pcm
            while len(self._pcm) > PCM_CACHE_SIZE:
                self._pcm.popitem(last=False)
        return pcm

    def delete(self, sound_id: int) -> None:
        """The file and the row (the store drops the steps that played it)."""
        row = self.store.get_sound(sound_id)
        if row is None:
            return
        with self._lock:
            for key in [k for k in self._pcm if k[0] == sound_id]:
                del self._pcm[key]
        if row["path"]:
            self._disk_path(row["path"]).unlink(missing_ok=True)
        self.store.delete_sound(sound_id)
