"""Per-speaker utterance capture from a Discord voice channel.

Each speaker's 20 ms packets are classified as speech or not by loudness
relative to that speaker's own background noise (the quiet end of their last
few seconds). An utterance starts on speech and ends after ``silence_s``
without speech, whether packets stop coming (normal mics) or keep coming full
of noise (open mics, fans, TV). Long monologues are cut every ``max_s``.
"""
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
from discord.ext import voice_recv

log = logging.getLogger(__name__)

BYTES_PER_SECOND = 48000 * 2 * 2  # 48 kHz, stereo, 16-bit
FRAME_SECONDS = 0.02  # one Discord packet

# Speech = at least this much louder than the speaker's noise floor...
SPEECH_MARGIN_DB = 8.0
# ...and not near-silence in absolute terms.
MIN_SPEECH_DBFS = -55.0
# Noise floor: 10th percentile of the speaker's last 5 s of packet levels.
NOISE_WINDOW_FRAMES = 250
NOISE_PERCENTILE = 10
# Audio kept from just before speech starts, and after it ends, so word
# edges aren't clipped.
PREROLL_FRAMES = 10
TAIL_FRAMES = 15


def level_dbfs(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    rms = math.sqrt(float(np.mean(samples * samples))) if len(samples) else 0.0
    return 20 * math.log10(max(rms, 1.0) / 32768)


@dataclass
class Utterance:
    user_id: int
    user_name: str
    started_at: datetime
    pcm: bytes

    @property
    def duration(self) -> float:
        return len(self.pcm) / BYTES_PER_SECOND


@dataclass
class _Burst:
    started_at: datetime = field(default_factory=datetime.now)
    last_speech: float = field(default_factory=time.monotonic)
    chunks: list[bytes] = field(default_factory=list)
    size: int = 0
    speech_frames: int = 0
    chunks_at_last_speech: int = 0


@dataclass
class _Speaker:
    name: str
    levels: deque = field(default_factory=lambda: deque(maxlen=NOISE_WINDOW_FRAMES))
    preroll: deque = field(default_factory=lambda: deque(maxlen=PREROLL_FRAMES))
    burst: _Burst | None = None

    def is_speech(self, level: float) -> bool:
        self.levels.append(level)
        floor = float(np.percentile(self.levels, NOISE_PERCENTILE))
        return level >= MIN_SPEECH_DBFS and level >= floor + SPEECH_MARGIN_DB


class UtteranceSink(voice_recv.AudioSink):
    def __init__(
        self,
        on_utterance: Callable[[Utterance], None],
        *,
        silence_s: float = 0.8,
        min_s: float = 0.4,
        max_s: float = 15.0,
    ):
        super().__init__()
        self.on_utterance = on_utterance
        self.silence_s = silence_s
        self.min_speech_frames = int(min_s / FRAME_SECONDS)
        self.max_bytes = int(max_s * BYTES_PER_SECOND)

        self._speakers: dict[int, _Speaker] = {}
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        # write() runs on voice-recv's router thread, so pauses are detected
        # on a thread of our own rather than the event loop.
        self._watcher = threading.Thread(target=self._watch, name="utterance-watcher", daemon=True)
        self._watcher.start()

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData) -> None:
        if not data.pcm or data.packet.is_silence():
            return

        user_id = user.id if user else self.voice_client._get_id_from_ssrc(data.packet.ssrc)
        if user_id is None:
            return  # haven't learned who owns this ssrc yet

        level = level_dbfs(data.pcm)
        full = None
        with self._lock:
            speaker = self._speakers.get(user_id)
            if speaker is None:
                speaker = self._speakers[user_id] = _Speaker(user.display_name if user else str(user_id))
            speech = speaker.is_speech(level)
            burst = speaker.burst

            if burst is None:
                if not speech:
                    speaker.preroll.append(data.pcm)
                    return
                burst = speaker.burst = _Burst()
                burst.chunks.extend(speaker.preroll)
                burst.size = sum(len(c) for c in speaker.preroll)
                speaker.preroll.clear()
                log.debug("%s started speaking", speaker.name)

            burst.chunks.append(data.pcm)
            burst.size += len(data.pcm)
            if speech:
                burst.last_speech = time.monotonic()
                burst.speech_frames += 1
                burst.chunks_at_last_speech = len(burst.chunks)
            if burst.size >= self.max_bytes:
                full = (speaker.name, burst)
                speaker.burst = None

        if full:
            self._emit(user_id, *full)

    def _watch(self) -> None:
        while not self._stopped.wait(0.1):
            now = time.monotonic()
            ended = []
            with self._lock:
                for uid, speaker in self._speakers.items():
                    if speaker.burst and now - speaker.burst.last_speech >= self.silence_s:
                        ended.append((uid, speaker.name, speaker.burst))
                        speaker.burst = None
            for uid, name, burst in ended:
                self._emit(uid, name, burst)

    def _emit(self, user_id: int, name: str, burst: _Burst) -> None:
        if burst.speech_frames < self.min_speech_frames:
            return  # a cough, a click or a noise spike, not speech
        # Drop the noise after the last word (open mics keep sending it).
        chunks = burst.chunks[: burst.chunks_at_last_speech + TAIL_FRAMES]
        utterance = Utterance(user_id, name, burst.started_at, b"".join(chunks))
        try:
            self.on_utterance(utterance)
        except Exception:
            log.exception("on_utterance callback failed")

    def cleanup(self) -> None:
        self._stopped.set()
        with self._lock:
            remaining = [(uid, s.name, s.burst) for uid, s in self._speakers.items() if s.burst]
            for s in self._speakers.values():
                s.burst = None
        for uid, name, burst in remaining:
            self._emit(uid, name, burst)
