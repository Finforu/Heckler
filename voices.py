"""Per-speaker voice references for !userclone, built from what people say.

Every transcribed utterance is a free voice sample with its exact text, which
is what OmniVoice needs. Each speaker keeps their most recent ~10 s, saved
under voices/ so the voice survives a restart. Only collected while
!userclone is on.
"""
import json
import logging
import threading
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # OmniVoice's native rate; references are stored at it
TARGET_SECONDS = 10.0
MIN_CLIP_SECONDS = 1.0
MIN_CLIP_WORDS = 2
GAP = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)


def discord_pcm_to_mono(pcm: bytes) -> np.ndarray:
    """48 kHz stereo s16 -> 24 kHz mono float32 (averaging adjacent samples)."""
    stereo = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32768
    mono = stereo.mean(axis=1)
    return mono[: len(mono) // 2 * 2].reshape(-1, 2).mean(axis=1)


class VoiceBank:
    def __init__(self, directory: Path):
        self.directory = directory
        self._clips: dict[int, deque[tuple[np.ndarray, str]]] = {}
        self._lock = threading.Lock()

    def add(self, user_id: int, pcm: bytes, text: str) -> None:
        audio = discord_pcm_to_mono(pcm)
        if len(audio) < MIN_CLIP_SECONDS * SAMPLE_RATE or len(text.split()) < MIN_CLIP_WORDS:
            return
        with self._lock:
            clips = self._clips.get(user_id)
            if clips is None:
                clips = self._clips[user_id] = self._load(user_id)
            clips.append((audio, text))
            # Drop the oldest clips while the rest still reach the target length.
            while len(clips) > 1 and sum(len(a) for a, _ in list(clips)[1:]) >= TARGET_SECONDS * SAMPLE_RATE:
                clips.popleft()
            snapshot = list(clips)
        self._save(user_id, snapshot)

    def reference(self, user_id: int, max_seconds: float | None = None) -> tuple[np.ndarray, str] | None:
        """(audio at SAMPLE_RATE, its transcript), or None if we've never heard
        them. With max_seconds, only their newest clips that fit; clips longer
        than that on their own are skipped (trimmed only if nothing else fits)."""
        with self._lock:
            clips = self._clips.get(user_id)
            if clips is None:
                clips = self._clips[user_id] = self._load(user_id)
            if not clips:
                return None
            chosen = list(clips)
            if max_seconds is not None:
                limit = int(max_seconds * SAMPLE_RATE)
                chosen, total = [], 0
                for audio, text in reversed(clips):
                    if len(audio) > limit:
                        continue
                    if total + len(audio) > limit:
                        break
                    chosen.insert(0, (audio, text))
                    total += len(audio)
                if not chosen:
                    audio, text = clips[-1]
                    chosen = [(audio[-limit:], text)]
            audio = np.concatenate([part for a, _ in chosen for part in (a, GAP)][:-1])
            text = " ".join(t for _, t in chosen)
        return audio, text

    def _paths(self, user_id: int) -> tuple[Path, Path]:
        return self.directory / f"{user_id}.wav", self.directory / f"{user_id}.json"

    def _save(self, user_id: int, clips: list[tuple[np.ndarray, str]]) -> None:
        wav_path, meta_path = self._paths(user_id)
        try:
            self.directory.mkdir(exist_ok=True)
            sf.write(wav_path, np.concatenate([a for a, _ in clips]), SAMPLE_RATE, subtype="PCM_16")
            meta_path.write_text(json.dumps([[len(a), t] for a, t in clips], ensure_ascii=False))
        except OSError as e:
            log.warning("Couldn't save voice for %s: %s", user_id, e)

    def _load(self, user_id: int) -> deque[tuple[np.ndarray, str]]:
        wav_path, meta_path = self._paths(user_id)
        clips: deque[tuple[np.ndarray, str]] = deque()
        if not (wav_path.exists() and meta_path.exists()):
            return clips
        try:
            audio, _ = sf.read(wav_path, dtype="float32")
            offset = 0
            for length, text in json.loads(meta_path.read_text()):
                clips.append((audio[offset : offset + length], text))
                offset += length
        except (OSError, ValueError) as e:
            log.warning("Couldn't load saved voice for %s: %s", user_id, e)
            clips.clear()
        return clips
