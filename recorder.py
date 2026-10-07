"""!record: keep one person's utterances, and what the bot made of them, as
test data (testdata/<session>/: one .wav per utterance + clips.jsonl).

Only the person who turned it on is recorded; everyone else in the call is
left out.
"""
import json
from datetime import datetime
from pathlib import Path

from transcriber import pcm_to_wav


class SessionRecorder:
    def __init__(self, root: Path, user_id: int, user_name: str):
        self.dir = root / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        self.user_name = user_name
        self.count = 0

    def save(self, utterance, heard: str, language: str, **outcome) -> None:
        """outcome: what the bot did with it (command, gag, echo...)."""
        self.count += 1
        name = f"{self.count:04d}.wav"
        (self.dir / name).write_bytes(pcm_to_wav(utterance.pcm))
        entry = {
            "file": name,
            "time": utterance.started_at.isoformat(timespec="milliseconds"),
            "duration": round(utterance.duration, 2),
            "heard": heard,
            "language": language,
            **outcome,
        }
        with open(self.dir / "clips.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
