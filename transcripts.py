"""Written transcripts of calls: one JSON line per sentence, a file per day
(transcripts/YYYY-MM-DD.jsonl).

Off unless a server turns them on (/transcripts on), and then only the words
of people who agreed are written (the bot checks both before calling
append()). Anyone can have their lines removed (/transcripts delete).
"""
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class Transcripts:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        """record: time (ISO), guild_id, guild, channel, user_id, user, language, duration, text."""
        day = str(record["time"])[:10]
        with self._lock:
            self.directory.mkdir(exist_ok=True)
            with open(self.directory / f"{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def delete_user(self, user_id: int, guild_id: int | None = None) -> int:
        """Remove everything this person said (in one server, or all).
        Returns how many lines went. Lines from before servers were recorded
        by id count as any server."""
        removed = 0
        with self._lock:
            for path in sorted(self.directory.glob("*.jsonl")) if self.directory.is_dir() else []:
                kept, dropped = [], 0
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            kept.append(line)
                            continue
                        same_user = row.get("user_id") == user_id
                        same_guild = guild_id is None or row.get("guild_id") in (None, guild_id)
                        if same_user and same_guild:
                            dropped += 1
                        else:
                            kept.append(line)
                if not dropped:
                    continue
                removed += dropped
                if not kept:
                    path.unlink()
                    continue
                fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(kept)
                os.replace(tmp, path)
        if removed:
            log.info("Removed %d transcript lines of user %s", removed, user_id)
        return removed
