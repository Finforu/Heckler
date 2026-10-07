"""In-process event bus: the bot publishes what happens, the dashboard (and
later the stats writer) listens.

Events are plain JSON-able dicts with a "type" key:

  {"type": "feed", "id": 17, "time": "...", "guild_id": ..., "guild": "...",
   "user_id": ..., "user": "...", "text": "...", "lang": "es", "duration": 1.2,
   "matched": {"kind": "gag" | "command" | "sound" | ..., "name": "..."} | None,
   "reply": "what the bot will say" | None, "voice": "<voice name>" | None,
   "outcome": "matched" | "no match" | "ignored (echo)"}
      one per transcribed utterance, at the moment the bot decided what to do
  {"type": "feed_update", "id": 17, "outcome": "played" | "skipped" | "skipped (cooldown)" | "failed",
   "voice": "..."}   (any subset of fields, merged into the entry with that id)
      what finally happened to that utterance's reply
  {"type": "log", "time": "...", "level": "INFO", "message": "..."}

Everything runs on the bot's event loop; publish() from another thread must
go through loop.call_soon_threadsafe.
"""
import asyncio
import itertools
from collections import deque
from datetime import datetime

HISTORY = 200
QUEUE_SIZE = 500


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._history: deque[dict] = deque(maxlen=HISTORY)
        self._ids = itertools.count(1)

    def next_id(self) -> int:
        return next(self._ids)

    def publish(self, event: dict) -> dict:
        event.setdefault("time", datetime.now().isoformat(timespec="seconds"))
        self._history.append(event)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # a stuck dashboard tab misses events rather than growing forever
        return event

    def subscribe(self) -> asyncio.Queue:
        """A queue that receives every event published from now on."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def recent(self, n: int = HISTORY) -> list[dict]:
        return list(self._history)[-n:]


bus = EventBus()
