"""The one place speech gets generated: a single worker thread (one GPU),
fed by a priority queue.

Live replies go first, then dashboard previews, then background work
(building voices, pre-generating lines). Without priorities, "pre-generate
everything" or a voice build would hold every gag up behind it.

The queue is visible (snapshot()) so the dashboard can show what's waiting.
"""
import concurrent.futures
import itertools
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

PRIORITY = {"reply": 0, "preview": 1, "build": 2, "pregen": 3}


@dataclass(order=True)
class _Job:
    priority: int
    seq: int  # FIFO within a priority
    kind: str = field(compare=False)
    text: str = field(compare=False)
    voice: str = field(compare=False)
    fn: Callable[..., Any] = field(compare=False)
    args: tuple = field(compare=False)
    future: concurrent.futures.Future = field(compare=False)


class TTSQueue:
    def __init__(self) -> None:
        self._queue: queue.PriorityQueue[_Job] = queue.PriorityQueue()
        self._seq = itertools.count()
        self._pending: list[_Job] = []  # what's waiting, for snapshot()
        self._running: _Job | None = None
        self._lock = threading.Lock()
        threading.Thread(target=self._work, name="tts", daemon=True).start()

    def submit(self, kind: str, fn: Callable[..., Any], *args, text: str = "", voice: str = "") -> concurrent.futures.Future:
        """Run fn(*args) on the TTS thread. kind: a PRIORITY key. Await it
        with asyncio.wrap_future()."""
        job = _Job(PRIORITY[kind], next(self._seq), kind, text, voice, fn, args, concurrent.futures.Future())
        with self._lock:
            self._pending.append(job)
        self._queue.put(job)
        return job.future

    def snapshot(self) -> dict:
        with self._lock:
            pending = sorted(self._pending)
            running = self._running
        describe = lambda j: {"kind": j.kind, "text": j.text, "voice": j.voice}
        return {
            "pending": len(pending),
            "running": describe(running) if running else None,
            "items": [describe(j) for j in pending[:50]],
        }

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            with self._lock:
                self._pending.remove(job)
                self._running = job
            try:
                if job.future.set_running_or_notify_cancel():
                    try:
                        job.future.set_result(job.fn(*job.args))
                    except BaseException as e:
                        job.future.set_exception(e)
            finally:
                with self._lock:
                    self._running = None
