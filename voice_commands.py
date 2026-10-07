"""Voice commands: "Hey Heckler, leave".

A command is the wake word followed by a request, in the same utterance or
in the speaker's next one ("Hey Heckler..." pause "...leave"). What a request
does is up to the server's command reactions (reactions.py); this module
only finds the wake word and splits off the request.

The wake word is matched loosely because speech-to-text spells a made-up
name however it likes: split in two ("Heck ler"), slightly off ("Hackler"),
glued to the word before it ("Heyheckler"), or spread over several words.

The wake words are the bot's name (BOT_NAME) plus WAKE_WORDS from .env
(other names people call it, or spellings speech-to-text keeps writing).
Each needs at least 5 letters to be matched reliably.
"""
import os
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from transcriber import normalize

DEFAULT_NAME = "Heckler"
WAKE_SIMILARITY = 0.75
# How long after a bare "Hey Heckler" the same speaker's next utterance
# counts as the request.
ARM_SECONDS = 8.0


@dataclass
class VoiceCommand:
    # "wake" for a bare wake word (waiting for the request), else "request".
    action: str
    request: str        # what was said after the wake word, normalized


def bot_name() -> str:
    """The bot's name, as people say it. Read when called, not at import:
    bot.py loads .env after its imports."""
    return os.getenv("BOT_NAME", "").strip() or DEFAULT_NAME


def wake_words() -> tuple[str, ...]:
    """Normalized, without spaces: "Heck Ler" -> "heckler"."""
    names = [bot_name(), *os.getenv("WAKE_WORDS", "").split(",")]
    return tuple(dict.fromkeys(w for w in (normalize(n).replace(" ", "") for n in names) if w))


def _sounds_like(text: str, wake_word: str) -> bool:
    # Compare the end of `text` too, so a name glued to the word before it
    # ("heyheckler") still counts.
    for n in range(len(wake_word) - 2, len(wake_word) + 3):
        tail = text[-n:]
        if len(tail) >= 5 and SequenceMatcher(None, tail, wake_word).ratio() >= WAKE_SIMILARITY:
            return True
    return False


def find_wake_word(words: list[str], names: tuple[str, ...]) -> int | None:
    """Index of the first word after a wake word, or None if none was said."""
    for i in range(len(words)):
        # One word, or up to three joined: "heck ler", "hack la".
        for joined in range(1, 4):
            if i + joined > len(words):
                break
            said = "".join(words[i : i + joined])
            if any(_sounds_like(said, name) for name in names):
                return i + joined
    return None


class CommandListener:
    def __init__(self, names: tuple[str, ...] | None = None):
        self.names = names or wake_words()
        self._armed: dict[int, float] = {}  # user_id -> deadline

    def reset(self) -> None:
        self._armed.clear()

    def feed(self, user_id: int, text: str) -> VoiceCommand | None:
        words = normalize(text).split()

        start = find_wake_word(words, self.names)
        if start is None:
            deadline = self._armed.pop(user_id, None)
            if deadline is None or time.monotonic() > deadline:
                return None
            start = 0  # follow-up to a bare wake word: the whole utterance is the request

        request = " ".join(words[start:])
        if not request:
            self._armed[user_id] = time.monotonic() + ARM_SECONDS
            return VoiceCommand("wake", "")

        self._armed.pop(user_id, None)
        return VoiceCommand("request", request)
