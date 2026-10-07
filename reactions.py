"""The reaction engine: turns a server's reaction rows (see store.py) into
matchers, and answers "what does the bot do now?" for something said, a
voice command, a slash command or an event (someone joined, a timer rang...).

A reaction row holds:
  triggers  any one of them firing is enough:
              {"type": "phrase", "phrases": ["buenas noches", "buena noche"]}
              {"type": "swap", "word": "café", "to": "chocolate", "also": ["cafés"],
               "connectors": ["de", "e", "y"]}
              {"type": "command", "phrases": ["vete", "largate"]}   (after the wake word)
              {"type": "slash", "name": "bruh"}
              {"type": "event", "event": "hello", "user_id": 123}   (user_id optional)
  options   alternatives, one picked at random (never the same twice in a
            row); each is a list of steps played in order:
              {"type": "say", "text": "No {name}, {subject} {connector} {to}.", "voice_id": null}
              {"type": "sound", "sound_id": 4}
              {"type": "builtin", "action": "leave"}   (leave stop timer timer_cancel
                  timer_list time coin dice pick repeat)

  phrase  the phrase as whole words anywhere in the sentence, ignoring
          capitals, accents and punctuation. Any language.
  swap    a word game from Spanish, "<something> de/e <word>" -> the word swapped:
              "la taza de café" -> "No, la taza de chocolate."
          The <something> (the subject) runs back to the nearest el/la/un/
          mi/ese... (at most 4 words). A spoken "e" usually comes out of
          speech-to-text as "y", so "y" counts and is said back as "e". Said
          fast, the "e" gets swallowed ("la taza café"); that only counts
          right after an article, so "tengo café" doesn't fire. A word that's
          close enough ("cafés" for "café") counts too.
          The connectors default to Spanish's de/e/y; give an English swap
          connectors ["of"] ("the cup of coffee" -> "No, the cup of tea.").
          The swallowed connector only applies to the Spanish ones.

Templates: {name} (who said it), {subject}, {connector}, {to} (swaps),
{said} and {message} (timers), {result} (what a helper came up with: the
time, a coin, dice, the person picked, the time left on a timer). An option that needs a value that is missing
or empty is skipped, and a reaction with no usable option doesn't fire, so
the next one gets its chance: that's how "{name}, {message}." gives way to
"{name}, your {said} are up." when a timer has no message.
"""
import random
import re
import threading
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable

from transcriber import normalize

# Where a swap's subject starts ("LA taza de café", "THE cup of coffee"):
# Spanish and English together, so a swap works whatever the server's language.
DETERMINERS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "al", 'de', "y",
    "mi", "mis", "tu", "tus", "su", "sus",
    "ese", "esa", "esos", "esas", "este", "esta", "estos", "estas", "aquel", "aquella",
} | {
    "the", "a", "an", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "some",
}
# The Spanish connectors: only with these is a swallowed one ("la taza café") understood.
SPANISH_CONNECTORS = {"de", "e", "y"}
MAX_SUBJECT_WORDS = 4
# A swap word said fast or misheard ("cafes" for "cafe") still counts if it's
# this close to one of the reaction's words.
WORD_SIMILARITY = 0.8
DEFAULT_CONNECTORS = ("de", "e", "y")
DEFAULT_SWAP_REPLY = "No, {subject} {connector} {to}."

TRIGGER_TYPES = ("phrase", "swap", "command", "slash", "event")
EVENTS = ("wake", "ack", "leave", "unknown", "hello", "bye", "arrival", "timer_set", "timer_ring",
          "alarm_set", "timer_cancelled", "timer_left", "timer_none", "time_now", "coin_result", "dice_result",
          "pick_result", "nothing_to_repeat", "llm_unavailable")
STEP_TYPES = ("say", "sound", "builtin")
# "play" (the old test sound) is gone from the bot; still accepted so older rows stay valid.
BUILTINS = ("leave", "stop", "play", "timer", "timer_cancel", "timer_list", "time", "coin", "dice", "pick",
            "repeat")
PLACEHOLDERS = ("name", "subject", "connector", "to", "said", "message", "result")
SPEAKER = "@speaker"  # voice_id: the triggering person's own voice (if they consented)

_PLACEHOLDER = re.compile(r"\{(" + "|".join(PLACEHOLDERS) + r")\}")


# ───────────────────────────── helpers ─────────────────────────────

def fill(template: str, **values: str) -> str:
    """Replace {key} placeholders; any other braces are left alone."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER.findall(text))


def speakable_name(display_name: str) -> str:
    """A display name without the emoji and symbols a voice can't say."""
    return " ".join(re.findall(r"[^\W_]+", display_name)) or display_name


def _words(text: str) -> list[str]:
    # Letters only (accents kept for the voice); "e'" and "pa'" lose the apostrophe.
    return re.findall(r"[^\W\d_]+", text.lower())


def _subject(before: list[str]) -> str | None:
    window = before[-MAX_SUBJECT_WORDS:]
    for i in range(len(window) - 1, -1, -1):
        if normalize(window[i]) in DETERMINERS:
            return " ".join(window[i:]) if i < len(window) - 1 else None
    return window[-1] if window else None


def _user_id(value) -> int | None:
    """User ids may arrive as strings (JSON from a browser can't hold 64-bit ints)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _people(by_users) -> tuple[int | str, ...] | None:
    """by_users -> user ids and normalized names; None = anyone."""
    if by_users is None:
        return None
    people = [by_users] if isinstance(by_users, (int, str)) else list(by_users)
    return tuple(_user_id(p) if _user_id(p) is not None else normalize(str(p)) for p in people)


def validate(triggers, options) -> None:
    """ValueError if a reaction's triggers or options are malformed. The
    store calls this before every write, so the engine can trust the rows."""
    if not isinstance(triggers, list) or not triggers:
        raise ValueError("triggers must be a non-empty list")
    for t in triggers:
        kind = t.get("type") if isinstance(t, dict) else None
        if kind not in TRIGGER_TYPES:
            raise ValueError(f"trigger type must be one of {TRIGGER_TYPES}: {t!r}")
        if kind in ("phrase", "command"):
            phrases = t.get("phrases")
            if not isinstance(phrases, list) or not phrases or not all(
                    isinstance(p, str) and normalize(p) for p in phrases):
                raise ValueError(f"a {kind} trigger needs a list of phrases: {t!r}")
        elif kind == "swap":
            if not normalize(str(t.get("word", ""))) or not isinstance(t.get("to"), str):
                raise ValueError(f"a swap trigger needs word and to: {t!r}")
            for key in ("also", "connectors"):
                if key in t and not (isinstance(t[key], list) and all(isinstance(w, str) for w in t[key])):
                    raise ValueError(f"swap {key} must be a list of words: {t!r}")
        elif kind == "slash":
            if not isinstance(t.get("name"), str) or not t["name"].strip("/ "):
                raise ValueError(f"a slash trigger needs a name: {t!r}")
        elif kind == "event":
            if t.get("event") not in EVENTS:
                raise ValueError(f"event must be one of {EVENTS}: {t!r}")
            if t.get("user_id") is not None and _user_id(t["user_id"]) is None:
                raise ValueError(f"event user_id must be a user id: {t!r}")
    if not isinstance(options, list) or not options:
        raise ValueError("options must be a non-empty list of step lists")
    for option in options:
        if not isinstance(option, list) or not option:
            raise ValueError(f"each option must be a non-empty list of steps: {option!r}")
        for step in option:
            kind = step.get("type") if isinstance(step, dict) else None
            if kind == "say" and isinstance(step.get("text"), str) and step["text"].strip():
                continue
            if kind == "sound" and isinstance(step.get("sound_id"), int):
                continue
            if kind == "builtin" and step.get("action") in BUILTINS:
                continue
            raise ValueError(f"bad step (say needs text, sound a sound_id, builtin one of {BUILTINS}): {step!r}")


def describe(reaction: dict) -> str:
    """One line for listings: 'swap "coffee" -> "tea"', 'phrase [...] -> [...]'."""
    parts = []
    for t in reaction["triggers"]:
        if t["type"] == "swap":
            parts.append(f'swap "{t["word"]}" -> "{t["to"]}"')
        elif t["type"] in ("phrase", "command"):
            parts.append(f'{t["type"]} {t["phrases"]}')
        elif t["type"] == "slash":
            parts.append(f'/{t["name"].strip("/ ")}')
        else:
            parts.append(f'event {t["event"]}' + (f' ({t["user_id"]})' if t.get("user_id") else ""))
    says = [" + ".join(s["text"] if s["type"] == "say" else f'[{s["type"]}]' for s in o) for o in reaction["options"]]
    text = f"{' | '.join(parts)} -> {says}"
    if reaction.get("by_users"):
        text += f" (only {', '.join(map(str, reaction['by_users']))})"
    return text


# ───────────────────────────── matchers ─────────────────────────────
# Each takes (normalized text, words with accents, speaker's name) and
# returns the template values when it fires, or None.

def _phrase_matcher(trigger: dict) -> Callable[[str, list[str], str], dict | None]:
    patterns = [re.compile(rf"\b{re.escape(normalize(p))}\b") for p in trigger["phrases"]]

    def match(norm: str, words: list[str], name: str) -> dict | None:
        return {"name": name} if any(p.search(norm) for p in patterns) else None

    return match


def _swap_matcher(trigger: dict) -> Callable[[str, list[str], str], dict | None]:
    targets = {normalize(w) for w in (trigger["word"], *trigger.get("also", ()))}
    joins = {normalize(c) for c in trigger.get("connectors", DEFAULT_CONNECTORS)}
    to = trigger["to"]

    def is_target(word: str) -> bool:
        word = normalize(word)
        return word in targets or (
            len(word) >= 4 and any(SequenceMatcher(None, word, t).ratio() >= WORD_SIMILARITY for t in targets)
        )

    def match(norm: str, words: list[str], name: str) -> dict | None:
        for i in range(1, len(words)):
            if not is_target(words[i]):
                continue
            if i >= 2 and normalize(words[i - 1]) in joins:  # "la taza e café"
                subject, connector = _subject(words[: i - 1]), words[i - 1]
                if normalize(connector) == "y":  # a spoken "e", as speech-to-text writes it
                    connector = "e"
            else:
                # Said fast, the "e" gets swallowed: "la taza café". Only counts
                # after an article, so "tengo café" doesn't fire. Spanish only.
                if not joins & SPANISH_CONNECTORS:
                    continue
                subject, connector = _subject(words[:i]), "e"
                if not subject or normalize(subject.split()[0]) not in DETERMINERS:
                    continue
            if subject:
                return {"name": name, "subject": subject, "connector": connector, "to": to}
        return None

    return match


# ───────────────────────────── engine ─────────────────────────────

@dataclass
class Match:
    reaction: dict
    # The chosen option: steps with templates filled in, and every say step's
    # voice_id resolved (step -> reaction -> setting voice.default). None means
    # the bot's own voice, "@speaker" the person's own.
    steps: list[dict]
    trigger: dict | None = None
    values: dict = field(default_factory=dict)
    # "cooldown" or "chance": the reaction matched but was held back (steps
    # is empty, and no other reaction got a turn). Fire only when None.
    blocked: str | None = None
    # Its effective chance (the gag intensity included): a dry run doesn't roll it.
    chance: float = 1.0

    @property
    def text(self) -> str:
        """Everything said, for logs and echo detection."""
        return " ".join(s["text"] for s in self.steps if s["type"] == "say")


@dataclass
class _Reaction:
    row: dict
    by: tuple[int | str, ...] | None
    word_triggers: list[tuple[dict, Callable]]
    is_word: bool  # gags and anything else set off by what people say: the gag settings apply

    def said_by(self, user_id: int | None, display_name: str, name: str) -> bool:
        if self.by is None:
            return True
        names = {normalize(display_name or ""), normalize(name or "")}
        return any(who == user_id if isinstance(who, int) else who in names for who in self.by)


@dataclass
class _Guild:
    reactions: list[_Reaction]
    word: list[_Reaction]
    commands: list[tuple[re.Pattern, int, _Reaction]]  # (pattern, phrase length, reaction)
    default_voice: object
    cooldown_s: float
    intensity: float


class ReactionEngine:
    """Compiled reactions per server, built on first use and dropped when
    the store says something changed. Thread-safe."""

    def __init__(self, store, *, clock: Callable[[], float] = time.monotonic):
        self.store = store
        self.clock = clock
        self._lock = threading.RLock()
        self._guilds: dict[int, _Guild] = {}
        # Survive recompiles: the last option played per reaction (no
        # repeats), and when each (guild, user, reaction) last fired.
        self._last: dict[int, list] = {}
        self._fired: dict[tuple[int, int | None, int], float] = {}
        self._dry_random = random.Random()
        store.on_change(self._changed)

    def _changed(self, table: str, guild_id: int | None) -> None:
        if table in ("reactions", "settings", "voices", "sounds"):
            self.invalidate(guild_id)

    def invalidate(self, guild_id: int | None = None) -> None:
        """Recompile this server's reactions on next use (None: every server)."""
        with self._lock:
            if guild_id is None:
                self._guilds.clear()
            else:
                self._guilds.pop(guild_id, None)

    def _compiled(self, guild_id: int) -> _Guild:
        with self._lock:
            compiled = self._guilds.get(guild_id)
            if compiled is None:
                compiled = self._guilds[guild_id] = self._compile(guild_id)
            return compiled

    def _compile(self, guild_id: int) -> _Guild:
        reactions, word, commands = [], [], []
        for row in self.store.list_reactions(guild_id):
            if not row["enabled"] or row["status"] != "approved":
                continue
            word_triggers = []
            for t in row["triggers"]:
                if t["type"] == "phrase":
                    word_triggers.append((t, _phrase_matcher(t)))
                elif t["type"] == "swap":
                    word_triggers.append((t, _swap_matcher(t)))
            r = _Reaction(row, _people(row["by_users"]), word_triggers, is_word=bool(word_triggers))
            reactions.append(r)
            if word_triggers:
                word.append(r)
            for t in row["triggers"]:
                if t["type"] == "command":
                    for phrase in t["phrases"]:
                        norm = normalize(phrase)
                        commands.append((re.compile(rf"\b{re.escape(norm)}\b"), len(norm), r))
        settings = self.store.settings(guild_id)
        return _Guild(reactions, word, commands, settings.get("voice.default"),
                      float(settings.get("gags.cooldown_s") or 0), float(settings.get("gags.intensity", 1.0)))

    # ------------------------------------------------------------ lookups
    def match(self, guild_id: int, text: str, user_id: int, display_name: str, name: str, *,
              dry_run: bool = False) -> Match | None:
        """What the bot answers to this person saying `text` (phrase and swap
        triggers), or None. The first reaction that matches wins; if it's
        cooling down or its chance says no, the Match says so in `blocked`
        and nothing else gets a turn.

        dry_run (the dashboard's test bench) changes nothing: no cooldown
        stamp, no picker memory, and the chance isn't rolled."""
        g = self._compiled(guild_id)
        norm, words = normalize(text), _words(text)
        for r in g.word:
            if not r.said_by(user_id, display_name, name):
                continue
            for trigger, matcher in r.word_triggers:
                values = matcher(norm, words, name)
                if values:
                    result = self._fire(guild_id, g, r, trigger, values, user_id, dry_run)
                    if result is not _UNUSABLE:
                        return result
                    break  # no option fits these values: let the next reaction try
        return None

    def for_event(self, guild_id: int, event: str, user_id: int | None = None, display_name: str = "",
                  name: str = "", values: dict | None = None, *, dry_run: bool = False) -> Match | None:
        """The reaction to an event: the person's own one if they have one
        (an event trigger with their user_id, or by_users naming them),
        otherwise the server's general one. values fill {said}, {message}..."""
        g = self._compiled(guild_id)
        values = {"name": name, **(values or {})}
        own, general = [], []
        for r in g.reactions:
            for t in r.row["triggers"]:
                if t["type"] != "event" or t["event"] != event:
                    continue
                trigger_user = _user_id(t.get("user_id"))
                if trigger_user is not None:
                    if trigger_user == user_id and r.said_by(user_id, display_name, name):
                        own.append((r, t))
                elif r.by is not None:
                    if r.said_by(user_id, display_name, name):
                        own.append((r, t))
                else:
                    general.append((r, t))
                break
        for r, t in own + general:
            result = self._fire(guild_id, g, r, t, values, user_id, dry_run)
            if result is not _UNUSABLE:
                return result
        return None

    def for_command(self, guild_id: int, request: str, user_id: int | None = None, display_name: str = "",
                    name: str = "", values: dict | None = None, *, dry_run: bool = False) -> Match | None:
        """The command reaction for what was said after the wake word. When
        several phrases match, the one said first wins; on a tie the longer
        phrase ("para la musica" beats "para"), then the earlier reaction."""
        g = self._compiled(guild_id)
        request = normalize(request)
        best = None  # (start, -length, reaction)
        for pattern, length, r in g.commands:
            if not r.said_by(user_id, display_name, name):
                continue
            m = pattern.search(request)
            if m and (best is None or (m.start(), -length) < best[:2]):
                best = (m.start(), -length, r)
        if best is None:
            return None
        r = best[2]
        trigger = next(t for t in r.row["triggers"] if t["type"] == "command")
        result = self._fire(guild_id, g, r, trigger, {"name": name, **(values or {})}, user_id, dry_run)
        return None if result is _UNUSABLE else result

    def for_slash(self, guild_id: int, slash: str, user_id: int | None = None, display_name: str = "",
                  name: str = "", *, dry_run: bool = False) -> Match | None:
        """The reaction for a slash trigger such as "/sound bruh" (pass "bruh")."""
        g = self._compiled(guild_id)
        wanted = normalize(slash.strip("/ "))
        for r in g.reactions:
            if not r.said_by(user_id, display_name, name):
                continue
            for t in r.row["triggers"]:
                if t["type"] == "slash" and normalize(t["name"].strip("/ ")) == wanted:
                    result = self._fire(guild_id, g, r, t, {"name": name}, user_id, dry_run)
                    return None if result is _UNUSABLE else result
        return None

    def fixed_texts(self, guild_id: int) -> set[tuple[str, object]]:
        """(text, voice_id) of every line that never changes (no
        placeholders), so the bot can generate them ahead of time. Lines in
        the speaker's own voice can't be, so they're left out."""
        g = self._compiled(guild_id)
        out = set()
        for r in g.reactions:
            for option in r.row["options"]:
                for step in option:
                    if step["type"] == "say" and not placeholders(step["text"]):
                        voice = self._voice(g, r, step)
                        if voice != SPEAKER:
                            out.add((step["text"], voice))
        return out

    # ------------------------------------------------------------ firing
    @staticmethod
    def _voice(g: _Guild, r: _Reaction, step: dict):
        for voice in (step.get("voice_id"), r.row["voice_id"], g.default_voice):
            if voice is not None:
                return voice
        return None

    @staticmethod
    def _usable(r: _Reaction, values: dict) -> list[list[dict]]:
        """The options whose placeholders all have a value."""
        return [o for o in r.row["options"]
                if all(values.get(key) for step in o if step["type"] == "say" for key in placeholders(step["text"]))]

    @staticmethod
    def _chance(g: _Guild, r: _Reaction) -> float:
        return float(r.row["chance"]) * (g.intensity if r.is_word else 1.0)

    def _cooling_down(self, guild_id: int, g: _Guild, r: _Reaction, user_id, now: float) -> bool:
        row = r.row
        cooldown = row["cooldown_s"] if row["cooldown_s"] is not None else (g.cooldown_s if r.is_word else 0)
        last = self._fired.get((guild_id, user_id, row["id"]))
        return bool(cooldown) and last is not None and now - last < cooldown

    def _fire(self, guild_id: int, g: _Guild, r: _Reaction, trigger: dict, values: dict, user_id,
              dry_run: bool = False):
        """Pick and fill an option. _UNUSABLE when no option fits the values;
        a Match with `blocked` set when chance or cooldown hold it back."""
        usable = self._usable(r, values)
        if not usable:
            return _UNUSABLE
        row = r.row
        chance = self._chance(g, r)
        if not dry_run and chance < 1 and random.random() >= chance:
            return Match(row, [], trigger, values, blocked="chance", chance=chance)
        with self._lock:
            now = self.clock()
            if self._cooling_down(guild_id, g, r, user_id, now):
                return Match(row, [], trigger, values, blocked="cooldown", chance=chance)
            # Never the same option twice in a row (by value, like the old picker).
            last = self._last.get(row["id"], [])
            choices = [o for o in usable if not last or o != last[0]] or usable
            if dry_run:
                option = self._dry_random.choice(choices)  # leaves the global random sequence alone too
            else:
                self._fired[(guild_id, user_id, row["id"])] = now
                option = random.choice(choices)
                self._last[row["id"]] = [option]
        strings = {k: str(v) for k, v in values.items() if v is not None}
        steps = []
        for step in option:
            step = dict(step)
            if step["type"] == "say":
                step["text"] = fill(step["text"], **strings)
                step["voice_id"] = self._voice(g, r, step)
            steps.append(step)
        return Match(row, steps, trigger, values, chance=chance)

    # ------------------------------------------------------------ test bench
    def explain(self, guild_id: int, text: str, user_id: int | None = None, display_name: str = "",
                name: str = "") -> list[dict]:
        """Every word reaction that matches `text`, in order, and why only the
        first one would fire. Also near misses: words a trigger *almost*
        matched (similarity from NEAR_MISS up to where it would have matched:
        WORD_SIMILARITY for swap words, exact for phrases), the seed of "add
        this spelling to the trigger?". Changes nothing.

        Entries: {reaction_id, name, kind, trigger, would_fire, reason, chance}
        reason: None | "cooldown" | "not this user" | "no usable option" | "first match wins"
        Near misses: {reaction_id, name, kind, trigger, near_miss: True, heard,
        trigger_word, similarity, would_fire: False, reason: "near miss"}"""
        g = self._compiled(guild_id)
        norm, words = normalize(text), _words(text)
        norm_words = norm.split()
        out, decided = [], False  # decided: a reaction already took this sentence (fired or was blocked)
        with self._lock:
            now = self.clock()
            for r in g.word:
                row = r.row
                entry = {"reaction_id": row["id"], "name": row["name"], "kind": row["kind"]}
                hit = None
                for trigger, matcher in r.word_triggers:
                    values = matcher(norm, words, name)
                    if values:
                        hit = (trigger, values)
                        break
                if hit is None:
                    out += [{**entry, **miss} for miss in _near_misses(r, norm_words)]
                    continue
                trigger, values = hit
                reason = None
                if not r.said_by(user_id, display_name, name):
                    reason = "not this user"
                elif decided:
                    reason = "first match wins"
                elif not self._usable(r, values):
                    reason = "no usable option"
                else:
                    decided = True
                    if self._cooling_down(guild_id, g, r, user_id, now):
                        reason = "cooldown"
                out.append({**entry, "trigger": trigger, "would_fire": reason is None, "reason": reason,
                            "chance": self._chance(g, r)})
        return out


# Below WORD_SIMILARITY but at least this close: worth suggesting as a new spelling.
NEAR_MISS = 0.6


def _near_misses(r: _Reaction, heard: list[str]) -> list[dict]:
    """Phrases (and swap words) that a run of the same number of words in
    the sentence came close to, without matching."""
    out, seen = [], set()
    for trigger, _ in r.word_triggers:
        if trigger["type"] == "phrase":
            wanted = [normalize(p) for p in trigger["phrases"]]
        else:
            wanted = [normalize(w) for w in (trigger["word"], *trigger.get("also", ()))]
        for target in wanted:
            size = len(target.split())
            for i in range(len(heard) - size + 1):
                said = " ".join(heard[i:i + size])
                if len(said) < 3 or said == target or (said, target) in seen:
                    continue
                ratio = SequenceMatcher(None, said, target).ratio()
                # Swap words of 4+ letters already match from WORD_SIMILARITY
                # up; phrases (and short words) only match exactly.
                fuzzy = trigger["type"] == "swap" and len(said) >= 4
                if NEAR_MISS <= ratio < (WORD_SIMILARITY if fuzzy else 1):
                    seen.add((said, target))
                    out.append({"trigger": trigger, "near_miss": True, "heard": said, "trigger_word": target,
                                "similarity": round(ratio, 2), "would_fire": False, "reason": "near miss"})
    return out


_UNUSABLE = object()
