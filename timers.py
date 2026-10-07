"""Voice timers, in English and Spanish.

Durations (parse_timer):

    "remind me in 10 minutes to take the pizza out"   -> (600, "10 minutes", "take the pizza out")
    "set a timer for an hour and a half"              -> (5400, "an hour and a half", "")
    "remind me in 1 hour 30 minutes"                  -> (5400, "1 hour 30 minutes", "")
    "recuérdame en diez minutos sacar la pizza"       -> (600, "diez minutos", "sacar la pizza")
    "avísame en hora y media"                         -> (5400, "hora y media", "")

A duration is a number and a unit, possibly followed by more of them
("1 hour and 30 minutes") or a half ("and a half", "y media"), or a set
phrase ("half an hour", "media hora"). What to remind about is whatever
comes after it, minus little words like "to" / "que".

Clock times (parse_clock), for alarms:

    "remind me at 5 pm to call mom"                   -> (seconds until then, "5 PM", "call mom")
    "avísame a las 5 y media de la tarde"             -> (..., "las 5 y media de la tarde", "")

A time without am/pm is the next time the clock shows it ("at 5" at 3 PM
is 5 PM). parse_duration reads the /timer command's duration ("10" minutes,
"1h30m", "90s", or words). describe(), spoken_left() and clock_text() say
times back.
"""
import re
from datetime import datetime, timedelta

from transcriber import normalize

MAX_SECONDS = 12 * 3600       # a timer
MAX_ALARM_SECONDS = 24 * 3600  # an alarm at a clock time: the next one is always within a day

LANGUAGES = {
    "en": {
        "numbers": {
            "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
            "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
            "couple": 2,
        },
        "units": {
            "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
            "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
            "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
        },
        # "twenty five", "forty-five" (the hyphen splits it into two words)
        "tens_joiner": None,
        "phrases": {("half", "an", "hour"): 1800, ("half", "hour"): 1800, ("quarter", "of", "an", "hour"): 900,
                    ("half", "a", "minute"): 30},
        # Between the parts of one duration: "1 hour AND 30 minutes", "an hour AND a half".
        "and": {"and"},
        "half": [("a", "half")],
        "quarter": [],
        # Little words between the duration and the message: "in 10 minutes TO call mom".
        "filler": {"to", "that", "about", "for", "and", "please", "of", "so"},
        "names": {1: ("second", "seconds"), 60: ("minute", "minutes"), 3600: ("hour", "hours")},
        # Clock times: "at 5", "at 5:30 pm", "at five thirty".
        "at": [("at",)],
        "am": [("am",), ("a", "m"), ("in", "the", "morning")],
        "pm": [("pm",), ("p", "m"), ("in", "the", "afternoon"), ("in", "the", "evening"), ("at", "night"),
               ("tonight",)],
    },
    "es": {
        "numbers": {
            "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
            "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
            "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
            "veintiuno": 21, "veintiun": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
            "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
            "treinta": 30, "cuarenta": 40, "cuarentaicinco": 45, "cincuenta": 50, "sesenta": 60, "setenta": 70,
            "ochenta": 80, "noventa": 90,
        },
        "units": {
            "segundo": 1, "segundos": 1, "seg": 1, "s": 1,
            "minuto": 60, "minutos": 60, "min": 60, "m": 60,
            "hora": 3600, "horas": 3600, "h": 3600,
        },
        "tens_joiner": "y",  # "cuarenta y cinco minutos"
        "phrases": {("media", "hora"): 1800, ("un", "cuarto", "de", "hora"): 900, ("hora", "y", "media"): 5400,
                    ("medio", "minuto"): 30},
        "and": {"y"},
        "half": [("media",), ("medio",)],
        "quarter": [("cuarto",)],  # "dos horas y cuarto"
        "filler": {"que", "para", "a", "de", "y", "e", "pa", "por", "favor"},
        "names": {1: ("segundo", "segundos"), 60: ("minuto", "minutos"), 3600: ("hora", "horas")},
        "at": [("a", "las"), ("a", "la")],
        "am": [("am",), ("a", "m"), ("de", "la", "manana"), ("de", "la", "madrugada")],
        "pm": [("pm",), ("p", "m"), ("de", "la", "tarde"), ("de", "la", "noche"), ("del", "mediodia")],
    },
}
DEFAULT_LANGUAGE = "en"
_DECIMAL = re.compile(r"\d+(?:[.,]\d+)?")
_CLOCK = re.compile(r"(\d{1,2}):(\d{2})")


def _tokens(text: str) -> tuple[list[str], list[str]]:
    """The words as said, and normalized. "5:30", "1.5" and digits stay whole;
    letters and digits glued together are split ("10min" -> "10", "min")."""
    words = re.findall(r"\d{1,2}:\d{2}|\d+(?:[.,]\d+)?|[^\W\d_]+", text.lower())
    return words, [w if w[0].isdigit() else normalize(w) for w in words]


def _number(word: str, spec: dict) -> float | None:
    if _DECIMAL.fullmatch(word):
        return float(word.replace(",", "."))
    return spec["numbers"].get(word)


def _number_at(norm: list[str], i: int, spec: dict) -> tuple[float, int] | None:
    """A number starting at norm[i] -> (value, index after it): "45",
    "forty five", "cuarenta y cinco", "a couple (of)"."""
    if i >= len(norm):
        return None
    if norm[i] == "a" and i + 1 < len(norm) and norm[i + 1] == "couple":
        i += 1
    n = _number(norm[i], spec)
    if n is None:
        return None
    end = i + 1
    if norm[i] == "couple" and end < len(norm) and norm[end] == "of":
        end += 1
    if n % 10 == 0 and n >= 20 and n == int(n):
        joiner = spec["tens_joiner"]
        if joiner and end + 1 < len(norm) and norm[end] == joiner:
            unit = _number(norm[end + 1], spec)
            if unit is not None and 0 < unit < 10 and unit == int(unit):
                return n + unit, end + 2  # "cuarenta y cinco"
        elif not joiner and end < len(norm):
            unit = _number(norm[end], spec)
            if unit is not None and 0 < unit < 10 and unit == int(unit) and norm[end] not in ("a", "an"):
                return n + unit, end + 1  # "twenty five"
    return n, end


def _starts(norm: list[str], i: int, options) -> int:
    """Length of the first of `options` (word tuples) at norm[i], else 0."""
    for words in options:
        if tuple(norm[i:i + len(words)]) == tuple(words):
            return len(words)
    return 0


def _amount(norm: list[str], i: int, spec: dict) -> tuple[float, int, int | None] | None:
    """A duration starting exactly at norm[i]: (seconds, end, smallest unit
    used, None after a set phrase)."""
    for phrase, seconds in spec["phrases"].items():
        if tuple(norm[i:i + len(phrase)]) == phrase:
            return seconds, i + len(phrase), None
    number = _number_at(norm, i, spec)
    if number and number[1] < len(norm) and norm[number[1]] in spec["units"]:
        unit = spec["units"][norm[number[1]]]
        return number[0] * unit, number[1] + 1, unit
    return None


def _extend(norm: list[str], seconds: float, end: int, unit: int | None, spec: dict) -> tuple[float, int]:
    """More of the same duration: "... and 30 minutes", "... and a half", "... y cuarto"."""
    while unit:
        j = end + (1 if end < len(norm) and norm[end] in spec["and"] else 0)
        if size := _starts(norm, j, spec["half"]):
            return seconds + unit / 2, j + size
        if unit == 3600 and (size := _starts(norm, j, spec["quarter"])):
            return seconds + 900, j + size
        more = _amount(norm, j, spec)
        if not more or not more[2] or more[2] >= unit:
            break
        seconds, end, unit = seconds + more[0], more[1], more[2]
    return seconds, end


def _message(words: list[str], end: int, spec: dict) -> str:
    rest = words[end:]
    while rest and normalize(rest[0]) in spec["filler"]:
        rest = rest[1:]
    return " ".join(rest)


def _parse(words: list[str], norm: list[str], spec: dict) -> tuple[int, str, str] | None:
    for i in range(len(norm)):
        found = _amount(norm, i, spec)
        if not found:
            continue
        seconds, end = _extend(norm, found[0], found[1], found[2], spec)
        if not 0 < seconds <= MAX_SECONDS:
            return None
        return int(round(seconds)), " ".join(words[i:end]), _message(words, end, spec)
    return None


def _languages(lang: str | None) -> list[str]:
    order = [lang] if lang in LANGUAGES else []
    return order + [code for code in LANGUAGES if code not in order]


def parse_timer(text: str, lang: str | None = None) -> tuple[int, str, str] | None:
    """-> (seconds, the duration as said, the message or ""), or None.
    Tries `lang` first, then the other languages (people mix them)."""
    words, norm = _tokens(text)
    for code in _languages(lang):
        parsed = _parse(words, norm, LANGUAGES[code])
        if parsed:
            return parsed
    return None


# ───────────────────────────── clock times ─────────────────────────────

def _clock_at(norm: list[str], i: int, spec: dict) -> tuple[int, int, str | None, int] | None:
    """A clock time right after "at" / "a las": (hour, minute, "am"/"pm"/None, end)."""
    if i >= len(norm):
        return None
    if m := _CLOCK.fullmatch(norm[i]):
        hour, minute, end = int(m[1]), int(m[2]), i + 1
    else:
        number = _number_at(norm, i, spec)
        if not number or number[0] != int(number[0]):
            return None
        hour, minute, end = int(number[0]), 0, number[1]
        if spec["tens_joiner"]:  # "las 5 y media", "las 5 y cuarto", "las 5 y 20", "las 5 menos cuarto"
            if end < len(norm) and norm[end] == "y":
                if _starts(norm, end + 1, spec["half"]):
                    minute, end = 30, end + 2
                elif _starts(norm, end + 1, spec["quarter"]):
                    minute, end = 15, end + 2
                elif (more := _number_at(norm, end + 1, spec)) and 0 < more[0] < 60 and more[0] == int(more[0]):
                    minute, end = int(more[0]), more[1]
            elif norm[end:end + 2] == ["menos", "cuarto"]:
                hour, minute, end = hour - 1, 45, end + 2
        else:  # "five thirty", "5 15", "five o'clock"
            if norm[end:end + 2] == ["o", "clock"]:
                end += 2
            elif (more := _number_at(norm, end, spec)) and 0 <= more[0] < 60 and more[0] == int(more[0]) \
                    and norm[end] not in ("a", "an") and (more[1] >= len(norm) or norm[more[1]] not in spec["units"]):
                minute, end = int(more[0]), more[1]
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    meridiem = None
    for name in ("am", "pm"):
        if size := _starts(norm, end, spec[name]):
            meridiem, end = name, end + size
            break
    if meridiem and hour > 12:
        return None  # "at 17 pm"
    return hour, minute, meridiem, end


def next_time(hour: int, minute: int, meridiem: str | None, now: datetime) -> datetime:
    """When the clock next shows this time (after `now`, in now's time zone)."""
    if meridiem == "pm":
        hours = [hour if hour >= 12 else hour + 12]
    elif meridiem == "am":
        hours = [0 if hour == 12 else hour]
    elif hour == 0 or hour > 12:
        hours = [hour]
    else:
        hours = [hour % 12, hour % 12 + 12]  # no am/pm: whichever comes first
    candidates = []
    for h in hours:
        at = now.replace(hour=h, minute=minute, second=0, microsecond=0)
        candidates.append(at if at > now else at + timedelta(days=1))
    return min(candidates)


def parse_clock(text: str, lang: str | None, now: datetime) -> tuple[int, str, str] | None:
    """An alarm at a clock time: -> (seconds from `now`, the time as the bot
    says it (clock_text), the message or ""), or None."""
    words, norm = _tokens(text)
    for code in _languages(lang):
        spec = LANGUAGES[code]
        for i in range(len(norm)):
            size = _starts(norm, i, spec["at"])
            if not size or not (found := _clock_at(norm, i + size, spec)):
                continue
            hour, minute, meridiem, end = found
            at = next_time(hour, minute, meridiem, now)
            seconds = (at - now).total_seconds()
            if not 0 < seconds <= MAX_ALARM_SECONDS:
                return None
            return int(round(seconds)), clock_text(at, code), _message(words, end, spec)
    return None


def clock_text(at: datetime, lang: str | None = None) -> str:
    """A time of day as the bot says it: "5 PM", "5:30 PM" / "las 5 de la
    tarde", "las 5 y media de la tarde"."""
    h12 = at.hour % 12 or 12
    if (lang or DEFAULT_LANGUAGE) != "es":
        return f"{h12}{'' if at.minute == 0 else f':{at.minute:02d}'} {'AM' if at.hour < 12 else 'PM'}"
    minute = {0: "", 15: " y cuarto", 30: " y media"}.get(at.minute, f" y {at.minute}")
    if at.hour < 6 or at.hour >= 20:
        period = "de la noche" if at.hour >= 20 or at.hour == 0 else "de la madrugada"
    elif at.hour < 12:
        period = "de la mañana"
    elif at.hour < 13:
        period = "del mediodía"
    else:
        period = "de la tarde"
    return f"{'la' if h12 == 1 else 'las'} {h12}{minute} {period}"


# ───────────────────────────── /timer ─────────────────────────────

_SHORT = re.compile(r"(?:(\d+(?:[.,]\d+)?)\s*h(?:ours?|rs?|oras?)?)?\s*"
                    r"(?:(\d+(?:[.,]\d+)?)\s*m(?:in(?:ute)?s?|inutos?)?)?\s*"
                    r"(?:(\d+(?:[.,]\d+)?)\s*s(?:ec(?:ond)?s?|egundos?)?)?")


def parse_duration(text: str, lang: str | None = None) -> int | None:
    """The /timer command's duration in seconds: a plain number is minutes
    ("10", "2.5"); also "1h30m", "90s", "1h 15m", or words ("half an hour")."""
    text = (text or "").strip().lower()
    if not text:
        return None
    if _DECIMAL.fullmatch(text):
        seconds = float(text.replace(",", ".")) * 60
    elif (m := _SHORT.fullmatch(text)) and any(m.groups()):
        seconds = sum(float(v.replace(",", ".")) * unit for v, unit in zip(m.groups(), (3600, 60, 1)) if v)
    else:
        parsed = parse_timer(text, lang)
        seconds = parsed[0] if parsed else 0
    seconds = int(round(seconds))
    return seconds if 0 < seconds <= MAX_SECONDS else None


# ───────────────────────────── saying times ─────────────────────────────

def _names(lang: str | None) -> dict:
    return LANGUAGES.get(lang or DEFAULT_LANGUAGE, LANGUAGES[DEFAULT_LANGUAGE])["names"]


def describe(seconds: float, lang: str | None = None) -> str:
    """600 -> "10 minutes" / "10 minutos"; 90 -> "1.5 minutes"; 3600 -> "1 hour"."""
    names = _names(lang)
    seconds = float(seconds)
    if seconds >= 3600 and seconds % 900 == 0:
        unit = 3600
    elif seconds >= 60 and seconds % 30 == 0:
        unit = 60
    else:
        unit = 1
    amount = seconds / unit
    singular, plural = names[unit]
    return f"{amount:g} {singular if amount == 1 else plural}"


def spoken_left(seconds: float, lang: str | None = None) -> str:
    """Time left, rounded the way people say it: "45 seconds", "4 minutes",
    "1 hour and 20 minutes" / "1 hora y 20 minutos"."""
    names = _names(lang)
    joiner = " y " if lang == "es" else " and "
    word = lambda n, unit: f"{n} {names[unit][0] if n == 1 else names[unit][1]}"
    seconds = max(1, int(round(seconds)))
    if seconds < 60:
        return word(seconds, 1)
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return word(minutes, 60)
    hours, minutes = divmod(minutes, 60)
    return word(hours, 3600) + (joiner + word(minutes, 60) if minutes else "")
