"""Voice timers, in English and Spanish:

    "remind me in 10 minutes to take the pizza out"   -> (600, "10 minutes", "take the pizza out")
    "set a timer for half an hour"                    -> (1800, "half an hour", "")
    "recuérdame en diez minutos sacar la pizza"       -> (600, "diez minutos", "sacar la pizza")

parse_timer finds a duration (a number and a unit, or "half an hour" / "media
hora") and what to remind about: whatever comes after it, minus little words
like "to" / "que". describe() says a number of seconds the other way round.
"""
import re

from transcriber import normalize

MAX_SECONDS = 12 * 3600

LANGUAGES = {
    "en": {
        "numbers": {
            "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
            "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "couple": 2,
        },
        "units": {
            "second": 1, "seconds": 1, "sec": 1, "secs": 1,
            "minute": 60, "minutes": 60, "min": 60, "mins": 60,
            "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
        },
        # "twenty five", "forty-five" (the hyphen splits it into two words)
        "tens_joiner": None,
        "phrases": {("half", "an", "hour"): 1800, ("half", "hour"): 1800, ("quarter", "of", "an", "hour"): 900},
        # Little words between the duration and the message: "in 10 minutes TO call mom".
        "filler": {"to", "that", "about", "for", "and", "please", "of", "so"},
        "names": {1: ("second", "seconds"), 60: ("minute", "minutes"), 3600: ("hour", "hours")},
    },
    "es": {
        "numbers": {
            "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
            "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
            "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
            "veinticinco": 25, "treinta": 30, "cuarenta": 40, "cuarentaicinco": 45, "cincuenta": 50,
            "sesenta": 60, "noventa": 90,
        },
        "units": {
            "segundo": 1, "segundos": 1, "seg": 1,
            "minuto": 60, "minutos": 60, "min": 60,
            "hora": 3600, "horas": 3600,
        },
        "tens_joiner": "y",  # "cuarenta y cinco minutos"
        "phrases": {("media", "hora"): 1800, ("un", "cuarto", "de", "hora"): 900},
        "filler": {"que", "para", "a", "de", "y", "e", "pa", "por", "favor"},
        "names": {1: ("segundo", "segundos"), 60: ("minuto", "minutos"), 3600: ("hora", "horas")},
    },
}
DEFAULT_LANGUAGE = "en"


def _parse(words: list[str], norm: list[str], spec: dict) -> tuple[int, str, str] | None:
    numbers, units = spec["numbers"], spec["units"]
    number = lambda w: int(w) if w.isdigit() else numbers.get(w)
    for i, word in enumerate(norm):
        phrase = next((p for p in spec["phrases"] if tuple(norm[i: i + len(p)]) == p), None)
        if phrase:
            seconds, start, end = spec["phrases"][phrase], i, i + len(phrase)
        elif word in units and i > 0 and (n := number(norm[i - 1])):
            start = i - 1
            joiner = spec["tens_joiner"]
            if joiner and i >= 3 and norm[i - 2] == joiner and (tens := number(norm[i - 3])) \
                    and tens % 10 == 0 and n < 10:
                n, start = tens + n, i - 3  # "cuarenta y cinco"
            elif not joiner and i >= 2 and (tens := number(norm[i - 2])) and tens % 10 == 0 \
                    and tens >= 20 and n < 10:
                n, start = tens + n, i - 2  # "twenty five"
            if start > 0 and norm[start - 1] in ("a", "an") and n == 2 and norm[start] == "couple":
                start -= 1  # "a couple minutes"
            seconds, end = n * units[word], i + 1
        else:
            continue
        if not 0 < seconds <= MAX_SECONDS:
            return None
        rest = words[end:]
        while rest and normalize(rest[0]) in spec["filler"]:
            rest = rest[1:]
        return seconds, " ".join(words[start:end]), " ".join(rest)
    return None


def parse_timer(text: str, lang: str | None = None) -> tuple[int, str, str] | None:
    """-> (seconds, the duration as said, the message or ""), or None.
    Tries `lang` first, then the other languages (people mix them)."""
    words = re.findall(r"[^\W_]+", text.lower())  # keeps accents and digits
    norm = [normalize(w) for w in words]
    order = [lang] if lang in LANGUAGES else []
    order += [code for code in LANGUAGES if code not in order]
    for code in order:
        parsed = _parse(words, norm, LANGUAGES[code])
        if parsed:
            return parsed
    return None


def describe(seconds: float, lang: str | None = None) -> str:
    """600 -> "10 minutes" / "10 minutos"; 90 -> "1.5 minutes"; 3600 -> "1 hour"."""
    names = LANGUAGES.get(lang or DEFAULT_LANGUAGE, LANGUAGES[DEFAULT_LANGUAGE])["names"]
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
