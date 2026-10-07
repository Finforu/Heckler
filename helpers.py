"""Little voice helpers, in English and Spanish: flip a coin, roll dice,
pick someone in the call. Each returns the words the bot fills into its
reply's {result} ("Heads", "3 and 5, 8 in total", "Ana").

    parse_dice("roll two dice")         -> (2, 6)
    parse_dice("roll a d20")            -> (1, 20)
    parse_dice("tira tres dados de 10") -> (3, 10)

Times of day ("what time is it") are in timers.py (clock_text).
"""
import random
import re

from timers import LANGUAGES as NUMBER_WORDS
from transcriber import normalize

MAX_DICE = 10
MAX_SIDES = 1000
COIN = {"en": ("Heads", "Tails"), "es": ("Cara", "Cruz")}
DICE_WORDS = {"die", "dice", "dado", "dados"}
SIDE_WORDS = {"sided", "sides", "side", "caras", "cara", "lados"}


def _number(word: str) -> int | None:
    if word.isdigit():
        return int(word)
    for spec in NUMBER_WORDS.values():
        if word in spec["numbers"] and word not in ("a", "an", "couple"):
            return int(spec["numbers"][word])
    return None


def parse_dice(text: str) -> tuple[int, int]:
    """(how many dice, how many sides) asked for; one six-sided die unless said otherwise."""
    count, sides = 1, 6
    lowered = text.lower()
    if m := re.search(r"\b(\d{0,2})\s*d\s*(\d{1,4})\b", lowered):  # "d20", "3d6", "d 20"
        count, sides = int(m[1] or 1), int(m[2])
    else:
        words = normalize(lowered).split()
        for i, word in enumerate(words):
            if word in DICE_WORDS and i > 0 and (n := _number(words[i - 1])):
                count = n                                           # "two dice", "dos dados"
            if word in SIDE_WORDS and i > 0 and (n := _number(words[i - 1])):
                sides = n                                           # "a 20 sided die", "dado de 20 caras"
            if word in DICE_WORDS and words[i + 1:i + 2] == ["de"] and (n := _number(words[i + 2] if i + 2 < len(words) else "")):
                sides = n                                           # "dado de 20"
    return min(max(count, 1), MAX_DICE), min(max(sides, 2), MAX_SIDES)


def roll(count: int, sides: int, lang: str | None = None, rng: random.Random | None = None) -> str:
    """"17" for one die; "3 and 5, 8 in total" / "3 y 5, 8 en total" for several."""
    rng = rng or random
    rolls = [rng.randint(1, sides) for _ in range(count)]
    if len(rolls) == 1:
        return str(rolls[0])
    joiner, total = (" y ", "en total") if lang == "es" else (" and ", "in total")
    listed = ", ".join(map(str, rolls[:-1])) + joiner + str(rolls[-1])
    return f"{listed}, {sum(rolls)} {total}"


def flip(lang: str | None = None, rng: random.Random | None = None) -> str:
    return (rng or random).choice(COIN.get(lang or "en", COIN["en"]))


def pick(names: list[str], rng: random.Random | None = None) -> str | None:
    """One of `names`, or None when there's nobody to pick."""
    return (rng or random).choice(names) if names else None
