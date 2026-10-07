import random

import pytest

from helpers import flip, parse_dice, pick, roll


@pytest.mark.parametrize("text, dice", [
    ("roll", (1, 6)), ("roll a die", (1, 6)), ("roll two dice", (2, 6)), ("roll 3 dice", (3, 6)),
    ("roll a d20", (1, 20)), ("roll 2d8", (2, 8)), ("roll a D 12", (1, 12)), ("roll a 20 sided die", (1, 20)),
    ("tira un dado", (1, 6)), ("tira dos dados", (2, 6)), ("tira un dado de 20", (1, 20)),
    ("tira tres dados de 10 caras", (3, 10)), ("roll 99 dice", (10, 6)), ("roll a d1", (1, 2)),
])
def test_parse_dice(text, dice):
    assert parse_dice(text) == dice


def test_roll():
    rng = random.Random(1)
    one = roll(1, 20, "en", rng)
    assert one.isdigit() and 1 <= int(one) <= 20
    two = roll(2, 6, "en", random.Random(3))
    a, rest = two.split(" and ")
    b, total = rest.split(", ")
    assert int(a) + int(b) == int(total.split()[0]) and total.endswith("in total")
    assert " y " in roll(3, 6, "es", rng) and roll(3, 6, "es", rng).endswith("en total")


def test_flip_and_pick():
    assert {flip("en", random.Random(i)) for i in range(20)} == {"Heads", "Tails"}
    assert {flip("es", random.Random(i)) for i in range(20)} == {"Cara", "Cruz"}
    assert flip("xx") in ("Heads", "Tails")
    assert pick(["Ana", "Bo"], random.Random(0)) in ("Ana", "Bo")
    assert pick([]) is None
