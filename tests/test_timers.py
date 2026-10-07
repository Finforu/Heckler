import pytest

from timers import describe, parse_timer


@pytest.mark.parametrize("text, lang, expected", [
    ("remind me in 5 minutes to take the pizza out", "en", (300, "5 minutes", "take the pizza out")),
    ("Set a timer for 10 min", "en", (600, "10 min", "")),
    ("remind me in half an hour about the laundry", "en", (1800, "half an hour", "the laundry")),
    ("remind me in an hour", "en", (3600, "an hour", "")),
    ("remind me in twenty five minutes", "en", (1500, "twenty five minutes", "")),
    ("remind me in forty-five seconds to stretch", "en", (45, "forty five seconds", "stretch")),
    ("timer 2 hours", "en", (7200, "2 hours", "")),
    # Spanish keeps behaving as before
    ("recuérdame en 10 minutos sacar la pizza", "es", (600, "10 minutos", "sacar la pizza")),
    ("avísame en cinco minutos que hay que irse", "es", (300, "cinco minutos", "hay que irse")),
    ("pon un timer de media hora", "es", (1800, "media hora", "")),
    ("recuérdame en cuarenta y cinco minutos", "es", (2700, "cuarenta y cinco minutos", "")),
    ("en una hora para llamar a mamá", "es", (3600, "una hora", "llamar a mamá")),
    ("en un cuarto de hora", "es", (900, "un cuarto de hora", "")),
])
def test_parse(text, lang, expected):
    assert parse_timer(text, lang) == expected


def test_other_languages_are_tried_too():
    assert parse_timer("recuérdame en 10 minutos", "en") == (600, "10 minutos", "")
    assert parse_timer("remind me in 10 minutes", "es") == (600, "10 minutes", "")
    assert parse_timer("remind me in 10 minutes") == (600, "10 minutes", "")


@pytest.mark.parametrize("text", ["remind me later", "set a timer", "recuérdame algo", "in 0 minutes",
                                  "in 999 hours", ""])
def test_no_duration(text):
    assert parse_timer(text, "en") is None


@pytest.mark.parametrize("seconds, lang, expected", [
    (300, "en", "5 minutes"), (60, "en", "1 minute"), (90, "en", "1.5 minutes"), (3600, "en", "1 hour"),
    (5400, "en", "1.5 hours"), (45, "en", "45 seconds"), (1, "en", "1 second"),
    (300, "es", "5 minutos"), (60, "es", "1 minuto"), (7200, "es", "2 horas"), (30, "es", "30 segundos"),
    (300, None, "5 minutes"), (300, "xx", "5 minutes"),
])
def test_describe(seconds, lang, expected):
    assert describe(seconds, lang) == expected
