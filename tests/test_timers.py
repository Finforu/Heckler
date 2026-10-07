from datetime import datetime

import pytest

from timers import clock_text, describe, next_time, parse_clock, parse_duration, parse_timer, spoken_left


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
    # several parts, halves, decimals
    ("remind me in an hour and a half to stretch", "en", (5400, "an hour and a half", "stretch")),
    ("set a timer for 1 hour and 30 minutes", "en", (5400, "1 hour and 30 minutes", "")),
    ("timer 1 hour 15 minutes 30 seconds", "en", (4530, "1 hour 15 minutes 30 seconds", "")),
    ("remind me in a minute and a half", "en", (90, "a minute and a half", "")),
    ("remind me in 1.5 hours", "en", (5400, "1.5 hours", "")),
    ("remind me in a couple of minutes", "en", (120, "a couple of minutes", "")),
    ("set a 10min timer", "en", (600, "10 min", "timer")),
    ("avísame en hora y media", "es", (5400, "hora y media", "")),
    ("recuérdame en una hora y media que salga", "es", (5400, "una hora y media", "salga")),
    ("en dos horas y cuarto", "es", (8100, "dos horas y cuarto", "")),
    ("en una hora y 20 minutos", "es", (4800, "una hora y 20 minutos", "")),
    ("en dos minutos y medio", "es", (150, "dos minutos y medio", "")),
    ("en veintidós minutos", "es", (1320, "veintidós minutos", "")),
    ("en 10 minutos y sacar la pizza", "es", (600, "10 minutos", "sacar la pizza")),
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


NOW = datetime(2026, 10, 7, 15, 0)  # 3 PM


@pytest.mark.parametrize("text, lang, seconds, said, message", [
    ("remind me at 5 pm to call mom", "en", 2 * 3600, "5 PM", "call mom"),
    ("remind me at 5:30 to stretch", "en", 2.5 * 3600, "5:30 PM", "stretch"),  # no am/pm: the next one
    ("set an alarm at five thirty p.m.", "en", 2.5 * 3600, "5:30 PM", ""),
    ("at 7 in the morning", "en", 16 * 3600, "7 AM", ""),
    ("at 2 pm", "en", 23 * 3600, "2 PM", ""),  # already past today: tomorrow
    ("at 17:45", "en", 2.75 * 3600, "5:45 PM", ""),
    ("avísame a las 5 y media de la tarde", "es", 2.5 * 3600, "las 5 y media de la tarde", ""),
    ("recuérdame a las cinco que hay partido", "es", 2 * 3600, "las 5 de la tarde", "hay partido"),
    ("a la una", "es", 10 * 3600, "la 1 de la madrugada", ""),
    ("a las 6 menos cuarto", "es", 2.75 * 3600, "las 5 y 45 de la tarde", ""),
])
def test_parse_clock(text, lang, seconds, said, message):
    assert parse_clock(text, lang, NOW) == (int(seconds), said, message)


@pytest.mark.parametrize("text", ["remind me at work", "at 25", "at 17 pm", "remind me in 10 minutes", ""])
def test_no_clock(text):
    assert parse_clock(text, "en", NOW) is None


def test_next_time():
    assert next_time(12, 0, None, NOW) == datetime(2026, 10, 8, 0, 0)  # midnight comes before noon now
    assert next_time(12, 0, "am", NOW) == datetime(2026, 10, 8, 0, 0)
    assert next_time(12, 0, "pm", NOW) == datetime(2026, 10, 8, 12, 0)
    assert next_time(15, 0, None, NOW) == datetime(2026, 10, 8, 15, 0)  # exactly now: tomorrow


@pytest.mark.parametrize("hour, minute, lang, expected", [
    (17, 0, "en", "5 PM"), (0, 5, "en", "12:05 AM"), (12, 30, "en", "12:30 PM"),
    (9, 15, "es", "las 9 y cuarto de la mañana"), (13, 0, "es", "la 1 de la tarde"), (21, 40, "es", "las 9 y 40 de la noche"),
])
def test_clock_text(hour, minute, lang, expected):
    assert clock_text(datetime(2026, 1, 1, hour, minute), lang) == expected


@pytest.mark.parametrize("text, seconds", [
    ("10", 600), ("2.5", 150), ("1h30m", 5400), ("90s", 90), ("1h 15m", 4500), ("45 min", 2700),
    ("half an hour", 1800), ("media hora", 1800), ("0", None), ("800", None), ("soon", None), ("", None),
])
def test_parse_duration(text, seconds):
    assert parse_duration(text, "en") == seconds


@pytest.mark.parametrize("seconds, lang, expected", [
    (12, "en", "12 seconds"), (1, "en", "1 second"), (255, "en", "4 minutes"), (61, "en", "1 minute"),
    (3600, "en", "1 hour"), (4810, "en", "1 hour and 20 minutes"), (7200, "es", "2 horas"),
    (4810, "es", "1 hora y 20 minutos"),
])
def test_spoken_left(seconds, lang, expected):
    assert spoken_left(seconds, lang) == expected
