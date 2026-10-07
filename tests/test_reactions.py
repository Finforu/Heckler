"""The reaction engine on its own: voices, order, chance, cooldowns, dry runs,
explain, events, commands, skits."""
import random

import pytest

from reactions import Match, ReactionEngine

G = 1


def test_commands_longer_phrase_wins_ties(store):
    leave = [[{"type": "builtin", "action": "leave"}]]
    stop = [[{"type": "builtin", "action": "stop"}]]
    store.add_reaction(G, "command", "short", [{"type": "command", "phrases": ["para"]}], leave)
    store.add_reaction(G, "command", "long", [{"type": "command", "phrases": ["Para la música"]}], stop)
    engine = ReactionEngine(store)
    assert engine.for_command(G, "para la musica").reaction["name"] == "long"
    assert engine.for_command(G, "oye para").reaction["name"] == "short"
    assert engine.for_command(G, "para ya, para la musica").reaction["name"] == "short"  # earliest wins


# --------------------------------------------------- engine-only features
def say(text, voice_id=None):
    step = {"type": "say", "text": text}
    if voice_id is not None:
        step["voice_id"] = voice_id
    return step


def phrase(*phrases):
    return [{"type": "phrase", "phrases": list(phrases)}]


def test_voice_resolution(store):
    narrator = store.add_voice("narrator", guild_id=G)
    pirate = store.add_voice("pirate", guild_id=G)
    store.add_reaction(G, "gag", "a", phrase("hola"),
                       [[say("uno", pirate), say("dos"), say("tres", "@speaker")]], voice_id=narrator)
    store.add_reaction(G, "gag", "b", phrase("chao"), [[say("cuatro")]])
    engine = ReactionEngine(store)
    match = engine.match(G, "hola", 1, "x", "x")
    assert [s["voice_id"] for s in match.steps] == [pirate, narrator, "@speaker"]
    assert engine.match(G, "chao", 1, "x", "x").steps[0]["voice_id"] is None
    store.set_setting(G, "voice.default", pirate)  # recompiles by itself
    assert engine.match(G, "chao", 2, "x", "x").steps[0]["voice_id"] == pirate
    assert ("tres", "@speaker") not in engine.fixed_texts(G)
    assert ("cuatro", pirate) in engine.fixed_texts(G)


def test_enabled_status_and_live_changes(store):
    a = store.add_reaction(G, "gag", "a", phrase("hola"), [[say("A")]], enabled=False)
    b = store.add_reaction(G, "gag", "b", phrase("hola"), [[say("B")]], status="pending")
    engine = ReactionEngine(store)
    store.set_setting(G, "gags.cooldown_s", 0)
    assert engine.match(G, "hola", 1, "x", "x") is None
    store.update_reaction(b, status="approved")
    assert engine.match(G, "hola", 1, "x", "x").text == "B"
    store.update_reaction(a, enabled=True)
    assert engine.match(G, "hola", 1, "x", "x").text == "A"
    store.reorder_reactions(G, [b, a])
    assert engine.match(G, "hola", 1, "x", "x").text == "B"
    assert engine.match(G + 1, "hola", 1, "x", "x") is None  # per server


def fires(match) -> bool:
    return match is not None and not match.blocked


def test_chance_and_intensity(store):
    store.set_setting(G, "gags.cooldown_s", 0)
    store.add_reaction(G, "gag", "never", phrase("hola"), [[say("never")]], chance=0)
    store.add_reaction(G, "gag", "later", phrase("hola"), [[say("later")]])
    engine = ReactionEngine(store)
    # a reaction that matched but whose chance said no stops there, and says so
    m = engine.match(G, "hola", 1, "x", "x")
    assert m.blocked == "chance" and m.steps == [] and m.reaction["name"] == "never" and m.chance == 0

    store.add_reaction(G, "gag", "half", phrase("chao"), [[say("half")]], chance=0.5)
    random.seed(3)
    results = [engine.match(G, "chao", 1, "x", "x") for _ in range(400)]
    assert 150 < sum(map(fires, results)) < 250
    assert {m.blocked for m in results} == {None, "chance"}
    store.set_setting(G, "gags.intensity", 0.5)
    assert engine.match(G, "chao", 1, "x", "x", dry_run=True).chance == 0.25
    store.set_setting(G, "gags.intensity", 0)
    assert not any(fires(engine.match(G, "chao", 1, "x", "x")) for _ in range(50))


def test_cooldown_per_person_and_reaction(store):
    clock = [100.0]
    store.add_reaction(G, "gag", "a", phrase("hola"), [[say("A")]])  # gags.cooldown_s: 3 by default
    store.add_reaction(G, "gag", "b", phrase("chao"), [[say("B")]], cooldown_s=10)
    store.add_reaction(G, "gag", "c", phrase("hola"), [[say("C")]])
    store.add_reaction(G, "response", "hi", [{"type": "event", "event": "hello"}], [[say("hi")]])
    engine = ReactionEngine(store, clock=lambda: clock[0])
    assert fires(engine.match(G, "hola", 1, "x", "x"))
    m = engine.match(G, "hola", 1, "x", "x")              # same person, too soon
    assert m.blocked == "cooldown" and m.reaction["name"] == "a"  # and "c" doesn't get a turn
    assert fires(engine.match(G, "hola", 2, "x", "x"))    # someone else
    assert fires(engine.match(G, "chao", 1, "x", "x"))    # another reaction
    clock[0] += 3
    assert fires(engine.match(G, "hola", 1, "x", "x"))
    assert engine.match(G, "chao", 1, "x", "x").blocked == "cooldown"  # its own 10 s
    clock[0] += 7
    assert fires(engine.match(G, "chao", 1, "x", "x"))
    # events don't use the gag cooldown
    assert fires(engine.for_event(G, "hello", 1)) and fires(engine.for_event(G, "hello", 1))


def test_blocked_commands_and_events(store):
    store.add_reaction(G, "command", "vete", [{"type": "command", "phrases": ["vete"]}],
                       [[{"type": "builtin", "action": "leave"}]], cooldown_s=60)
    store.add_reaction(G, "response", "hi", [{"type": "event", "event": "hello"}], [[say("hi")]], chance=0)
    store.add_reaction(G, "sound", "bruh", [{"type": "slash", "name": "bruh"}], [[say("bruh")]], chance=0)
    engine = ReactionEngine(store)
    assert fires(engine.for_command(G, "vete", 1))
    assert engine.for_command(G, "vete", 1).blocked == "cooldown"
    assert engine.for_event(G, "hello", 1).blocked == "chance"
    assert engine.for_slash(G, "bruh").blocked == "chance"


def test_dry_run_changes_nothing(store):
    clock = [100.0]
    store.add_reaction(G, "gag", "a", phrase("hola"), [[say("uno")], [say("dos")]], chance=0.0001)
    store.add_reaction(G, "command", "vete", [{"type": "command", "phrases": ["vete"]}],
                       [[{"type": "builtin", "action": "leave"}]], cooldown_s=60)
    store.add_reaction(G, "response", "hi", [{"type": "event", "event": "hello"}], [[say("hi {name}")]],
                       cooldown_s=60)
    engine = ReactionEngine(store, clock=lambda: clock[0])
    random.seed(5)
    state = random.getstate()
    for _ in range(20):
        m = engine.match(G, "hola", 1, "x", "x", dry_run=True)
        assert fires(m) and m.chance == 0.0001  # the chance isn't rolled, only reported
        assert fires(engine.for_command(G, "vete", 1, dry_run=True))
        assert engine.for_event(G, "hello", 1, name="Ana", dry_run=True).text == "hi Ana"
        assert fires(engine.for_slash(G, "nothing", dry_run=True)) is False
    assert random.getstate() == state  # not even the global random sequence moved
    assert engine._fired == {} and engine._last == {}
    # still free to fire for real
    assert fires(engine.for_command(G, "vete", 1)) and engine.for_command(G, "vete", 1).blocked == "cooldown"
    # a dry run reports the cooldown, without stamping a new one
    assert engine.for_command(G, "vete", 1, dry_run=True).blocked == "cooldown"


def test_explain(store):
    clock = [100.0]
    store.add_reaction(G, "gag", "only pepe", phrase("hola"), [[say("A")]], by_users=["Pepe"])
    store.add_reaction(G, "gag", "needs subject", phrase("hola"), [[say("{subject}")]])
    store.add_reaction(G, "gag", "hola", phrase("hola"), [[say("B")]], chance=0.5)
    store.add_reaction(G, "gag", "hola again", phrase("hola amigo"), [[say("C")]])
    store.add_reaction(G, "gag", "trabajo", phrase("trabajo"), [[say("D")]])
    store.add_reaction(G, "gag", "gato", [{"type": "swap", "word": "gato", "to": "perro"}], [[say("E")]])
    engine = ReactionEngine(store, clock=lambda: clock[0])

    rows = engine.explain(G, "hola amigo, mucho trabaho y gatitos", 1, "Ana", "Ana")
    matched = [(r["name"], r["would_fire"], r["reason"]) for r in rows if not r.get("near_miss")]
    assert matched == [("only pepe", False, "not this user"), ("needs subject", False, "no usable option"),
                       ("hola", True, None), ("hola again", False, "first match wins")]
    assert next(r for r in rows if r["name"] == "hola")["chance"] == 0.5
    near = {(r["name"], r["heard"], r["trigger_word"]) for r in rows if r.get("near_miss")}
    assert ("trabajo", "trabaho", "trabajo") in near  # 0.86, but phrases only match exactly
    assert ("gato", "gatitos", "gato") in near          # 0.73; swap words would match from 0.8
    assert all(r["similarity"] >= 0.6 for r in rows if r.get("near_miss"))
    assert not [r for r in rows if r.get("near_miss") and r["name"] in ("hola", "hola again")]  # they matched
    assert engine._fired == {} and engine._last == {}  # no side effects

    random.seed(1)
    while not fires(engine.match(G, "hola", 1, "Ana", "Ana")):  # chance 0.5
        pass
    rows = engine.explain(G, "hola", 1, "Ana", "Ana")
    assert [(r["name"], r["reason"]) for r in rows] == [
        ("only pepe", "not this user"), ("needs subject", "no usable option"), ("hola", "cooldown")]


def test_unusable_options_let_the_next_reaction_fire(store):
    store.set_setting(G, "gags.cooldown_s", 0)
    store.add_reaction(G, "gag", "needs subject", phrase("hola"), [[say("{subject}!")]])
    store.add_reaction(G, "gag", "plain", phrase("hola"), [[say("{name}, hola")], [say("{message}")]])
    engine = ReactionEngine(store)
    for _ in range(5):
        assert engine.match(G, "hola", 1, "x", "Pepe").text == "Pepe, hola"


def test_personal_events(store):
    general = [{"type": "event", "event": "hello"}]
    store.add_reaction(G, "response", "general", general, [[say("Hola {name}")]])
    store.add_reaction(G, "response", "by id", [{"type": "event", "event": "hello", "user_id": "42"}],
                       [[say("Llegó el 42")]])
    store.add_reaction(G, "response", "by name", general, [[say("Llegó Pepe")]], by_users=["Pépe"])
    engine = ReactionEngine(store)
    assert engine.for_event(G, "hello", 42, "x", "x").text == "Llegó el 42"
    assert engine.for_event(G, "hello", 7, "Pepe", "Pepe").text == "Llegó Pepe"
    assert engine.for_event(G, "hello", 8, "Ana", "Ana").text == "Hola Ana"
    assert engine.for_event(G, "bye", 8, "Ana", "Ana") is None


def test_skits_and_slash(store):
    bruh = store.add_sound(G, "bruh", "bruh.ogg")
    store.add_reaction(G, "sound", "bruh", [{"type": "slash", "name": "/Bruh"}, {"type": "phrase", "phrases": ["bruh"]}],
                       [[say("mira"), {"type": "sound", "sound_id": bruh}, {"type": "builtin", "action": "stop"}]])
    engine = ReactionEngine(store)
    match = engine.for_slash(G, "bruh")
    assert isinstance(match, Match) and [s["type"] for s in match.steps] == ["say", "sound", "builtin"]
    assert match.steps[1]["sound_id"] == bruh and match.text == "mira"
    assert engine.match(G, "puro bruh", 1, "x", "x").trigger["type"] == "phrase"
    assert engine.for_slash(G, "nope") is None
