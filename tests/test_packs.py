import random
import zipfile
from pathlib import Path

import pytest
import yaml

import packs
from reactions import ReactionEngine

ROOT = Path(__file__).resolve().parent.parent
PACKS = ROOT / "packs"


@pytest.fixture
def full_guild(store, tmp_path):
    """Guild 1: the Spanish starter pack (gags turned on) plus voices, sounds,
    a skit, settings, a nickname and someone's own greeting."""
    packs.import_pack(store, 1, PACKS / "base-es.yaml")
    for row in store.list_reactions(1):
        store.update_reaction(row["id"], enabled=True)
    store.set_person(1, 42, nickname="Pepito")
    store.add_reaction(1, "response", "hello: Pepito", [{"type": "event", "event": "hello", "user_id": 42}],
                       [[{"type": "say", "text": "Llegó Pepito."}]])
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "bruh.OGG").write_bytes(b"fake ogg")
    (tmp_path / "audio" / "pirate.wav").write_bytes(b"fake wav")
    bruh = store.add_sound(1, "bruh", "audio/bruh.OGG", duration_s=1.2, gain_db=-3)
    pirate = store.add_voice("Pirate Arr!", guild_id=1, kind="clone", ref_text="arr matey", source_path="audio/pirate.wav",
                             tags=["fun"], status="ready", prompt_path="models/x.pt")
    store.add_voice("narrator", guild_id=1, kind="designed", instruct="deep, slow")
    store.add_voice("someone", guild_id=1, kind="speaker", owner_user_id=42)
    store.add_reaction(1, "gag", "skit", [{"type": "phrase", "phrases": ["hola"]}, {"type": "slash", "name": "hola"}], [
        [{"type": "say", "text": "Ahoy", "voice_id": pirate}, {"type": "sound", "sound_id": bruh}],
        [{"type": "say", "text": "Tú también", "voice_id": "@speaker"}],
        [{"type": "say", "text": "Nada"}],
    ], voice_id=pirate, by_users=["Pepe"], cooldown_s=5, chance=0.5, enabled=False, created_by=42, status="pending")
    store.set_setting(1, "gags.cooldown_s", 1)
    store.set_setting(1, "bot.wake_words", ["robo"])
    return tmp_path


def comparable(store, guild_id):
    """The server's content with ids swapped for names, as packs see it."""
    voices = {v["id"]: v["name"] for v in store.list_voices(guild_id)}
    sounds = {s["id"]: s["name"] for s in store.list_sounds(guild_id)}

    def step(s):
        s = dict(s)
        if s.get("voice_id") is not None and s["voice_id"] != "@speaker":
            s["voice_id"] = voices[s["voice_id"]]
        if "sound_id" in s:
            s["sound_id"] = sounds[s["sound_id"]]
        return s

    reactions = []
    for r in store.list_reactions(guild_id):
        r = {k: v for k, v in r.items() if k not in ("id", "guild_id", "position", "created_at", "updated_at",
                                                     "uses", "last_used_at")}
        r["voice_id"] = voices.get(r["voice_id"], r["voice_id"])
        r["options"] = [[step(s) for s in o] for o in r["options"]]
        reactions.append(r)
    return {
        "reactions": reactions,
        "settings": store.settings(guild_id, effective=False),
        "people": {p["user_id"]: p["nickname"] for p in store.list_people(guild_id)},
        "sounds": [{k: s[k] for k in ("name", "duration_s", "gain_db", "enabled")} for s in store.list_sounds(guild_id)],
        "voices": [{k: v[k] for k in ("name", "kind", "ref_text", "instruct", "tags", "owner_user_id")}
                   for v in store.list_voices(guild_id, include_global=False)],
    }


def test_round_trip(store, full_guild):
    tmp = full_guild
    path = packs.export_pack(store, 1, tmp / "out" / "pack.yaml", base_dir=tmp)
    with zipfile.ZipFile(path.with_suffix(".zip")) as z:
        assert sorted(z.namelist()) == ["sounds/bruh.ogg", "voices/Pirate-Arr-source.wav"]

    result = packs.import_pack(store, 2, path, mode="replace", base_dir=tmp)
    assert result["warnings"] == []
    assert comparable(store, 2) == comparable(store, 1)

    # the audio came along, into data/media/<guild>
    sound = store.find_sound(2, "bruh")
    assert sound["path"] == "data/media/2/sounds/bruh.ogg" and (tmp / sound["path"]).read_bytes() == b"fake ogg"
    voice = store.find_voice(2, "pirate arr!")
    assert (tmp / voice["source_path"]).read_bytes() == b"fake wav"
    assert voice["status"] == "queued" and voice["prompt_path"] is None  # prompts are rebuilt, never packed

    # exporting the copy gives the same pack
    again = packs.export_pack(store, 2, tmp / "again.yaml", base_dir=tmp)
    assert again.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_the_copy_behaves_the_same(store, full_guild):
    path = packs.export_pack(store, 1, full_guild / "pack.yaml", base_dir=full_guild)
    packs.import_pack(store, 2, path, base_dir=full_guild)
    store.set_setting(2, "gags.cooldown_s", 0)
    store.set_setting(1, "gags.cooldown_s", 0)
    engine = ReactionEngine(store)
    for i, text in enumerate(["la taza de café", "¡Buenas noches!", "tengo hambre", "la taza café", "nada"] * 3):
        random.seed(i)
        a = engine.match(1, text, 5, "Ana", "Ana")
        random.seed(i)
        b = engine.match(2, text, 5, "Ana", "Ana")
        assert (a and a.text) == (b and b.text)


def test_merge_and_replace(store, tmp_path):
    say = lambda text: [[{"type": "say", "text": text}]]
    hola = [{"type": "phrase", "phrases": ["hola"]}]
    store.add_reaction(1, "gag", "hola", hola, say("from pack"))
    path = packs.export_pack(store, 1, tmp_path / "p.yaml")

    store.add_reaction(2, "gag", "hola", hola, say("old"))
    store.add_reaction(2, "gag", "other", hola, say("other"))
    store.add_reaction(2, "command", "hola", [{"type": "command", "phrases": ["hola"]}],
                       [[{"type": "builtin", "action": "stop"}]])  # same name, other kind: not the same reaction
    packs.import_pack(store, 2, path)
    rows = {(r["kind"], r["name"]): r for r in store.list_reactions(2)}
    assert len(rows) == 3 and rows[("gag", "hola")]["options"] == say("from pack")

    packs.import_pack(store, 2, path, mode="replace")
    assert [(r["kind"], r["name"]) for r in store.list_reactions(2)] == [("gag", "hola")]


def test_personal_data_can_be_left_out(store, full_guild):
    store.set_person(1, 42, nickname="Pepito")
    data, _ = packs.pack_data(store, 1, include_personal=False, base_dir=full_guild)
    assert "people" not in data
    assert all("user_id" not in r and "created_by" not in r for r in data["reactions"])
    assert "someone" not in [v["name"] for v in data["voices"]]
    full, _ = packs.pack_data(store, 1, base_dir=full_guild)
    assert full["people"] and any(r.get("user_id") for r in full["reactions"])


def test_hand_written_pack(store, tmp_path):
    store.add_voice("bot")  # a global voice the pack can name
    path = tmp_path / "hand.yaml"
    path.write_text("""
format: 1
reactions:
  - name: hola
    phrase: hola
    voice: bot
    options:
      - Hola.
      - [Mira, {sound: missing}, {say: Eso, voice: "@speaker"}, {say: Y eso, voice: nobody}]
  - name: varios
    triggers: [{phrase: chao}, {swap: arroz, to: mango, connectors: [de]}]
    say: Chao {name}
    by: Pepe
  - name: bienvenida
    event: hello
    user_id: "42"
    say: Llegó el 42
  - name: vete
    command: [vete, largate]
    do: leave
""", encoding="utf-8")
    result = packs.import_pack(store, 1, path)
    assert result["reactions"] == 4
    assert len(result["warnings"]) == 2  # the missing sound and the unknown voice
    rows = {r["name"]: r for r in store.list_reactions(1)}
    assert rows["hola"]["kind"] == "gag" and rows["hola"]["voice_id"] == store.find_voice(1, "bot")["id"]
    assert rows["hola"]["options"][1] == [{"type": "say", "text": "Mira"},
                                          {"type": "say", "text": "Eso", "voice_id": "@speaker"},
                                          {"type": "say", "text": "Y eso", "voice_id": None}]
    assert rows["varios"]["by_users"] == ["Pepe"] and rows["varios"]["triggers"][1]["connectors"] == ["de"]
    assert rows["bienvenida"]["kind"] == "response" and rows["bienvenida"]["triggers"][0]["user_id"] == 42
    assert rows["vete"]["kind"] == "command"
    engine = ReactionEngine(store)
    assert engine.for_command(1, "largate ya").steps == [{"type": "builtin", "action": "leave"}]
    assert engine.for_event(1, "hello", 42).text == "Llegó el 42"


@pytest.mark.parametrize("body, message", [
    ("reactions: [{name: x, say: hi}]", "needs exactly one of"),
    ("reactions: [{name: x, phrase: hi}]", "needs exactly one of say"),
    ("reactions: [{name: x, swap: coffee, say: hi}]", "needs `to`"),
    ("reactions: [{name: x, event: party, say: hi}]", "event must be one of"),
    ("reactions: [{name: x, phrase: hi, do: explode}]", "bad step"),
    ("reactions: [{phrase: hi, say: hi}]", "needs a name"),
    ("format: 99", "newer"),
])
def test_bad_packs_change_nothing(store, tmp_path, body, message):
    path = tmp_path / "bad.yaml"
    path.write_text("settings: {quota.gags: 1}\npeople: [{user_id: 5, nickname: X}]\n" + body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        packs.import_pack(store, 1, path)
    assert store.list_reactions(1) == [] and store.settings(1, effective=False) == {} and store.list_people(1) == []


def test_zip_paths_cannot_escape(store, tmp_path):
    path = tmp_path / "evil.yaml"
    path.write_text("sounds: [{name: x, file: ../../evil.ogg}]\nreactions: []\n", encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as z:
        z.writestr("../../evil.ogg", b"x")
    with pytest.raises(ValueError, match="outside"):
        packs.import_pack(store, 1, path, base_dir=tmp_path)
    assert not (tmp_path.parent / "evil.ogg").exists()


# ------------------------------------------------------------ starter packs
@pytest.mark.parametrize("name, command, said", [
    ("base-es", "vete ya", "5 minutos"),
    ("base-en", "go away now", "5 minutes"),
])
def test_base_packs(store, name, command, said):
    path = PACKS / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "people" not in data and not any("user_id" in r or "by" in r for r in data["reactions"])
    result = packs.import_pack(store, 1, path)
    assert result["warnings"] == []
    rows = store.list_reactions(1)
    assert {r["kind"] for r in rows} == {"command", "response", "gag"}
    assert all(not r["enabled"] for r in rows if r["kind"] == "gag")  # examples, off
    actions = {s["action"] for r in rows for o in r["options"] for s in o if s["type"] == "builtin"}
    assert actions == {"leave", "stop", "timer", "timer_cancel", "timer_list", "time", "coin", "dice", "pick",
                       "repeat"}  # no test-sound "play"
    engine = ReactionEngine(store)
    assert engine.for_command(1, command).steps == [{"type": "builtin", "action": "leave"}]
    for event in ("wake", "ack", "leave", "unknown", "arrival"):
        assert engine.for_event(1, event).text
    assert "Ana" in engine.for_event(1, "hello", 5, "Ana", "Ana").text
    assert "Ana" in engine.for_event(1, "bye", 5, "Ana", "Ana").text
    assert said in engine.for_event(1, "timer_set", values={"said": said}).text
    assert engine.for_event(1, "timer_ring", 5, "Ana", "Ana", {"said": said, "message": "la pizza"}).text \
        == "Ana, la pizza."
    assert said in engine.for_event(1, "timer_ring", 5, "Ana", "Ana", {"said": said, "message": ""}).text
    assert engine.match(1, "buenas noches good night", 5, "Ana", "Ana") is None  # gags start off
    for event in ("timer_cancelled", "timer_none", "nothing_to_repeat", "llm_unavailable"):
        assert engine.for_event(1, event).text
    assert said in engine.for_event(1, "alarm_set", values={"said": said}).text
    for event in ("time_now", "coin_result", "dice_result", "pick_result"):
        assert "XYZ" in engine.for_event(1, event, values={"result": "XYZ"}).text
    left = engine.for_event(1, "timer_left", 5, "Ana", "Ana", {"result": "4 min", "message": "la pizza"}).text
    assert "4 min" in left and "la pizza" in left
    assert engine.for_event(1, "timer_left", 5, "Ana", "Ana", {"result": "4 min", "message": ""}).text


# What people say after the wake word -> the action it sets off.
@pytest.mark.parametrize("name, request_, action", [
    ("base-en", "remind me in 10 minutes to stretch", "timer"),
    ("base-en", "set an alarm at 5 pm", "timer"),
    ("base-en", "cancel my timer", "timer_cancel"),
    ("base-en", "cancel all my timers", "timer_cancel"),
    ("base-en", "stop the timer", "timer_cancel"),
    ("base-en", "how much time is left on my timer", "timer_list"),
    ("base-en", "what time is it", "time"),
    ("base-en", "what's the time", "time"),
    ("base-en", "flip a coin", "coin"),
    ("base-en", "heads or tails", "coin"),
    ("base-en", "roll two dice", "dice"),
    ("base-en", "roll a d20", "dice"),
    ("base-en", "pick someone", "pick"),
    ("base-en", "who goes first", "pick"),
    ("base-en", "say that again", "repeat"),
    ("base-en", "stop", "stop"),
    ("base-es", "recuérdame en 10 minutos sacar la pizza", "timer"),
    ("base-es", "avísame a las 5", "timer"),
    ("base-es", "cancela mi temporizador", "timer_cancel"),
    ("base-es", "para el temporizador", "timer_cancel"),
    ("base-es", "¿cuánto falta?", "timer_list"),
    ("base-es", "¿qué hora es?", "time"),
    ("base-es", "cara o cruz", "coin"),
    ("base-es", "tira dos dados", "dice"),
    ("base-es", "elige a alguien", "pick"),
    ("base-es", "¿qué dijiste?", "repeat"),
    ("base-es", "para", "stop"),
])
def test_helper_commands(store, name, request_, action):
    packs.import_pack(store, 1, PACKS / f"{name}.yaml")
    m = ReactionEngine(store).for_command(1, request_)
    assert m is not None and m.steps == [{"type": "builtin", "action": action}]


def test_english_swap_example(store, tmp_path):
    path = tmp_path / "swap.yaml"
    path.write_text("reactions: [{name: tea, swap: coffee, to: tea, connectors: [of], "
                    "say: 'No, {subject} {connector} {to}.'}]", encoding="utf-8")
    packs.import_pack(store, 1, path)
    engine = ReactionEngine(store)
    assert engine.match(1, "I want the big cup of coffee", 5, "Ana", "Ana").text == "No, the big cup of tea."
    assert engine.match(1, "the cup coffee", 6, "Ana", "Ana") is None  # no swallowed "of" in English


def test_seed_guild(store, tmp_path):
    assert packs.seed_guild(store, 1) == "pack base-en: 34 reactions"  # the default: English
    assert store.get_setting(1, packs.STARTER_VERSION) == 2
    assert packs.seed_guild(store, 1) == "kept: the server already has content"
    assert len(store.list_reactions(1)) == 34

    store.set_setting(2, "language", "es")  # the server's language picks the pack
    assert packs.seed_guild(store, 2) == "pack base-es: 34 reactions"
    assert store.get_setting(2, "voices.preview_text") == "¡Hola! Así suena mi voz."
    store.set_setting(3, "language", "xx")  # a language with no pack: English
    assert packs.seed_guild(store, 3).startswith("pack base-en")

    store.set_setting(4, "content.starter_pack", "base-es")  # an explicit choice wins
    assert packs.seed_guild(store, 4).startswith("pack base-es")
    store.set_setting(5, "content.starter_pack", "none")
    assert packs.seed_guild(store, 5).startswith("nothing") and store.list_reactions(5) == []
    store.set_setting(6, "content.starter_pack", "missing")
    assert "no pack missing.yaml" in packs.seed_guild(store, 6)
    store.set_setting(7, "content.starter_pack", "../store")
    assert "isn't a pack name" in packs.seed_guild(store, 7)
    store.set_setting(0, "content.starter_pack", "base-es")  # a global choice works too
    assert packs.seed_guild(store, 8).startswith("pack base-es")


def test_upgrade_adds_only_newer_reactions(store, tmp_path):
    """A server seeded from version 1 gets what version 2 added, once, and
    keeps its own edits and deletions of the old content."""
    data = yaml.safe_load((PACKS / "base-en.yaml").read_text(encoding="utf-8"))
    old = dict(data, version=1, reactions=[r for r in data["reactions"] if r.get("since", 1) == 1])
    (tmp_path / "packs").mkdir()
    (tmp_path / "packs" / "base-en.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
    assert packs.seed_guild(store, 1, packs_dir=tmp_path / "packs") == "pack base-en: 16 reactions"

    leave = next(r for r in store.list_reactions(1) if r["name"] == "leave" and r["kind"] == "command")
    store.delete_reaction(leave["id"])  # the admin removed one
    stop = next(r for r in store.list_reactions(1) if r["name"] == "stop")
    store.update_reaction(stop["id"], triggers=[{"type": "command", "phrases": ["shush"]}])  # and changed one
    store.add_reaction(1, "command", "coin", [{"type": "command", "phrases": ["my coin"]}],
                       [[{"type": "say", "text": "Mine"}]])  # and already made their own "coin"

    assert packs.seed_guild(store, 1) == "pack base-en v2: added 17 new reactions"
    names = [(r["kind"], r["name"]) for r in store.list_reactions(1)]
    assert len(names) == 16 - 1 + 1 + 17 and ("command", "leave") not in names
    assert names.count(("command", "coin")) == 1
    assert store.get_reaction(stop["id"])["triggers"] == [{"type": "command", "phrases": ["shush"]}]
    assert store.get_setting(1, packs.STARTER_VERSION) == 2
    assert packs.seed_guild(store, 1) == "kept: the server already has content"  # only once


def test_starter_version_isnt_exported(store):
    packs.seed_guild(store, 1)
    data, _ = packs.pack_data(store, 1)
    assert packs.STARTER_VERSION not in data.get("settings", {})
