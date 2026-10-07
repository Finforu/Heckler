"""The /gag /sound /voices rules, without Discord."""
import asyncio
import io

import numpy as np
import pytest
import soundfile as sf

import content_commands as cc
from content_commands import Actor, CommandError, QuotaReached
from reactions import ReactionEngine
from sound_library import SoundLibrary
from voice_library import VoiceLibrary

G = 1
ME = Actor(42, "Pepe")
OTHER = Actor(43, "Ana")
ADMIN = Actor(7, "Boss", is_admin=True)


class Services:
    def __init__(self, store, root):
        self.store = store
        self.engine = ReactionEngine(store)
        self.library = VoiceLibrary(store, root / "data")
        self.sounds = SoundLibrary(store, root / "data")
        self.built = []
        self.on = True

    def voice_on(self):
        return self.on

    def queue_build(self, voice_id):
        self.built.append(voice_id)


@pytest.fixture
def services(store, tmp_path):
    return Services(store, tmp_path)


def wav(seconds=1.0, rate=48000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    buf = io.BytesIO()
    sf.write(buf, (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), rate, format="WAV")
    return buf.getvalue()


# ------------------------------------------------------------ permissions
def test_is_admin(store):
    assert cc.is_admin(store, G, is_owner=True, manage_guild=False)
    assert cc.is_admin(store, G, is_owner=False, manage_guild=True)
    assert not cc.is_admin(store, G, is_owner=False, manage_guild=False, role_ids=[5])
    store.set_setting(G, "admin.role_id", 5)
    assert cc.is_admin(store, G, is_owner=False, manage_guild=False, role_ids=[3, 5])
    assert not cc.is_admin(store, 2, is_owner=False, manage_guild=False, role_ids=[5])  # per server


# ------------------------------------------------------------ gags
def test_add_gag_and_it_works(services, store):
    text = cc.add_gag(services, G, ME, "buenas noches, buena noche", "Buenas noches, {name}.\nA dormir.")
    assert "✅" in text and "1/15" in text
    row = store.list_reactions(G, kind="gag")[0]
    assert row["name"] == "buenas noches" and row["created_by"] == ME.user_id and row["status"] == "approved"
    assert row["triggers"] == [{"type": "phrase", "phrases": ["buenas noches", "buena noche"]}]
    assert len(row["options"]) == 2
    m = services.engine.match(G, "bueno, buena noche a todos", 1, "Ana", "Ana")
    assert m.text in ("Buenas noches, Ana.", "A dormir.")


def test_gag_options(services, store):
    voice = store.add_voice("narrator", guild_id=G, kind="designed", instruct="male", status="ready")
    cc.add_gag(services, G, ME, "hola mundo", "hola", only_me=True, voice=str(voice))
    cc.add_gag(services, G, ME, "hola mundo", "otra vez", voice="@speaker")  # same trigger: a new name
    first, second = store.list_reactions(G, kind="gag")
    assert first["by_users"] == [ME.user_id] and first["voice_id"] == voice
    assert second["name"] == "hola mundo 2" and second["voice_id"] == "@speaker"
    for bad in ("nobody", "999"):
        with pytest.raises(CommandError, match="don't know the voice"):
            cc.add_gag(services, G, ME, "chao mundo", "x", voice=bad)
    store.update_voice(voice, status="building")
    with pytest.raises(CommandError, match="isn't ready"):
        cc.add_gag(services, G, ME, "chao mundo", "x", voice="narrator")


@pytest.mark.parametrize("triggers", ["que", "la, el", "jaja", "si no", "", " , "])
def test_broad_triggers_are_refused(services, triggers):
    with pytest.raises(CommandError):
        cc.add_gag(services, G, ME, triggers, "respuesta")


def test_admins_only_get_a_warning_for_broad_triggers(services):
    assert "⚠️" in cc.add_gag(services, G, ADMIN, "jaja", "respuesta")


def test_reply_limits(services):
    with pytest.raises(CommandError):
        cc.add_gag(services, G, ME, "buenas noches", "")
    with pytest.raises(CommandError):
        cc.add_gag(services, G, ME, "buenas noches", "\n".join(f"r{i}" for i in range(11)))
    with pytest.raises(CommandError):
        cc.add_gag(services, G, ME, "buenas noches", "x" * 400)


def test_overlap_warning(services):
    cc.add_gag(services, G, ME, "buenas noches", "a")
    text = cc.add_gag(services, G, OTHER, "noches", "b")
    assert "Overlaps with **buenas noches**" in text
    assert "Overlaps" not in cc.add_gag(services, G, OTHER, "pizza fria", "c")


def test_needs_approval(services, store):
    store.set_setting(G, "user_content.needs_approval", True)
    assert "waiting for an admin's approval" in cc.add_gag(services, G, ME, "buenas noches", "a")
    assert store.list_reactions(G, kind="gag")[0]["status"] == "pending"
    assert services.engine.match(G, "buenas noches", 1, "x", "x") is None
    cc.add_gag(services, G, ADMIN, "buenos dias", "a")  # admins skip the queue
    assert store.list_reactions(G, kind="gag")[1]["status"] == "approved"


def test_gag_quota_and_request_flow(services, store):
    store.set_setting(G, "quota.gags", 2)
    cc.add_gag(services, G, ME, "uno uno", "a")
    cc.add_gag(services, G, ME, "dos dos", "a")
    with pytest.raises(QuotaReached) as limit:
        cc.add_gag(services, G, ME, "tres tres", "a")
    assert limit.value.resource == "gags" and (limit.value.used, limit.value.limit) == (2, 2)
    cc.add_gag(services, G, ADMIN, "tres tres", "a")  # admins aren't limited

    request_id = cc.file_quota_request(services, G, ME.user_id, "gags")
    request = store.get_quota_request(request_id)
    assert request["amount"] == cc.GRANT and "<@42>" in cc.request_text(services, request, "Server")
    with pytest.raises(CommandError, match="Only admins"):
        cc.decide_request(services, request_id, True, ME)
    text, request = cc.decide_request(services, request_id, True, ADMIN)
    assert "+5" in text and request["status"] == "approved" and "5 more" in cc.decision_text(services, request)
    assert store.quota(G, ME.user_id, "gags") == (2, 7)
    cc.add_gag(services, G, ME, "tres tres", "a")
    with pytest.raises(CommandError, match="already decided"):
        cc.decide_request(services, request_id, False, ADMIN)

    denied = cc.file_quota_request(services, G, ME.user_id, "gags")
    text, request = cc.decide_request(services, denied, False, ADMIN)
    assert request["status"] == "denied" and "denied" in cc.decision_text(services, request)


def test_remove_list_and_autocomplete(services, store):
    cc.add_gag(services, G, ME, "pizza fria", "a")
    cc.add_gag(services, G, OTHER, "pasta tibia", "b")
    rows = store.list_reactions(G, kind="gag")
    assert cc.own_names(rows, ME) == ["pizza fria"]
    assert cc.own_names(rows, ADMIN, "PAS") == ["pasta tibia"]
    listing = cc.list_gags(services, G, ME)
    assert "★ **pizza fria**" in listing and "**pasta tibia**" in listing and "★ **pasta" not in listing

    with pytest.raises(CommandError, match="isn't yours"):
        cc.remove_gag(services, G, ME, "pasta tibia")
    with pytest.raises(CommandError, match="no gag called"):
        cc.remove_gag(services, G, ME, "nada")
    assert "🗑️" in cc.remove_gag(services, G, ME, "Pizza Fria")
    assert "🗑️" in cc.remove_gag(services, G, ADMIN, "pasta tibia")
    assert store.list_reactions(G, kind="gag") == []


def test_explain_text(services):
    cc.add_gag(services, G, ME, "pizza fria", "a", only_me=True)
    cc.add_gag(services, G, OTHER, "trabajo", "b")
    text = cc.explain_text(services, G, OTHER, "una pizza fria y mucho trabaho")
    assert "🚫 pizza fria: only for other people" in text
    assert "Heard “trabaho”, close to “trabajo”" in text
    assert "would go off" in cc.explain_text(services, G, ME, "pizza fria")
    assert "Nothing would go off" in cc.explain_text(services, G, ME, "nada que ver")


# ------------------------------------------------------------ sounds
def test_sounds(services, store):
    text = cc.add_sound(services, G, ME, "Air Horn", wav(1), "horn.wav", triggers="bocina fuerte",
                        command="toca la bocina")
    assert "🔊 Sound **air-horn** added" in text and "1/15" in text
    sound = store.find_sound(G, "air-horn")
    row = store.list_reactions(G, kind="sound")[0]
    assert row["name"] == "air-horn" and row["created_by"] == ME.user_id
    assert [t["type"] for t in row["triggers"]] == ["slash", "phrase", "command"]
    assert row["options"] == [[{"type": "sound", "sound_id": sound["id"]}]]
    assert services.engine.for_slash(G, "air-horn").steps[0]["sound_id"] == sound["id"]
    assert services.engine.match(G, "suena la bocina fuerte", 1, "x", "x").reaction["id"] == row["id"]
    assert services.engine.for_command(G, "toca la bocina ya").reaction["id"] == row["id"]

    with pytest.raises(CommandError, match="already used"):  # someone else's command phrase
        cc.add_sound(services, G, OTHER, "otro", wav(1), command="toca la bocina")
    with pytest.raises(CommandError, match="already"):
        cc.add_sound(services, G, OTHER, "air horn", wav(1))
    with pytest.raises(CommandError, match="Too broad"):
        cc.add_sound(services, G, OTHER, "otro", wav(1), triggers="si")
    with pytest.raises(CommandError):
        cc.add_sound(services, G, OTHER, "otro", b"not audio" * 50)
    assert len(store.list_sounds(G)) == 1 and len(store.list_reactions(G, kind="sound")) == 1

    assert cc.find_sound(services, G, "Air Horn")["id"] == sound["id"]
    assert cc.sound_names(services, G, "air") == ["air-horn"]
    assert "★ **air-horn**" in cc.list_sounds(services, G, ME)
    with pytest.raises(CommandError, match="isn't yours"):
        cc.remove_sound(services, G, OTHER, "air-horn")
    cc.remove_sound(services, G, ME, "air-horn")
    assert store.list_sounds(G) == [] and store.list_reactions(G, kind="sound") == []
    with pytest.raises(CommandError):
        cc.find_sound(services, G, "air-horn")


def test_sound_volume(services, store):
    cc.add_sound(services, G, ME, "horn", wav(1))
    assert "100%" in cc.sound_volume(services, G, OTHER, "horn")  # anyone can look
    with pytest.raises(CommandError, match="isn't yours"):
        cc.sound_volume(services, G, OTHER, "horn", 50)
    assert "50%" in cc.sound_volume(services, G, ME, "horn", 50)
    assert store.find_sound(G, "horn")["gain_db"] == pytest.approx(-6.02, abs=0.01)
    assert "horn** (1.0s, 50%)" in cc.list_sounds(services, G, ME)
    with pytest.raises(CommandError, match="0 to 100%"):  # the server's cap
        cc.sound_volume(services, G, ME, "horn", 150)
    store.set_setting(G, "sounds.max_volume", 200)
    assert "150%" in cc.sound_volume(services, G, ADMIN, "horn", 150)  # admins may change anyone's


def test_sound_quota(services, store):
    store.set_setting(G, "quota.sounds", 1)
    cc.add_sound(services, G, ME, "uno", wav(1))
    with pytest.raises(QuotaReached):
        cc.add_sound(services, G, ME, "dos", wav(1))


# ------------------------------------------------------------ voices
@pytest.mark.parametrize("description, instruct", [
    ("mujer joven, voz grave", "female, young adult, low pitch"),
    ("hombre mayor con voz muy aguda", "male, elderly, very high pitch"),
    ("Male, British accent, whisper", "male, whisper, british accent"),
    ("anciana susurrando", "female, elderly, whisper"),
    ("niño con acento ruso", "male, child, russian accent"),
])
def test_describe_voice(description, instruct):
    assert cc.describe_voice(description) == instruct


@pytest.mark.parametrize("description", ["robot malvado", "", "hombre mujer", "grave y aguda"])
def test_describe_voice_errors(description):
    with pytest.raises(CommandError):
        cc.describe_voice(description)
    with pytest.raises(CommandError, match="these options"):
        cc.describe_voice("robot")


def test_create_designed_voice(services, store):
    text = cc.create_voice(services, G, ME, "Abuela", "anciana, voz grave")
    voice = store.find_voice(G, "abuela")
    assert "**Abuela**" in text and voice["kind"] == "designed" and voice["instruct"] == "female, elderly, low pitch"
    assert voice["created_by"] == ME.user_id and services.built == [voice["id"]]
    with pytest.raises(CommandError, match="already a voice"):
        cc.create_voice(services, G, OTHER, "abuela", "hombre")
    with pytest.raises(CommandError, match="these options"):
        cc.create_voice(services, G, OTHER, "robot", "robot malvado")
    services.on = False
    with pytest.raises(CommandError, match="voice is turned off"):
        cc.create_voice(services, G, OTHER, "otra", "hombre")


def test_voice_quota(services, store):
    store.set_setting(G, "quota.voices", 1)
    cc.create_voice(services, G, ME, "una", "hombre")
    with pytest.raises(QuotaReached):
        cc.create_voice(services, G, ME, "dos", "mujer")


def test_clone_voice(services, store):
    with pytest.raises(CommandError, match="permission"):
        cc.clone_voice(services, G, ME, "Narrador", wav(5), "sample.wav", False)
    text = cc.clone_voice(services, G, ME, "Narrador", wav(5), "sample.wav", True)
    voice = store.find_voice(G, "narrador")
    assert "🧬" in text and voice["status"] == "queued" and voice["kind"] == "clone"
    assert services.built == [voice["id"]] and (services.library.folder(voice["id"]) / "ref.wav").is_file()
    with pytest.raises(CommandError, match="couldn't use that audio"):
        cc.clone_voice(services, G, ME, "Ruido", b"not audio" * 100, "x.mp3", True)
    assert store.find_voice(G, "ruido") is None  # nothing left behind
    store.set_setting(G, "voices.max_mb", 0.001)
    with pytest.raises(CommandError, match="MB"):
        cc.clone_voice(services, G, ME, "Grande", wav(5), "x.wav", True)


def test_voice_choices_list_and_remove(services, store):
    ready = store.add_voice("narrator", guild_id=G, kind="designed", instruct="male", status="ready", created_by=ME.user_id)
    store.add_voice("draft", guild_id=G, kind="designed", instruct="male")
    bot_voice = store.add_voice("Robo", kind="clone", status="ready")
    store.set_setting(0, "voice.bot", bot_voice)
    speaker = store.add_voice("speaker 99", kind="speaker", owner_user_id=99, status="ready")

    values = [v for _, v in cc.voice_choices(services, G)]
    assert values[:2] == ["bot", "@speaker"] and str(ready) in values and str(speaker) not in values
    assert [v for _, v in cc.voice_choices(services, G, "narr")] == [str(ready)]
    assert cc.resolve_voice(services, G, None) is None and cc.resolve_voice(services, G, "bot") is None
    assert cc.resolve_voice(services, G, "Narrator") == ready
    with pytest.raises(CommandError):
        cc.resolve_voice(services, 2, str(ready))  # another server's voice

    listing = cc.list_voices(services, G)
    assert "**narrator**" in listing and "draft" not in listing and "<@99>" not in listing  # no consent
    store.set_consent(99, "accepted")
    assert "<@99>" in cc.list_voices(services, G)

    assert cc.voice_names(services, G, ME) == ["narrator"]
    with pytest.raises(CommandError, match="isn't yours"):
        cc.remove_voice(services, G, OTHER, "narrator")
    with pytest.raises(CommandError, match="bot's own"):
        cc.remove_voice(services, G, ADMIN, "Robo")
    with pytest.raises(CommandError, match="/voice delete"):
        cc.remove_voice(services, G, ADMIN, "speaker 99")
    assert "🗑️" in cc.remove_voice(services, G, ME, "narrator")
    assert store.get_voice(ready) is None


# ------------------------------------------------------------ the Discord glue loads
def test_cog_registers(services):
    import discord
    from discord.ext import commands

    async def go():
        bot = commands.Bot(command_prefix="!", intents=discord.Intents.none())
        cog = await cc.setup(bot, services)
        names = {c.qualified_name for c in bot.walk_commands()}
        assert {"gag add", "gag remove", "gag test", "sound add", "sound play", "voices create",
                "voices clone", "voices remove", "voices preview"} <= names
        assert {c.name for c in bot.tree.get_commands()} >= {"gag", "sound", "voices"}
        assert isinstance(cog, cc.ContentCommands)
        await bot.close()

    asyncio.run(go())


# ------------------------------------------------------------ languages
def test_replies_follow_the_server_language(services, store):
    store.set_setting(G, "language", "es")
    assert "Gag **buenas noches** creado" in cc.add_gag(services, G, ME, "buenas noches", "a")
    with pytest.raises(CommandError) as e:
        cc.remove_gag(services, G, OTHER, "buenas noches")
    assert e.value.render("es") == "❌ Eso no es tuyo." and str(e.value) == "❌ That isn't yours."
    store.set_setting(G, "quota.gags", 1)
    with pytest.raises(QuotaReached) as limit:
        cc.add_gag(services, G, ME, "otra cosa", "b")
    assert limit.value.render("es").startswith("🚫 Llegaste a tu límite de gags")
    with pytest.raises(CommandError) as bad:
        cc.add_sound(services, G, ADMIN, "!!!", wav(1))
    assert "nombre del sonido" in bad.value.render("es")
    assert "No saltaría nada" in cc.explain_text(services, G, ME, "nada que ver")
    assert store.get_setting(2, "language") is None
    assert "Nothing would go off" in cc.explain_text(services, 2, ME, "nothing")  # the default: English


def test_every_string_exists_in_every_language():
    """Each key the code uses is in locales/<lang>/commands.json for every language."""
    import json
    import re
    from pathlib import Path

    root = Path(cc.__file__).resolve().parent
    code = (root / "content_commands.py").read_text() + (root / "sound_library.py").read_text()
    used = set(re.findall(r'(?:i18n\.t\(|Error\()"([a-z_]+\.[a-z_. ]+)"', code))
    used |= set(re.findall(r'"((?:gag|sound|voice)\.not_found)"', code))
    used |= {f"quota.resource.{r}" for r in cc.RESOURCES} | {f"voice.trait.{c}" for c in cc.VOICE_TRAITS}
    used |= {f"voice.kind.{k}" for k in ("clone", "designed")}
    used |= {f"explain.reason.{r}" for r in ("cooldown", "not this user", "first match wins", "no usable option")}
    used |= {"quota.reached", "quota.request_saved"}
    assert len(used) > 60
    for lang in ("en", "es"):
        strings = json.loads((root / "locales" / lang / "commands.json").read_text(encoding="utf-8"))
        assert used <= set(strings), (lang, sorted(used - set(strings)))
