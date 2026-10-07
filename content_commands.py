"""What people make from Discord: their own gags (/gag), sounds (/sound) and
voices (/voices), within per-person quotas, plus the "request more" flow
that asks an admin.

Two layers:
  - plain functions (add_gag, add_sound, create_voice...) that take the
    services, the server, who's asking and the arguments, and return the
    reply text or raise CommandError. They hold all the rules and are what
    the tests exercise.
  - a thin discord.py Cog on top: parses options, opens the modal, shows
    buttons, sends the replies.

`services` is whatever the bot hands in; this module uses:
    store, engine, library (VoiceLibrary), sounds (SoundLibrary),
    voice_on() -> bool, queue_build(voice_id),
    async say_text(guild, text, voice_id), async play_sound(guild, sound_id)

Replies are short and emoji first, in the server's language (i18n.py; the
strings are in locales/*/commands.json). Command names and descriptions are
English.
"""
import asyncio
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

import i18n
from transcriber import normalize

log = logging.getLogger(__name__)

RESOURCES = ("gags", "sounds", "voices")
GRANT = 5  # what "Approve" adds
MAX_REPLIES = 10
MAX_REPLY_CHARS = 300
MAX_TRIGGERS = 10
MIN_TRIGGER_LETTERS = 5
# Words too common to set anything off on their own (English and Spanish).
STOPWORDS = frozenset("""
the a an and or of to in on at is are was be it its this that yes no so ok okay hey hi oh what who how why
you me my i we he she they not do just like but for with
que el la los las lo le les un una unos unas y e o u de del a al en con por para sin si no ya eso esto
esa ese esta este es son fue ser se me te mi tu su sus nos mas muy pero como cuando donde quien pues bueno
oye vale bien mal hola chao asi aqui alli ahi todo nada algo hay ha he va voy vas yo ella ellos
""".split())

VOICE_TRAITS: dict[str, dict[str, tuple[str, ...]]] = {
    # category -> OmniVoice item -> how people may say it (English and Spanish)
    "gender": {
        "male": ("male", "man", "hombre", "masculino", "masculina", "varon", "chico"),
        "female": ("female", "woman", "mujer", "femenino", "femenina", "chica"),
    },
    "age": {
        "child": ("child", "kid", "nino", "nina", "infantil"),
        "teenager": ("teenager", "teen", "adolescente"),
        "young adult": ("young adult", "young", "joven", "adulto joven"),
        "middle-aged": ("middle-aged", "middle aged", "mediana edad", "adulto", "adulta"),
        "elderly": ("elderly", "old", "anciano", "anciana", "viejo", "vieja", "mayor", "abuelo", "abuela"),
    },
    "pitch": {
        "very low pitch": ("very low pitch", "very low", "very deep", "muy grave", "muy profunda", "muy profundo"),
        "low pitch": ("low pitch", "low", "deep", "grave", "profunda", "profundo"),
        "moderate pitch": ("moderate pitch", "moderate", "normal", "media", "medio"),
        "high pitch": ("high pitch", "high", "agudo", "aguda"),
        "very high pitch": ("very high pitch", "very high", "muy agudo", "muy aguda"),
    },
    "style": {
        "whisper": ("whisper", "whispering", "whispered", "susurro", "susurrando", "susurrada", "susurrado"),
    },
    "accent": {
        f"{country} accent": (f"{country} accent", f"{country}", f"acento {spanish}", spanish)
        for country, spanish in (("american", "americano"), ("british", "britanico"), ("australian", "australiano"),
                                 ("chinese", "chino"), ("canadian", "canadiense"), ("indian", "indio"),
                                 ("korean", "coreano"), ("portuguese", "portugues"), ("russian", "ruso"),
                                 ("japanese", "japones"))
    },
}
# Spanish age words that say the gender too ("anciana": an elderly woman).
GENDERED = {"anciana": "female", "anciano": "male", "vieja": "female", "viejo": "male", "abuela": "female",
            "abuelo": "male", "nina": "female", "nino": "male", "adulta": "female", "adulto": "male"}
# Words that may sit around the traits without meaning anything.
DESCRIPTION_FILLER = frozenset("voz voice de del con with a an un una y and e of tono pitch tone acento accent "
                               "que sounds suena como like person persona".split())


@dataclass
class Actor:
    """Who's running a command."""
    user_id: int
    name: str = ""
    is_admin: bool = False


class CommandError(Exception):
    """Something to tell the person instead of doing what they asked: a key
    in locales/*/commands.json and its values. str() is the English text."""

    def __init__(self, key: str, **values):
        self.key, self.values = key, values
        super().__init__(i18n.t(key, "en", **values))

    def render(self, lang: str | None) -> str:
        return i18n.t(self.key, lang, **self.values)


class QuotaReached(CommandError):
    def __init__(self, resource: str, used: int, limit: int, lang: str | None = None):
        self.resource, self.used, self.limit = resource, used, limit
        super().__init__("quota.reached", resource=i18n.t(f"quota.resource.{resource}", lang), used=used,
                         limit=limit)


def language(services, guild_id: int | None) -> str:
    return i18n.guild_language(services.store, guild_id)


# ───────────────────────────── rules ─────────────────────────────

def is_admin(store, guild_id: int | None, *, is_owner: bool, manage_guild: bool, role_ids=()) -> bool:
    """The bot owner, anyone with Manage Server, or the admin.role_id role."""
    if is_owner or manage_guild:
        return True
    role = store.get_setting(guild_id, "admin.role_id") if guild_id else None
    return role is not None and int(role) in {int(r) for r in role_ids}


def check_quota(services, guild_id: int, actor: Actor, resource: str) -> tuple[int, int]:
    """(used, limit). Admins aren't limited."""
    used, limit = services.store.quota(guild_id, actor.user_id, resource)
    if not actor.is_admin and used >= limit:
        raise QuotaReached(resource, used, limit, language(services, guild_id))
    return used, limit


def _status(services, guild_id: int, actor: Actor) -> str:
    needs = services.store.get_setting(guild_id, "user_content.needs_approval")
    return "pending" if needs and not actor.is_admin else "approved"


def _quota(services, guild_id: int, actor: Actor, resource: str) -> dict:
    used, limit = services.store.quota(guild_id, actor.user_id, resource)
    return {"used": used, "limit": limit}


def split_list(text: str | None, separators: str = ",\n") -> list[str]:
    """'a, b\nc' -> ['a', 'b', 'c']: trimmed, no blanks, no repeats."""
    items = re.split("[" + re.escape(separators) + "]", text or "")
    out = []
    for item in (i.strip() for i in items):
        if item and item.casefold() not in {o.casefold() for o in out}:
            out.append(item)
    return out


def trigger_problem(phrase: str, lang: str | None = None) -> str | None:
    """Why a trigger would fire far too often, or None if it's fine."""
    words = normalize(phrase).split()
    if not words:
        return i18n.t("trigger.empty", lang)
    if all(w in STOPWORDS for w in words):
        return i18n.t("trigger.too_common", lang)
    if sum(len(w) for w in words) < MIN_TRIGGER_LETTERS:
        return i18n.t("trigger.too_short", lang, letters=MIN_TRIGGER_LETTERS)
    return None


def check_triggers(phrases: list[str], actor: Actor, lang: str | None = None) -> list[str]:
    """Raise on triggers that are too broad (admins only get a warning). Returns warnings."""
    if not phrases:
        raise CommandError("trigger.need_one")
    if len(phrases) > MAX_TRIGGERS:
        raise CommandError("trigger.too_many", count=MAX_TRIGGERS)
    problems = [f"“{p}”: {why}" for p in phrases if (why := trigger_problem(p, lang))]
    if problems and not actor.is_admin:
        raise CommandError("trigger.too_broad", problems="; ".join(problems))
    return [f"⚠️ {p}" for p in problems]


def overlap_warnings(services, guild_id: int, phrases: list[str], *, skip_id: int | None = None) -> list[str]:
    """Other reactions whose phrases contain these, or the other way round
    (the first one in the list wins, so one may shadow the other)."""
    lang = language(services, guild_id)
    warnings = []
    wanted = [normalize(p) for p in phrases]
    for row in services.store.list_reactions(guild_id):
        if row["id"] == skip_id or row["status"] == "rejected":
            continue
        theirs = [normalize(p) for t in row["triggers"] if t["type"] == "phrase" for p in t["phrases"]]
        for mine in wanted:
            if any(re.search(rf"\b{re.escape(mine)}\b", other) or re.search(rf"\b{re.escape(other)}\b", mine)
                   for other in theirs if other):
                warnings.append(i18n.t("trigger.overlap", lang, name=row["name"]))
                break
        if len(warnings) >= 3:
            break
    return warnings


def unique_name(existing: set[str], base: str) -> str:
    taken = {n.casefold() for n in existing}
    name, n = base, 2
    while name.casefold() in taken:
        name, n = f"{base} {n}", n + 1
    return name


def fit(rows: list[str], limit: int = 1900) -> str:
    """As many rows as fit in one Discord message."""
    out, size = [], 0
    for row in rows:
        if size + len(row) + 1 > limit:
            out.append(f"… +{len(rows) - len(out)}")
            break
        out.append(row)
        size += len(row) + 1
    return "\n".join(out)


def _with_status(text: str, status: str, lang: str) -> str:
    return text + i18n.t("common.pending", lang) if status == "pending" else text


# ───────────────────────────── voices (shared) ─────────────────────────────

SPEAKER_CHOICE = "@speaker"
BOT_CHOICE = "bot"


def voice_choices(services, guild_id: int, current: str = "") -> list[tuple[str, str]]:
    """(label, value) for a voice option: the bot's, your own, then the
    server's and global ready voices. At most 25 (Discord's limit)."""
    lang = language(services, guild_id)
    out = [(i18n.t("voice.choice_bot", lang), BOT_CHOICE), (i18n.t("voice.choice_own", lang), SPEAKER_CHOICE)]
    for v in services.store.list_voices(guild_id):
        if v["kind"] == "speaker" or v["status"] != "ready":
            continue
        out.append((f"{v['name']} ({i18n.t('voice.kind.' + v['kind'], lang)})", str(v["id"])))
    current = normalize(current)
    return [c for c in out if current in normalize(c[0])][:25]


def resolve_voice(services, guild_id: int, value: str | None):
    """An option value or a typed name -> None (the bot's voice), "@speaker" or a voice id."""
    if value is None or value.strip() == "" or value.strip().casefold() == BOT_CHOICE:
        return None
    value = value.strip()
    if value == SPEAKER_CHOICE:
        return SPEAKER_CHOICE
    row = services.store.get_voice(int(value)) if value.isdigit() else services.store.find_voice(guild_id, value)
    if row is None or row["guild_id"] not in (None, guild_id) or row["kind"] == "speaker":
        raise CommandError("voice.unknown", name=value)
    if row["status"] != "ready":
        raise CommandError("voice.not_ready", name=row["name"])
    return row["id"]


class VoiceDescriptionError(CommandError):
    pass


def describe_voice(description: str, lang: str | None = None) -> str:
    """'young woman, deep voice' / 'mujer joven, voz grave' -> 'female, young
    adult, low pitch': an OmniVoice instruct. VoiceDescriptionError (a
    ValueError-like CommandError listing what's possible) otherwise."""
    text = f" {normalize(description)} "
    found: dict[str, str] = {}
    implied: set[str] = set()
    synonyms = sorted(((syn, category, item) for category, items in VOICE_TRAITS.items()
                       for item, syns in items.items() for syn in syns), key=lambda s: -len(s[0]))
    for syn, category, item in synonyms:
        pattern = f" {normalize(syn)} "
        if pattern in text:
            if category in found and found[category] != item:
                raise VoiceDescriptionError("voice.description.conflict", category=i18n.t(f"voice.trait.{category}", lang),
                                            first=found[category], second=item)
            found[category] = item
            text = text.replace(pattern, " ")
            if normalize(syn) in GENDERED:
                implied.add(GENDERED[normalize(syn)])
    if "gender" not in found and len(implied) == 1:
        found["gender"] = implied.pop()
    leftover = [w for w in text.split() if w not in DESCRIPTION_FILLER]
    if leftover or not found:
        options = "; ".join(f"{i18n.t(f'voice.trait.{cat}', lang)}: {', '.join(items)}"
                            for cat, items in VOICE_TRAITS.items())
        raise VoiceDescriptionError("voice.description.unknown", words=", ".join(leftover) or "-", options=options)
    return ", ".join(found[c] for c in VOICE_TRAITS if c in found)


def _check_voice_name(services, guild_id: int, name: str) -> str:
    name = " ".join(name.split())
    if not 1 <= len(name) <= 32 or name in (BOT_CHOICE, SPEAKER_CHOICE) or name.isdigit():
        raise CommandError("voice.bad_name")
    if services.store.find_voice(guild_id, name):
        raise CommandError("voice.taken", name=name)
    return name


# ───────────────────────────── gags ─────────────────────────────

def add_gag(services, guild_id: int, actor: Actor, triggers: str, replies: str, *, only_me: bool = False,
            voice: str | None = None) -> str:
    """Create a gag: any of `triggers` (comma separated) -> one of `replies`
    (one per line) at random."""
    lang = language(services, guild_id)
    check_quota(services, guild_id, actor, "gags")
    phrases = split_list(triggers)
    warnings = check_triggers(phrases, actor, lang)
    lines = split_list(replies, "\n")
    if not lines:
        raise CommandError("gag.need_reply")
    if len(lines) > MAX_REPLIES or any(len(line) > MAX_REPLY_CHARS for line in lines):
        raise CommandError("gag.reply_limits", count=MAX_REPLIES, chars=MAX_REPLY_CHARS)
    voice_id = resolve_voice(services, guild_id, voice)
    store = services.store
    name = unique_name({r["name"] for r in store.list_reactions(guild_id, kind="gag")}, phrases[0][:40])
    status = _status(services, guild_id, actor)
    warnings += overlap_warnings(services, guild_id, phrases)
    try:
        store.add_reaction(guild_id, "gag", name, [{"type": "phrase", "phrases": phrases}],
                           [[{"type": "say", "text": line}] for line in lines], status=status,
                           created_by=actor.user_id, voice_id=voice_id,
                           by_users=[actor.user_id] if only_me else None)
    except ValueError as e:
        raise CommandError("common.invalid", error=str(e)) from None
    text = i18n.t("gag.added", lang, name=name, **_quota(services, guild_id, actor, "gags"))
    return "\n".join([_with_status(text, status, lang), *warnings])


def _find_own(rows: list[dict], name: str, actor: Actor, not_found: str) -> dict:
    wanted = name.strip().casefold()
    matches = [r for r in rows if r["name"].casefold() == wanted]
    if not matches:
        raise CommandError(not_found, name=name)
    mine = [r for r in matches if r.get("created_by") == actor.user_id]
    if mine:
        return mine[0]
    if actor.is_admin:
        return matches[0]
    raise CommandError("common.not_yours")


def remove_gag(services, guild_id: int, actor: Actor, name: str) -> str:
    row = _find_own(services.store.list_reactions(guild_id, kind="gag"), name, actor, "gag.not_found")
    services.store.delete_reaction(row["id"])
    return i18n.t("gag.removed", language(services, guild_id), name=row["name"])


def own_names(rows: list[dict], actor: Actor, current: str = "") -> list[str]:
    """Names for a remove command's autocomplete: yours (admins: all)."""
    current = current.casefold()
    names = [r["name"] for r in rows if actor.is_admin or r.get("created_by") == actor.user_id]
    return [n for n in dict.fromkeys(names) if current in n.casefold()][:25]


def list_gags(services, guild_id: int, actor: Actor) -> str:
    lang = language(services, guild_id)
    rows = []
    for r in services.store.list_reactions(guild_id, kind="gag"):
        if r["status"] == "rejected":
            continue
        phrases = [p for t in r["triggers"] if t["type"] == "phrase" for p in t["phrases"]]
        phrases += [f"~{t['word']}" for t in r["triggers"] if t["type"] == "swap"]
        says = [s["text"] for o in r["options"] for s in o if s["type"] == "say"]
        marks = ("★ " if r.get("created_by") == actor.user_id else "") + \
                ("⏳ " if r["status"] == "pending" else "") + ("💤 " if not r["enabled"] else "")
        reply = says[0] if says else "…"
        more = f" (+{len(says) - 1})" if len(says) > 1 else ""
        rows.append(f"• {marks}**{r['name']}**: {', '.join(phrases)[:60]} → {reply[:60]}{more}")
    if not rows:
        return i18n.t("gag.none", lang)
    return fit([i18n.t("gag.list_header", lang, **_quota(services, guild_id, actor, "gags")), *rows])


def explain_text(services, guild_id: int, actor: Actor, text: str) -> str:
    """What would happen if this person said `text`, and why."""
    lang = language(services, guild_id)
    name = services.store.nickname_for(guild_id, actor.user_id, actor.name)
    rows = services.engine.explain(guild_id, text, actor.user_id, actor.name, name)
    lines = []
    for r in rows:
        if r.get("near_miss"):
            lines.append(i18n.t("explain.near_miss", lang, heard=r["heard"], trigger=r["trigger_word"],
                                name=r["name"]))
        elif r["would_fire"]:
            chance = i18n.t("explain.chance", lang, percent=f"{r['chance']:.0%}") if r["chance"] < 1 else ""
            lines.append(i18n.t("explain.would_fire", lang, name=r["name"], chance=chance))
        else:
            icon = {"cooldown": "⏳", "not this user": "🚫", "first match wins": "↪️"}.get(r["reason"], "▫️")
            reason = i18n.t(f"explain.reason.{r['reason']}", lang)
            lines.append(f"{icon} {r['name']}: {reason}")
    return fit(lines) if lines else i18n.t("explain.nothing", lang)


# ───────────────────────────── sounds ─────────────────────────────

def _sound_error(e: ValueError) -> CommandError:
    key = getattr(e, "key", None)
    return CommandError(key, **e.values) if key else CommandError("common.invalid", error=str(e))


def add_sound(services, guild_id: int, actor: Actor, name: str, data: bytes, filename: str = "", *,
              triggers: str | None = None, command: str | None = None) -> str:
    """A new sound, playable with /sound play <name>, and also when someone
    says one of `triggers`, or `command` after the wake word."""
    lang = language(services, guild_id)
    check_quota(services, guild_id, actor, "sounds")
    phrases = split_list(triggers)
    warnings = check_triggers(phrases, actor, lang) if phrases else []
    commands_ = split_list(command)
    if any(not normalize(c) for c in commands_):
        raise CommandError("sound.command_empty")
    taken = {normalize(p) for r in services.store.list_reactions(guild_id) for t in r["triggers"]
             if t["type"] == "command" for p in t["phrases"]}
    clash = [c for c in commands_ if normalize(c) in taken]
    if clash:
        raise CommandError("sound.command_taken", phrases=", ".join(clash))
    try:
        sound = services.sounds.add(guild_id, name, data, created_by=actor.user_id, filename=filename)
    except ValueError as e:
        raise _sound_error(e) from None
    triggers_ = [{"type": "slash", "name": sound["name"]}]
    if phrases:
        triggers_.append({"type": "phrase", "phrases": phrases})
        warnings += overlap_warnings(services, guild_id, phrases)
    if commands_:
        triggers_.append({"type": "command", "phrases": commands_})
    status = _status(services, guild_id, actor)
    try:
        services.store.add_reaction(guild_id, "sound", sound["name"], triggers_,
                                    [[{"type": "sound", "sound_id": sound["id"]}]],
                                    status=status, created_by=actor.user_id)
    except ValueError as e:
        services.sounds.delete(sound["id"])
        raise CommandError("common.invalid", error=str(e)) from None
    text = i18n.t("sound.added", lang, name=sound["name"], seconds=f"{sound['duration_s']:.1f}",
                  **_quota(services, guild_id, actor, "sounds"))
    return "\n".join([_with_status(text, status, lang), *warnings])


def find_sound(services, guild_id: int, name: str) -> dict:
    from sound_library import sound_name

    try:
        row = services.store.find_sound(guild_id, sound_name(name))
    except ValueError:
        row = None
    if row is None or not row["enabled"]:
        raise CommandError("sound.not_found", name=name)
    return row


def remove_sound(services, guild_id: int, actor: Actor, name: str) -> str:
    row = find_sound(services, guild_id, name)
    _find_own([row], row["name"], actor, "sound.not_found")  # permission check
    for r in services.store.list_reactions(guild_id, kind="sound"):
        steps = [s for o in r["options"] for s in o]
        if steps and all(s["type"] == "sound" and s["sound_id"] == row["id"] for s in steps):
            services.store.delete_reaction(r["id"])  # it only played this sound
    services.sounds.delete(row["id"])
    return i18n.t("sound.removed", language(services, guild_id), name=row["name"])


def sound_volume(services, guild_id: int, actor: Actor, name: str, percent: float | None = None) -> str:
    """Show a sound's volume, or set it (its creator or an admin). 100% is
    the level every sound is evened out to; the server's sounds.max_volume
    is as loud as any may go."""
    from sound_library import db_to_volume, volume_to_db

    lang = language(services, guild_id)
    row = find_sound(services, guild_id, name)
    cap = float(services.store.get_setting(guild_id, "sounds.max_volume"))
    if percent is None:
        return i18n.t("sound.volume_is", lang, name=row["name"], volume=db_to_volume(row["gain_db"]), max=f"{cap:g}")
    _find_own([row], row["name"], actor, "sound.not_found")  # permission check
    if not 0 <= float(percent) <= cap:
        raise CommandError("sound.volume_range", max=f"{cap:g}")
    services.store.update_sound(row["id"], gain_db=volume_to_db(percent))
    return i18n.t("sound.volume_set", lang, name=row["name"], volume=f"{float(percent):g}")


def list_sounds(services, guild_id: int, actor: Actor) -> str:
    from sound_library import db_to_volume

    lang = language(services, guild_id)
    rows = []
    for s in services.store.list_sounds(guild_id):
        mine = "★ " if s["created_by"] == actor.user_id else ""
        volume = db_to_volume(s["gain_db"])
        rows.append(f"• {mine}**{s['name']}** ({s['duration_s'] or 0:.1f}s{'' if volume == 100 else f', {volume}%'})")
    if not rows:
        return i18n.t("sound.none", lang)
    return fit([i18n.t("sound.list_header", lang, **_quota(services, guild_id, actor, "sounds")), *rows])


def sound_names(services, guild_id: int, current: str = "") -> list[str]:
    current = current.casefold()
    return [s["name"] for s in services.store.list_sounds(guild_id) if current in s["name"]][:25]


# ───────────────────────────── voices ─────────────────────────────

def list_voices(services, guild_id: int) -> str:
    lang = language(services, guild_id)
    rows = []
    store = services.store
    for v in store.list_voices(guild_id):
        if v["status"] != "ready":
            continue
        if v["kind"] == "speaker":
            if not store.has_consent(v["owner_user_id"], "voice"):
                continue
            rows.append(i18n.t("voice.own_entry", lang, user=f"<@{v['owner_user_id']}>"))
        else:
            kind = i18n.t(f"voice.kind.{v['kind']}", lang)
            where = "" if v["guild_id"] is not None else f" · {i18n.t('voice.global', lang)}"
            rows.append(f"• **{v['name']}** ({kind}{where})")
    return fit([i18n.t("voice.list_header", lang), *rows]) if rows else i18n.t("voice.none", lang)


def create_voice(services, guild_id: int, actor: Actor, name: str, description: str) -> str:
    """A designed voice: OmniVoice makes it up from a description (no audio)."""
    lang = language(services, guild_id)
    if not services.voice_on():
        raise CommandError("common.voice_off")
    check_quota(services, guild_id, actor, "voices")
    name = _check_voice_name(services, guild_id, name)
    instruct = describe_voice(description, lang)
    voice_id = services.store.add_voice(name, guild_id=guild_id, kind="designed", instruct=instruct,
                                        created_by=actor.user_id, status="queued")
    services.queue_build(voice_id)
    return i18n.t("voice.created", lang, name=name, instruct=instruct, **_quota(services, guild_id, actor, "voices"))


def clone_voice(services, guild_id: int, actor: Actor, name: str, data: bytes, filename: str,
                confirm: bool) -> str:
    """A voice cloned from an uploaded sample. Blocking (it decodes and
    picks the speech on the CPU); the build itself goes to the TTS queue."""
    lang = language(services, guild_id)
    if not confirm:
        raise CommandError("voice.need_permission")
    if not services.voice_on():
        raise CommandError("common.voice_off")
    check_quota(services, guild_id, actor, "voices")
    name = _check_voice_name(services, guild_id, name)
    max_mb = float(services.store.get_setting(guild_id, "voices.max_mb"))
    if len(data) > max_mb * 1024 * 1024:
        raise CommandError("common.file_too_big", mb=f"{max_mb:g}")
    voice_id = services.store.add_voice(name, guild_id=guild_id, kind="clone", created_by=actor.user_id,
                                        status="draft")
    suffix = Path(filename or "sample").suffix.lower()[:8] or ".audio"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / f"sample{suffix}"
            sample.write_bytes(data)
            services.library.ingest(voice_id, sample)
    except Exception as e:
        services.library.delete_voice(voice_id)
        log.info("Clone sample for %r rejected: %s", name, e)
        raise CommandError("voice.bad_audio") from None
    services.store.update_voice(voice_id, status="queued")
    services.queue_build(voice_id)
    return i18n.t("voice.cloning", lang, name=name, **_quota(services, guild_id, actor, "voices"))


def remove_voice(services, guild_id: int, actor: Actor, name: str) -> str:
    store = services.store
    row = store.find_voice(guild_id, name.strip())
    if row is None:
        raise CommandError("voice.not_found", name=name)
    if row["kind"] == "speaker":
        raise CommandError("voice.own_use_voice_delete")
    if row["id"] == store.get_setting(0, "voice.bot"):
        raise CommandError("voice.is_bot")
    if row["guild_id"] is None and not actor.is_admin:
        raise CommandError("voice.is_global")
    _find_own([row], row["name"], actor, "voice.not_found")
    services.library.delete_voice(row["id"])
    return i18n.t("voice.removed", language(services, guild_id), name=row["name"])


def voice_names(services, guild_id: int, actor: Actor, current: str = "") -> list[str]:
    rows = [v for v in services.store.list_voices(guild_id, include_global=actor.is_admin)
            if v["kind"] != "speaker" and v["id"] != services.store.get_setting(0, "voice.bot")]
    return own_names(rows, actor, current)


# ───────────────────────────── quota requests ─────────────────────────────

def file_quota_request(services, guild_id: int, user_id: int, resource: str) -> int:
    if resource not in RESOURCES:
        raise CommandError("common.invalid", error=resource)
    return services.store.request_quota(guild_id, user_id, resource, GRANT, "from Discord")


def request_text(services, request: dict, guild_name: str = "") -> str:
    lang = language(services, request["guild_id"])
    used, limit = services.store.quota(request["guild_id"], request["user_id"], request["resource"])
    return i18n.t("quota.request_text", lang, user=f"<@{request['user_id']}>", guild=guild_name or "?",
                  resource=i18n.t(f"quota.resource.{request['resource']}", lang), used=used, limit=limit)


def decide_request(services, request_id: int, approve: bool, actor: Actor) -> tuple[str, dict]:
    """(text for the admin, the updated request)."""
    if not actor.is_admin:
        raise CommandError("quota.admins_only")
    try:
        request = services.store.decide_quota_request(request_id, approve, actor.user_id,
                                                      amount=GRANT if approve else None)
    except (KeyError, ValueError):
        raise CommandError("quota.already_decided") from None
    lang = language(services, request["guild_id"])
    resource = i18n.t(f"quota.resource.{request['resource']}", lang)
    if approve:
        _, limit = services.store.quota(request["guild_id"], request["user_id"], request["resource"])
        return i18n.t("quota.approved", lang, amount=GRANT, resource=resource, limit=limit), request
    return i18n.t("quota.denied", lang), request


def decision_text(services, request: dict) -> str:
    """What the requester is told."""
    lang = language(services, request["guild_id"])
    resource = i18n.t(f"quota.resource.{request['resource']}", lang)
    if request["status"] == "approved":
        return i18n.t("quota.you_got_more", lang, amount=request["amount"], resource=resource)
    return i18n.t("quota.you_were_denied", lang, resource=resource)


# ───────────────────────────── Discord glue ─────────────────────────────

_services = None  # set by setup(): the buttons below are rebuilt from their ids and need it
_bot = None


async def actor_for(bot, services, user, guild) -> Actor:
    member = guild.get_member(user.id) if guild is not None else None
    perms = member.guild_permissions if member is not None else None
    return Actor(
        user_id=user.id,
        name=getattr(user, "display_name", user.name),
        is_admin=is_admin(services.store, guild.id if guild else None, is_owner=await bot.is_owner(user),
                          manage_guild=bool(perms and perms.manage_guild),
                          role_ids=[r.id for r in member.roles] if member is not None else ()),
    )


async def send(ctx, text: str, view: discord.ui.View | None = None) -> None:
    kwargs = {"allowed_mentions": discord.AllowedMentions.none()}
    if view is not None:
        kwargs["view"] = view
    await ctx.send(text, ephemeral=True, **kwargs)


async def defer(ctx) -> None:
    """Buy time for slow work (once: a second defer is an error)."""
    if ctx.interaction is None or not ctx.interaction.response.is_done():
        await ctx.defer(ephemeral=True)


def request_view(guild_id: int, user_id: int, resource: str) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(QuotaRequestButton(guild_id, user_id, resource))
    return view


def decision_view(request_id: int, lang: str | None = None) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(QuotaDecisionButton("ok", request_id, lang))
    view.add_item(QuotaDecisionButton("no", request_id, lang))
    return view


async def notify_admins(bot, services, guild_id: int, text: str, view: discord.ui.View) -> bool:
    """Post in admin.channel_id if set, otherwise DM the bot owner(s)."""
    channel_id = services.store.get_setting(guild_id, "admin.channel_id")
    if channel_id:
        channel = bot.get_channel(int(channel_id))
        if channel is not None:
            try:
                await channel.send(text, view=view, allowed_mentions=discord.AllowedMentions.none())
                return True
            except discord.HTTPException as e:
                log.warning("Couldn't post a quota request in %s: %s", channel, e)
    app = await bot.application_info()
    people = [m for m in app.team.members] if app.team else [app.owner]
    sent = False
    for person in people:
        try:
            await person.send(text, view=view, allowed_mentions=discord.AllowedMentions.none())
            sent = True
        except discord.HTTPException:
            pass
    return sent


class QuotaRequestButton(discord.ui.DynamicItem[discord.ui.Button],
                         template=r"quota:ask:(?P<guild>\d+):(?P<user>\d+):(?P<resource>gags|sounds|voices)"):
    """"Request more" under a limit-reached reply. Keeps working after a restart."""

    def __init__(self, guild_id: int, user_id: int, resource: str):
        lang = language(_services, guild_id) if _services is not None else None
        super().__init__(discord.ui.Button(label=i18n.t("quota.request_button", lang), emoji="📨",
                                           style=discord.ButtonStyle.primary,
                                           custom_id=f"quota:ask:{guild_id}:{user_id}:{resource}"))
        self.guild_id, self.user_id, self.resource = guild_id, user_id, resource

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(int(match["guild"]), int(match["user"]), match["resource"])

    async def callback(self, interaction: discord.Interaction) -> None:
        lang = language(_services, self.guild_id)
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message(i18n.t("common.not_yours", lang), ephemeral=True)
        request_id = file_quota_request(_services, self.guild_id, self.user_id, self.resource)
        request = _services.store.get_quota_request(request_id)
        guild = _bot.get_guild(self.guild_id)
        sent = await notify_admins(_bot, _services, self.guild_id,
                                   request_text(_services, request, guild.name if guild else ""),
                                   decision_view(request_id, lang))
        answer = i18n.t("quota.request_sent" if sent else "quota.request_saved", lang)
        await interaction.response.edit_message(content=answer, view=None)


class QuotaDecisionButton(discord.ui.DynamicItem[discord.ui.Button],
                          template=r"quota:(?P<choice>ok|no):(?P<request>\d+)"):
    """Approve +5 / Deny, for admins."""

    def __init__(self, choice: str, request_id: int, lang: str | None = None):
        ok = choice == "ok"
        super().__init__(discord.ui.Button(
            label=i18n.t("quota.approve_button", lang, amount=GRANT) if ok else i18n.t("quota.deny_button", lang),
            style=discord.ButtonStyle.success if ok else discord.ButtonStyle.danger,
            custom_id=f"quota:{choice}:{request_id}"))
        self.choice, self.request_id = choice, request_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["choice"], int(match["request"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        request = _services.store.get_quota_request(self.request_id)
        if request is None:
            return await interaction.response.edit_message(content=i18n.t("quota.gone"), view=None)
        actor = await actor_for(_bot, _services, interaction.user, _bot.get_guild(request["guild_id"]))
        try:
            text, request = decide_request(_services, self.request_id, self.choice == "ok", actor)
        except CommandError as e:
            lang = language(_services, request["guild_id"])
            return await interaction.response.send_message(e.render(lang), ephemeral=True)
        content = f"{interaction.message.content if interaction.message else ''}\n{text} ({interaction.user})"
        await interaction.response.edit_message(content=content[-2000:], view=None)
        try:
            user = _bot.get_user(request["user_id"]) or await _bot.fetch_user(request["user_id"])
            await user.send(decision_text(_services, request))
        except discord.HTTPException:
            pass  # DMs closed: they'll see their new limit next time


class GagModal(discord.ui.Modal):
    """/gag add without text: a form for the triggers and the replies."""

    def __init__(self, cog: "ContentCommands", lang: str, voice: str | None, only_me: bool):
        super().__init__(title=i18n.t("gag.modal_title", lang))
        self.cog, self.voice, self.only_me = cog, voice, only_me
        self.triggers = discord.ui.TextInput(label=i18n.t("gag.modal_triggers", lang),
                                             placeholder=i18n.t("gag.modal_triggers_hint", lang), max_length=300)
        self.replies = discord.ui.TextInput(label=i18n.t("gag.modal_replies", lang),
                                            style=discord.TextStyle.paragraph, max_length=2000,
                                            placeholder=i18n.t("gag.modal_replies_hint", lang))
        self.add_item(self.triggers)
        self.add_item(self.replies)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        actor = await actor_for(self.cog.bot, self.cog.services, interaction.user, interaction.guild)
        text, view = self.cog.run(add_gag, interaction.guild.id, actor, self.triggers.value, self.replies.value,
                                  only_me=self.only_me, voice=self.voice)
        kwargs = {"view": view} if view else {}
        await interaction.response.send_message(text, ephemeral=True, **kwargs)


class ContentCommands(commands.Cog):
    def __init__(self, bot: commands.Bot, services):
        self.bot, self.services = bot, services

    def run(self, fn, guild_id: int, actor: Actor, *args, **kwargs) -> tuple[str, discord.ui.View | None]:
        """Call a command body; turn its errors into a reply (+ the request button at a limit)."""
        lang = language(self.services, guild_id)
        try:
            return fn(self.services, guild_id, actor, *args, **kwargs), None
        except QuotaReached as e:
            return e.render(lang), request_view(guild_id, actor.user_id, e.resource)
        except CommandError as e:
            return e.render(lang), None

    async def run_ctx(self, ctx, fn, *args, threaded: bool = False, **kwargs) -> None:
        if ctx.guild is None:
            return await send(ctx, i18n.t("common.server_only"))
        actor = await actor_for(self.bot, self.services, ctx.author, ctx.guild)
        if threaded:
            await defer(ctx)
            text, view = await asyncio.to_thread(self.run, fn, ctx.guild.id, actor, *args, **kwargs)
        else:
            text, view = self.run(fn, ctx.guild.id, actor, *args, **kwargs)
        await send(ctx, text, view)

    async def _actor(self, interaction) -> Actor:
        return await actor_for(self.bot, self.services, interaction.user, interaction.guild)

    async def _voice_autocomplete(self, interaction, current: str):
        return [app_commands.Choice(name=label, value=value)
                for label, value in voice_choices(self.services, interaction.guild_id, current)]

    # ------------------------------------------------------------ /gag
    @commands.hybrid_group(name="gag", fallback="list", description="Your gags (lists this server's)")
    async def gag(self, ctx):
        await self.run_ctx(ctx, list_gags)

    @gag.command(name="add", description="Make a gag: when someone says something, the bot answers")
    @app_commands.describe(voice="The voice it answers in", only_me="Only when I say it",
                           text="Quick form: 'phrase, another = reply | another reply' (skips the form)")
    async def gag_add(self, ctx, *, text: str | None = None, voice: str | None = None, only_me: bool = False):
        # Prefix commands: `text` takes the rest of the line (the options after it keep their defaults).
        if text is None:
            lang = language(self.services, ctx.guild.id if ctx.guild else None)
            if ctx.interaction is None:
                return await send(ctx, i18n.t("gag.usage", lang))
            return await ctx.interaction.response.send_modal(GagModal(self, lang, voice, only_me))
        triggers, _, replies = text.partition("=")
        await self.run_ctx(ctx, add_gag, triggers, "\n".join(replies.split("|")), only_me=only_me, voice=voice)

    @gag_add.autocomplete("voice")
    async def _gag_voice(self, interaction, current: str):
        return await self._voice_autocomplete(interaction, current)

    @gag.command(name="remove", description="Remove one of your gags")
    async def gag_remove(self, ctx, *, name: str):
        await self.run_ctx(ctx, remove_gag, name)

    @gag_remove.autocomplete("name")
    async def _gag_names(self, interaction, current: str):
        actor = await self._actor(interaction)
        rows = self.services.store.list_reactions(interaction.guild_id, kind="gag")
        return [app_commands.Choice(name=n, value=n) for n in own_names(rows, actor, current)]

    @gag.command(name="test", description="What would this sentence set off, and why")
    async def gag_test(self, ctx, *, text: str):
        await self.run_ctx(ctx, explain_text, text)

    # ------------------------------------------------------------ /sound
    @commands.hybrid_group(name="sound", fallback="list", description="Sounds (lists this server's)")
    async def sound(self, ctx):
        await self.run_ctx(ctx, list_sounds)

    @sound.command(name="add", description="Upload a sound")
    @app_commands.describe(name="A short name", file="Audio (mp3, ogg, wav...)",
                           triggers="Phrases that play it when someone says them (a, b)",
                           command="A voice command that plays it (said after the bot's name)")
    async def sound_add(self, ctx, name: str, file: discord.Attachment, triggers: str | None = None,
                        command: str | None = None):
        gid = ctx.guild.id if ctx.guild else 0
        max_mb = float(self.services.store.get_setting(gid, "sounds.max_mb"))
        if file.size > max_mb * 1024 * 1024:
            return await send(ctx, i18n.t("common.file_too_big", language(self.services, gid), mb=f"{max_mb:g}"))
        await defer(ctx)
        data = await file.read()
        await self.run_ctx(ctx, add_sound, name, data, file.filename, triggers=triggers, command=command,
                           threaded=True)

    @sound.command(name="play", description="Play a sound in the call")
    async def sound_play(self, ctx, *, name: str):
        if ctx.guild is None:
            return
        lang = language(self.services, ctx.guild.id)
        try:
            row = find_sound(self.services, ctx.guild.id, name)
        except CommandError as e:
            return await send(ctx, e.render(lang))
        await self.services.play_sound(ctx.guild, row["id"])
        await send(ctx, i18n.t("sound.playing", lang, name=row["name"]))

    @sound_play.autocomplete("name")
    async def _sound_names(self, interaction, current: str):
        return [app_commands.Choice(name=n, value=n) for n in sound_names(self.services, interaction.guild_id, current)]

    @sound.command(name="volume", description="How loud a sound plays (100 = normal); yours, or any for admins")
    @app_commands.describe(name="The sound", percent="New volume in percent, e.g. 50 (leave out to see it)")
    async def sound_volume(self, ctx, name: str, percent: app_commands.Range[int, 0, 400] | None = None):
        await self.run_ctx(ctx, sound_volume, name, percent)

    @sound_volume.autocomplete("name")
    async def _volume_sounds(self, interaction, current: str):
        return [app_commands.Choice(name=n, value=n) for n in sound_names(self.services, interaction.guild_id, current)]

    @sound.command(name="remove", description="Remove one of your sounds")
    async def sound_remove(self, ctx, *, name: str):
        await self.run_ctx(ctx, remove_sound, name)

    @sound_remove.autocomplete("name")
    async def _own_sounds(self, interaction, current: str):
        actor = await self._actor(interaction)
        rows = self.services.store.list_sounds(interaction.guild_id)
        return [app_commands.Choice(name=n, value=n) for n in own_names(rows, actor, current)]

    # ------------------------------------------------------------ /voices
    @commands.hybrid_group(name="voices", fallback="list", description="Voices (lists the ready ones)")
    async def voices(self, ctx):
        if ctx.guild is not None:
            await send(ctx, list_voices(self.services, ctx.guild.id))

    @voices.command(name="preview", description="Hear a voice in the call")
    async def voices_preview(self, ctx, voice: str, *, text: str | None = None):
        if ctx.guild is None:
            return
        lang = language(self.services, ctx.guild.id)
        try:
            voice_id = resolve_voice(self.services, ctx.guild.id, voice)
        except CommandError as e:
            return await send(ctx, e.render(lang))
        if voice_id == SPEAKER_CHOICE:
            own = self.services.library.speaker_voice(ctx.guild.id, ctx.author.id)
            if own is None:
                return await send(ctx, i18n.t("voice.no_own", lang))
            voice_id = own["id"]
        text = (text or self.services.store.get_setting(ctx.guild.id, "voices.preview_text")
                or i18n.t("voice.preview_text", lang))[:300]
        await defer(ctx)
        await self.services.say_text(ctx.guild, text, voice_id)
        await send(ctx, i18n.t("voice.preview_sent", lang))

    @voices_preview.autocomplete("voice")
    async def _preview_voice(self, interaction, current: str):
        return await self._voice_autocomplete(interaction, current)

    @voices.command(name="create", description="Design a voice from a description (e.g. 'female, elderly, low pitch')")
    async def voices_create(self, ctx, name: str, *, description: str):
        await self.run_ctx(ctx, create_voice, name, description)

    @voices.command(name="clone", description="Clone a voice from a recording you have permission to use")
    @app_commands.describe(confirm="I have permission to use this voice")
    async def voices_clone(self, ctx, name: str, file: discord.Attachment, confirm: bool):
        gid = ctx.guild.id if ctx.guild else 0
        max_mb = float(self.services.store.get_setting(gid, "voices.max_mb"))
        if file.size > max_mb * 1024 * 1024:
            return await send(ctx, i18n.t("common.file_too_big", language(self.services, gid), mb=f"{max_mb:g}"))
        await defer(ctx)
        data = await file.read()
        await self.run_ctx(ctx, clone_voice, name, data, file.filename, confirm, threaded=True)

    @voices.command(name="remove", description="Remove one of your voices")
    async def voices_remove(self, ctx, *, name: str):
        await self.run_ctx(ctx, remove_voice, name)

    @voices_remove.autocomplete("name")
    async def _own_voices(self, interaction, current: str):
        actor = await self._actor(interaction)
        return [app_commands.Choice(name=n, value=n)
                for n in voice_names(self.services, interaction.guild_id, actor, current)]


async def setup(bot: commands.Bot, services) -> ContentCommands:
    """Register the commands and the persistent buttons. Call from setup_hook."""
    global _services, _bot
    _services, _bot = services, bot
    cog = ContentCommands(bot, services)
    await bot.add_cog(cog)
    bot.add_dynamic_items(QuotaRequestButton, QuotaDecisionButton)
    return cog
