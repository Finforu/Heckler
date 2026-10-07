"""Packs: a server's content as a YAML file you can read, edit by hand,
share and load into another server.

    python packs.py export --guild 123 packs/mine.yaml
    python packs.py import --guild 456 packs/mine.yaml [--mode replace]

A pack holds reactions, settings, nicknames and sound/voice details. Audio
files travel in a zip with the same name next to the YAML (mine.yaml +
mine.zip); a pack without sounds or voices is just the YAML. Voices and
sounds are referred to by name, never by database id, so a pack works in
any server. Built voice prompts (.pt) are not packed: they're tied to the
model version, and loading one runs pickle code. Imported voices are queued
to be built again.

The format (only `name` and one trigger are required per reaction):

    format: 1
    settings:
      gags.cooldown_s: 3
    people:                      # nicknames: what the bot calls someone
      - {user_id: 123456789012345678, nickname: Pepe}
    voices:
      - name: narrator
        kind: designed           # clone (from an audio file) / designed (from instruct) / speaker
        instruct: male, elderly, low pitch
        file: voices/narrator-source.mp3   # inside the zip (clones)
    sounds:
      - {name: airhorn, file: sounds/airhorn.ogg, gain_db: -3}
    reactions:
      - name: good night
        phrase: [good night, goodnight]   # any of these, as whole words
        say: ["Good night, {name}.", Sleep well.]   # one picked at random each time
        by: [Pepe]               # only when these people say it (user ids or names)
      - name: coffee -> tea
        swap: coffee             # "the cup of coffee" -> "No, the cup of tea."
        to: tea
        also: [coffees]          # other spellings of the word
        connectors: [of]         # default: Spanish de / e / y
        say: "No, {subject} {connector} {to}."
      - name: skit
        phrase: hello there
        voice: narrator          # default voice for its lines
        options:                 # alternatives; each one a line or a list of steps
          - Hello.
          - [Hello., {sound: airhorn}, {say: What?, voice: "@speaker"}]
      - name: leave
        command: [leave, go away]   # said after the wake word
        do: leave                # a built-in action: leave / stop / timer
      - name: hello
        event: hello             # wake ack leave unknown hello bye arrival timer_set timer_ring
        user_id: 123456789012345678   # optional: this person's own greeting
        say: "{name} is here."

Several triggers go in a list instead: `triggers: [{phrase: hello}, {swap: coffee, to: tea, connectors: [of]}]`.
Optional per reaction: kind (gag / sound / command / response; guessed from
the trigger), voice, by, enabled, status, cooldown_s, chance, created_by.

Import modes: merge (default) updates reactions with the same kind and name,
sounds and voices with the same name, and adds the rest. replace first
deletes the server's reactions, and its sounds, voices, settings and
nicknames when the pack has those sections (people's own speaker voices are
never touched).
"""
import argparse
import re
import zipfile
from pathlib import Path

import yaml

import reactions as rules

ROOT = Path(__file__).resolve().parent
FORMAT = 1
TRIGGER_KEYS = ("phrase", "swap", "command", "slash", "event")
KIND_FOR_TRIGGER = {"phrase": "gag", "swap": "gag", "command": "command", "slash": "sound", "event": "response"}
VOICE_FIELDS = ("kind", "owner_user_id", "ref_text", "instruct", "speed", "num_step", "gain_db", "tags", "language")


# ───────────────────────────── YAML style ─────────────────────────────

class _Dumper(yaml.SafeDumper):
    """Short lists of plain values on one line ([a, b, c]); the rest in block style."""

    def ignore_aliases(self, data):
        return True


def _represent_list(dumper, data):
    flow = all(isinstance(x, (str, int, float, bool)) for x in data) and len(repr(data)) <= 70
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]+", "-", name, flags=re.UNICODE).strip("-") or "file"


def _one_or_list(values: list):
    return values[0] if len(values) == 1 else list(values)


# ───────────────────────────── export ─────────────────────────────

def _is_personal(row: dict) -> bool:
    """Tied to someone's Discord account: their own greetings, by= with ids."""
    if any(t["type"] == "event" and t.get("user_id") is not None for t in row["triggers"]):
        return True
    return any(rules._user_id(who) is not None for who in row.get("by_users") or [])


def _trigger_to_yaml(t: dict) -> dict:
    kind = t["type"]
    if kind in ("phrase", "command"):
        return {kind: _one_or_list(t["phrases"])}
    if kind == "swap":
        out = {"swap": t["word"], "to": t["to"]}
        if t.get("also"):
            out["also"] = list(t["also"])
        if list(t.get("connectors", rules.DEFAULT_CONNECTORS)) != list(rules.DEFAULT_CONNECTORS):
            out["connectors"] = list(t["connectors"])
        return out
    if kind == "slash":
        return {"slash": t["name"]}
    out = {"event": t["event"]}
    if t.get("user_id") is not None:
        out["user_id"] = t["user_id"]
    return out


def _reaction_to_yaml(row: dict, voice_names: dict, sound_names: dict) -> dict:
    out = {"name": row["name"]}
    triggers = [_trigger_to_yaml(t) for t in row["triggers"]]
    if row["kind"] != KIND_FOR_TRIGGER[row["triggers"][0]["type"]]:
        out["kind"] = row["kind"]
    if len(triggers) == 1:
        out.update(triggers[0])
    else:
        out["triggers"] = triggers

    def voice(voice_id):
        return voice_id if voice_id == rules.SPEAKER else voice_names.get(voice_id)

    if row["voice_id"] is not None and voice(row["voice_id"]):
        out["voice"] = voice(row["voice_id"])

    options = row["options"]
    plain_lines = all(len(o) == 1 and o[0]["type"] == "say" and o[0].get("voice_id") is None for o in options)
    if plain_lines:
        out["say"] = _one_or_list([o[0]["text"] for o in options])
    elif len(options) == 1 and len(options[0]) == 1 and options[0][0]["type"] == "builtin":
        out["do"] = options[0][0]["action"]
    else:
        def step(s):
            if s["type"] == "say":
                name = voice(s.get("voice_id")) if s.get("voice_id") is not None else None
                return {"say": s["text"], "voice": name} if name else s["text"]
            if s["type"] == "sound":
                return {"sound": sound_names.get(s["sound_id"], f"#{s['sound_id']}")}
            return {"builtin": s["action"]}

        out["options"] = [step(o[0]) if len(o) == 1 and isinstance(step(o[0]), str) else [step(s) for s in o]
                          for o in options]

    if row.get("by_users"):
        out["by"] = list(row["by_users"])
    if not row["enabled"]:
        out["enabled"] = False
    if row["status"] != "approved":
        out["status"] = row["status"]
    if row["cooldown_s"] is not None:
        out["cooldown_s"] = row["cooldown_s"]
    if row["chance"] != 1:
        out["chance"] = row["chance"]
    if row.get("created_by") is not None:
        out["created_by"] = row["created_by"]
    return out


def pack_data(store, guild_id: int, *, include_personal: bool = True, base_dir: Path = ROOT):
    """The pack as a dict, plus {path inside the zip: file on disk} for the audio."""
    files: dict[str, Path] = {}

    def add_file(path: str | None, arcname: str) -> str | None:
        if not path:
            return None
        source = Path(path) if Path(path).is_absolute() else base_dir / path
        if not source.is_file():
            return None
        arcname += source.suffix.lower()
        files[arcname] = source
        return arcname

    voices = [v for v in store.list_voices(guild_id, include_global=False)
              if include_personal or v["kind"] != "speaker"]
    voice_names = {v["id"]: v["name"] for v in store.list_voices(guild_id)}
    sounds = store.list_sounds(guild_id)
    sound_names = {s["id"]: s["name"] for s in sounds}

    data: dict = {"format": FORMAT}
    settings = store.settings(guild_id, effective=False)
    if settings:
        data["settings"] = settings
    if include_personal:
        people = [{"user_id": p["user_id"], "nickname": p["nickname"]}
                  for p in store.list_people(guild_id) if p["nickname"]]
        if people:
            data["people"] = people
    if voices:
        data["voices"] = []
        for v in voices:
            item = {"name": v["name"]}
            for key in VOICE_FIELDS:
                value = v.get(key)
                if value not in (None, [], "") and not (key == "gain_db" and value == 0) \
                        and not (key == "owner_user_id" and not include_personal):
                    item[key] = value
            source = add_file(v["source_path"], f"voices/{_slug(v['name'])}-source")
            ref = add_file(v["ref_path"], f"voices/{_slug(v['name'])}-ref")
            if source:
                item["file"] = source
            if ref:
                item["ref_file"] = ref
            data["voices"].append(item)
    if sounds:
        data["sounds"] = []
        for s in sounds:
            item = {"name": s["name"]}
            arcname = add_file(s["path"], f"sounds/{_slug(s['name'])}")
            if arcname:
                item["file"] = arcname
            if s["duration_s"] is not None:
                item["duration_s"] = s["duration_s"]
            if s["gain_db"]:
                item["gain_db"] = s["gain_db"]
            if not s["enabled"]:
                item["enabled"] = False
            if include_personal and s["created_by"] is not None:
                item["created_by"] = s["created_by"]
            data["sounds"].append(item)
    rows = [r for r in store.list_reactions(guild_id) if include_personal or not _is_personal(r)]
    data["reactions"] = []
    for row in rows:
        item = _reaction_to_yaml(row, voice_names, sound_names)
        if not include_personal:
            item.pop("created_by", None)
        data["reactions"].append(item)
    return data, files


def dump(data: dict, header: str = "") -> str:
    text = yaml.dump(data, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=110)
    # A blank line between top-level sections and between reactions reads better.
    text = re.sub(r"\n(?=[a-z_]+:)", "\n\n", text)
    text = re.sub(r"\n(?=- name:)", "\n\n", text)
    return header + text


def export_pack(store, guild_id: int, path: str | Path, *, include_personal: bool = True,
                base_dir: Path = ROOT, header: str = "") -> Path:
    """Write the server's content to `path` (.yaml), and its audio files to
    the .zip next to it when there are any. Returns the YAML path.
    include_personal=False leaves out what's tied to people's accounts:
    nicknames, their own greetings, by= with user ids, speaker voices."""
    path = Path(path)
    data, files = pack_data(store, guild_id, include_personal=include_personal, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(data, header), encoding="utf-8")
    archive = path.with_suffix(".zip")
    if files:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for arcname, source in files.items():
                z.write(source, arcname)
    elif archive.exists():
        archive.unlink()  # an old export's audio would be mistaken for this one's
    return path


# ───────────────────────────── import ─────────────────────────────

def _trigger_from_yaml(d: dict, where: str) -> dict:
    present = [k for k in TRIGGER_KEYS if k in d]
    if len(present) != 1:
        raise ValueError(f"{where}: needs exactly one of {', '.join(TRIGGER_KEYS)}")
    kind = present[0]
    as_list = lambda v: [str(x) for x in ([v] if isinstance(v, (str, int)) else list(v or []))]
    if kind in ("phrase", "command"):
        return {"type": kind, "phrases": as_list(d[kind])}
    if kind == "swap":
        if "to" not in d:
            raise ValueError(f"{where}: swap needs `to`")
        return {"type": "swap", "word": str(d["swap"]), "to": str(d["to"]), "also": as_list(d.get("also")),
                "connectors": as_list(d.get("connectors", list(rules.DEFAULT_CONNECTORS)))}
    if kind == "slash":
        return {"type": "slash", "name": str(d["slash"])}
    trigger = {"type": "event", "event": d["event"]}
    if d.get("user_id") is not None:
        trigger["user_id"] = rules._user_id(d["user_id"])
    return trigger


class _Resolver:
    """Voice and sound names -> ids, collecting what couldn't be found."""

    def __init__(self, store, guild_id: int, warnings: list[str]):
        self.store, self.guild_id, self.warnings = store, guild_id, warnings

    def voice(self, name, where: str):
        if name is None or name == rules.SPEAKER:
            return name
        voice = self.store.find_voice(self.guild_id, str(name))
        if voice is None:
            self.warnings.append(f"{where}: no voice named {name!r}, using the default voice")
            return None
        return voice["id"]

    def sound(self, name, where: str):
        sound = self.store.find_sound(self.guild_id, str(name))
        if sound is None:
            self.warnings.append(f"{where}: no sound named {name!r}, step dropped")
            return None
        return sound["id"]


def _options_from_yaml(d: dict, where: str, resolve: _Resolver) -> list[list[dict]]:
    given = [k for k in ("say", "options", "do") if k in d]
    if len(given) != 1:
        raise ValueError(f"{where}: needs exactly one of say, options, do")

    def step(s) -> dict | None:
        if isinstance(s, (str, int, float)):
            return {"type": "say", "text": str(s)}
        if not isinstance(s, dict):
            raise ValueError(f"{where}: bad step {s!r}")
        if "say" in s:
            out = {"type": "say", "text": str(s["say"])}
            if s.get("voice") is not None:
                out["voice_id"] = resolve.voice(s["voice"], where)
            return out
        if "sound" in s:
            sound_id = resolve.sound(s["sound"], where)
            return None if sound_id is None else {"type": "sound", "sound_id": sound_id}
        if "builtin" in s:
            return {"type": "builtin", "action": s["builtin"]}
        raise ValueError(f"{where}: a step is a line, or one of say / sound / builtin: {s!r}")

    if "say" in d:
        lines = d["say"] if isinstance(d["say"], list) else [d["say"]]
        return [[step(line)] for line in lines]
    if "do" in d:
        return [[{"type": "builtin", "action": d["do"]}]]
    options = []
    for alternative in d["options"] or []:
        steps = [step(s) for s in (alternative if isinstance(alternative, list) else [alternative])]
        steps = [s for s in steps if s is not None]
        if steps:
            options.append(steps)
    return options


def _extract(archive: zipfile.ZipFile | None, arcname: str | None, media_dir: Path, base_dir: Path,
             where: str, warnings: list[str]) -> str | None:
    """Copy one file out of the zip into media_dir; its path as stored (relative to base_dir when inside it)."""
    if not arcname:
        return None
    if archive is None or arcname not in archive.namelist():
        warnings.append(f"{where}: {arcname} isn't in the pack's zip")
        return None
    target = (media_dir / arcname).resolve()
    if not target.is_relative_to(media_dir.resolve()):
        raise ValueError(f"{where}: file path {arcname!r} points outside the pack")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archive.read(arcname))
    return str(target.relative_to(base_dir.resolve())) if target.is_relative_to(base_dir.resolve()) else str(target)


def load(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a pack")
    if int(data.get("format", FORMAT)) > FORMAT:
        raise ValueError(f"{path}: pack format {data['format']} is newer than this bot understands ({FORMAT})")
    return data


def import_pack(store, guild_id: int, path: str | Path, *, mode: str = "merge", media_dir: Path | None = None,
                base_dir: Path = ROOT) -> dict:
    """Load a pack into a server, all or nothing. Audio from the zip next to
    it goes to media_dir (default data/media/<guild_id>). Returns counts and
    a list of warnings (unknown voice or sound names, missing files)."""
    if mode not in ("merge", "replace"):
        raise ValueError("mode must be merge or replace")
    path = Path(path)
    data = load(path)
    media_dir = Path(media_dir) if media_dir else base_dir / "data" / "media" / str(guild_id)
    zip_path = path.with_suffix(".zip")
    archive = zipfile.ZipFile(zip_path) if zip_path.exists() else None
    warnings: list[str] = []
    counts = {"reactions": 0, "sounds": 0, "voices": 0, "people": 0, "settings": 0}
    resolve = _Resolver(store, guild_id, warnings)
    try:
        with store.batch():
            if mode == "replace":
                for row in store.list_reactions(guild_id):
                    store.delete_reaction(row["id"])
                if "sounds" in data:
                    for s in store.list_sounds(guild_id):
                        store.delete_sound(s["id"])
                if "voices" in data:
                    for v in store.list_voices(guild_id, include_global=False):
                        if v["kind"] != "speaker":
                            store.delete_voice(v["id"])
                if "settings" in data:
                    for key in store.settings(guild_id, effective=False):
                        store.delete_setting(guild_id, key)
                if "people" in data:
                    for p in store.list_people(guild_id):
                        store.set_person(guild_id, p["user_id"], nickname=None)

            for key, value in (data.get("settings") or {}).items():
                store.set_setting(guild_id, str(key), value)
                counts["settings"] += 1

            for p in data.get("people") or []:
                store.set_person(guild_id, int(p["user_id"]), nickname=p.get("nickname"))
                counts["people"] += 1

            for v in data.get("voices") or []:
                where = f"voice {v.get('name')!r}"
                if not v.get("name"):
                    raise ValueError("every voice needs a name")
                fields = {k: v[k] for k in VOICE_FIELDS if k in v}
                source = _extract(archive, v.get("file"), media_dir, base_dir, where, warnings)
                ref = _extract(archive, v.get("ref_file"), media_dir, base_dir, where, warnings)
                if source:
                    fields["source_path"] = source
                if ref:
                    fields["ref_path"] = ref
                # Prompts aren't packed: (re)build from what came in.
                fields.update(status="queued", prompt_path=None, prompt_hash=None, error=None)
                existing = next((x for x in store.list_voices(guild_id, include_global=False)
                                 if x["name"].casefold() == str(v["name"]).casefold()), None)
                if existing:
                    store.update_voice(existing["id"], **fields)
                else:
                    store.add_voice(str(v["name"]), guild_id=guild_id, **fields)
                counts["voices"] += 1

            for s in data.get("sounds") or []:
                where = f"sound {s.get('name')!r}"
                if not s.get("name"):
                    raise ValueError("every sound needs a name")
                fields = {k: s[k] for k in ("duration_s", "gain_db", "enabled", "created_by") if k in s}
                file = _extract(archive, s.get("file"), media_dir, base_dir, where, warnings)
                existing = store.find_sound(guild_id, str(s["name"]))
                if existing:
                    store.update_sound(existing["id"], **fields, **({"path": file} if file else {}))
                elif file:
                    store.add_sound(guild_id, str(s["name"]), file, **fields)
                else:
                    warnings.append(f"{where}: no audio file, skipped")
                    continue
                counts["sounds"] += 1

            existing = {(r["kind"], r["name"]): r["id"] for r in store.list_reactions(guild_id)}
            for i, r in enumerate(data.get("reactions") or []):
                if not isinstance(r, dict) or not r.get("name"):
                    raise ValueError(f"reaction #{i + 1}: needs a name")
                where = f"reaction {r['name']!r}"
                if "triggers" in r:
                    triggers = [_trigger_from_yaml(t, where) for t in r["triggers"] or []]
                else:
                    triggers = [_trigger_from_yaml(r, where)]
                if not triggers:
                    raise ValueError(f"{where}: no triggers")
                fields = dict(
                    triggers=triggers,
                    options=_options_from_yaml(r, where, resolve),
                    voice_id=resolve.voice(r.get("voice"), where),
                    by_users=([r["by"]] if isinstance(r["by"], (str, int)) else list(r["by"])) if r.get("by") else None,
                    enabled=bool(r.get("enabled", True)),
                    status=r.get("status", "approved"),
                    cooldown_s=r.get("cooldown_s"),
                    chance=float(r.get("chance", 1.0)),
                    created_by=r.get("created_by"),
                )
                kind = r.get("kind") or KIND_FOR_TRIGGER[triggers[0]["type"]]
                try:
                    if (kind, r["name"]) in existing:
                        store.update_reaction(existing[(kind, r["name"])], **fields)
                    else:
                        existing[(kind, r["name"])] = store.add_reaction(guild_id, kind, str(r["name"]), **fields)
                except ValueError as e:
                    raise ValueError(f"{where}: {e}") from None
                counts["reactions"] += 1
    finally:
        if archive is not None:
            archive.close()
    return {**counts, "warnings": warnings}


DEFAULT_STARTER = "base-en"


def seed_guild(store, guild_id: int, *, packs_dir: Path = ROOT / "packs") -> str:
    """Give a server with no reactions its starting content, and say what was done.

    The setting content.starter_pack picks a pack under packs/ (e.g. "base-es")
    or "none". Unset: base-<the server's language> (see i18n), else base-en."""
    if store.list_reactions(guild_id):
        return "kept: the server already has content"
    choice = store.get_setting(guild_id, "content.starter_pack")
    if choice is None:
        import i18n

        choice = f"base-{i18n.guild_language(store, guild_id)}"
        if not (packs_dir / f"{choice}.yaml").is_file():
            choice = DEFAULT_STARTER
    choice = str(choice).strip()
    if choice.casefold() == "none":
        return "nothing (content.starter_pack is none)"
    if not re.fullmatch(r"[\w.-]+", choice) or choice.startswith("."):
        return f"nothing: {choice!r} isn't a pack name"
    path = packs_dir / (choice if choice.endswith(".yaml") else f"{choice}.yaml")
    if not path.is_file():
        return f"nothing: no pack {path.name} in {packs_dir}"
    result = import_pack(store, guild_id, path)
    note = f" ({len(result['warnings'])} warnings)" if result["warnings"] else ""
    return f"pack {path.stem}: {result['reactions']} reactions{note}"


def main() -> None:
    from store import Store

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("export", "import"))
    parser.add_argument("path", help="the pack's .yaml (its audio goes in the .zip next to it)")
    parser.add_argument("--db", default=str(ROOT / "data" / "bot.db"))
    parser.add_argument("--guild", type=int, required=True)
    parser.add_argument("--mode", choices=("merge", "replace"), default="merge", help="import only")
    parser.add_argument("--no-personal", action="store_true",
                        help="export only: leave out nicknames, people's own greetings and speaker voices")
    args = parser.parse_args()
    store = Store(args.db)
    if args.action == "export":
        print(export_pack(store, args.guild, args.path, include_personal=not args.no_personal))
    else:
        result = import_pack(store, args.guild, args.path, mode=args.mode)
        for warning in result.pop("warnings"):
            print("warning:", warning)
        print(", ".join(f"{count} {what}" for what, count in result.items()))
    store.close()


if __name__ == "__main__":
    main()
