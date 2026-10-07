"""Everything a server configures, in one SQLite file (data/bot.db):
reactions (gags, sounds, commands, stock responses), voices, sounds,
nicknames, consent, settings, quotas and an event log.

One Store is shared by the bot and the dashboard. Calls are synchronous and
fast (a local file, tiny rows), so the bot calls them straight from the
event loop; one lock makes them safe from any thread. The database runs in
WAL mode so a reader (a backup, the sqlite3 shell) never blocks the bot.

Rows come back as plain dicts, JSON columns already decoded, so the
dashboard can serve them as they are. IDs stay ints here; whoever sends them
to a browser turns the 64-bit Discord ones into strings.

Settings resolve in three steps: the server's own value, then the global
one (guild_id 0), then DEFAULT_SETTINGS below.

The schema only ever moves forward: MIGRATIONS is a numbered list and a new
database runs all of them. To change the schema, append a migration; never
edit one that has shipped.
"""
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import reactions as reaction_rules

log = logging.getLogger(__name__)

GLOBAL = 0  # the guild_id that holds global settings

DEFAULT_SETTINGS: dict[str, Any] = {
    # None / empty: the bot's own configuration (BOT_NAME / WAKE_WORDS).
    "bot.name": None,
    "bot.wake_words": [],
    # Voice id for say steps that don't pick one; None = the bot's own voice.
    "voice.default": None,
    # Global only: the id of the bot's own voice (voice_library.ensure_bot_voice).
    "voice.bot": None,
    # Global only: disk space for generated lines before the oldest are dropped.
    "cache.tts_mb": 500,
    "quota.gags": 15,
    "quota.sounds": 15,
    "quota.voices": 5,
    "user_content.needs_approval": False,
    # Seconds before the same person can set off the same gag again.
    "gags.cooldown_s": 3,
    # Multiplies every gag's chance: 0.5 = gags fire half as often.
    "gags.intensity": 1.0,
    # How sure speech-to-text must be of a sentence (Whisper's mean word
    # log-probability: 0 = certain; clear speech is about -0.1 to -0.4,
    # mumbled or misheard below -0.7) before it can set off gags and
    # commands. Raise it (e.g. -0.5) if gags misfire; None = no check.
    "stt.min_confidence": -0.7,
    # The server's language ("en", "es"...): replies, speech-to-text, starter
    # pack. None = DEFAULT_LANGUAGE from .env (see i18n.py).
    "language": None,
    # Keep what people say in the event log. Off by default; even when on,
    # only the words of people who accepted (consent purpose "transcripts").
    "transcripts.enabled": False,
    # Who counts as an admin in Discord besides the bot owner and members
    # with Manage Server: members with this role id.
    "admin.role_id": None,
    # Where quota requests go (a channel id); None = a DM to the bot owner(s).
    "admin.channel_id": None,
    # Uploaded sounds: longer ones are cut, bigger files refused.
    "sounds.max_seconds": 15,
    "sounds.max_mb": 5,
    # Uploaded voice samples (/voices clone).
    "voices.max_mb": 20,
    # What a new server starts with: a pack under packs/ ("base-en", "base-es"...)
    # or "none". Unset: base-<the server's language>. See packs.seed_guild.
    "content.starter_pack": None,
    # What /voices preview says when given no text.
    "voices.preview_text": "Hello! This is how I sound.",
}

QUOTA_RESOURCES = ("gags", "sounds", "voices")
REACTION_KINDS = ("gag", "sound", "command", "response")
REACTION_STATUSES = ("approved", "pending", "rejected")
VOICE_KINDS = ("clone", "designed", "speaker")
VOICE_STATUSES = ("draft", "queued", "building", "ready", "failed")
CONSENT_STATUSES = ("pending", "accepted", "declined", "revoked")
CONSENT_PURPOSES = ("voice", "transcripts")
REQUEST_STATUSES = ("pending", "approved", "denied")

MIGRATIONS: list[str] = [
    # 1: everything
    """
    CREATE TABLE settings (
        guild_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,              -- JSON
        PRIMARY KEY (guild_id, key)
    );
    CREATE TABLE people (
        guild_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        nickname TEXT,
        display_name TEXT,
        last_seen TEXT,
        PRIMARY KEY (guild_id, user_id)
    );
    CREATE TABLE consent (
        user_id INTEGER PRIMARY KEY,      -- global: a voice belongs to the person
        status TEXT NOT NULL DEFAULT 'pending',
        text_version TEXT,
        asked_at TEXT,
        decided_at TEXT
    );
    CREATE TABLE voices (
        id INTEGER PRIMARY KEY,
        guild_id INTEGER,                 -- NULL: available in every server
        name TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'clone',
        owner_user_id INTEGER,
        ref_text TEXT,
        instruct TEXT,
        speed REAL,
        num_step INTEGER,
        gain_db REAL NOT NULL DEFAULT 0,
        tags TEXT NOT NULL DEFAULT '[]',  -- JSON
        language TEXT,
        source_path TEXT,
        ref_path TEXT,
        prompt_path TEXT,
        prompt_hash TEXT,
        status TEXT NOT NULL DEFAULT 'draft',
        error TEXT,
        created_by INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX voices_guild ON voices (guild_id);
    CREATE TABLE sounds (
        id INTEGER PRIMARY KEY,
        guild_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        duration_s REAL,
        gain_db REAL NOT NULL DEFAULT 0,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_by INTEGER,
        created_at TEXT NOT NULL,
        UNIQUE (guild_id, name)
    );
    CREATE TABLE reactions (
        id INTEGER PRIMARY KEY,
        guild_id INTEGER NOT NULL,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'approved',
        created_by INTEGER,               -- NULL: an admin or an import
        triggers TEXT NOT NULL,           -- JSON list
        options TEXT NOT NULL,            -- JSON list of step lists
        voice_id,                         -- untyped: a voice id or "@speaker"
        by_users TEXT,                    -- JSON list, NULL = anyone
        cooldown_s REAL,                  -- NULL: the setting
        chance REAL NOT NULL DEFAULT 1,
        position INTEGER NOT NULL DEFAULT 0,
        uses INTEGER NOT NULL DEFAULT 0,
        last_used_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX reactions_guild ON reactions (guild_id, position, id);
    CREATE TABLE quota_overrides (
        guild_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        resource TEXT NOT NULL,
        "limit" INTEGER NOT NULL,
        PRIMARY KEY (guild_id, user_id, resource)
    );
    CREATE TABLE quota_requests (
        id INTEGER PRIMARY KEY,
        guild_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        resource TEXT NOT NULL,
        amount INTEGER NOT NULL,
        reason TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        decided_at TEXT,
        decided_by INTEGER
    );
    CREATE INDEX quota_requests_guild ON quota_requests (guild_id, status);
    CREATE TABLE events (
        id INTEGER PRIMARY KEY,
        time TEXT NOT NULL,
        guild_id INTEGER,
        user_id INTEGER,
        user_name TEXT,
        text TEXT,
        reaction_id INTEGER,
        voice_id,
        outcome TEXT,
        details TEXT                      -- JSON
    );
    CREATE INDEX events_guild ON events (guild_id, id);
    """,
    # 2: one person's history (dashboard filter, "delete their history")
    """
    CREATE INDEX events_guild_user ON events (guild_id, user_id, id);
    """,
    # 3: consent per purpose: cloning someone's voice and keeping their words
    # are separate questions. Existing answers were about the voice.
    """
    CREATE TABLE consent_new (
        user_id INTEGER NOT NULL,
        purpose TEXT NOT NULL,            -- voice / transcripts
        status TEXT NOT NULL DEFAULT 'pending',
        text_version TEXT,
        asked_at TEXT,
        decided_at TEXT,
        PRIMARY KEY (user_id, purpose)
    );
    INSERT INTO consent_new (user_id, purpose, status, text_version, asked_at, decided_at)
        SELECT user_id, 'voice', status, text_version, asked_at, decided_at FROM consent;
    DROP TABLE consent;
    ALTER TABLE consent_new RENAME TO consent;
    """,
]

# Columns each table accepts from callers (the rest are managed here), and
# which of them hold JSON or booleans.
_COLUMNS = {
    "reactions": ("guild_id", "kind", "name", "enabled", "status", "created_by", "triggers", "options",
                  "voice_id", "by_users", "cooldown_s", "chance", "position", "uses", "last_used_at"),
    "voices": ("guild_id", "name", "kind", "owner_user_id", "ref_text", "instruct", "speed", "num_step",
               "gain_db", "tags", "language", "source_path", "ref_path", "prompt_path", "prompt_hash",
               "status", "error", "created_by"),
    "sounds": ("guild_id", "name", "path", "duration_s", "gain_db", "enabled", "created_by"),
}
_JSON = {"triggers", "options", "by_users", "tags", "details", "value"}
_BOOL = {"enabled"}


def now() -> str:
    """Local time with its UTC offset, to the second: what every *_at column holds."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, much faster commits
        self._depth = 0  # nesting of batch()
        self._pending: list[tuple[str, int | None]] = []  # changes to announce after commit
        self._listeners: list[Callable[[str, int | None], None]] = []
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ plumbing
    def _migrate(self) -> None:
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO schema_version (version) VALUES (0)")
                self._conn.commit()
                version = 0
            else:
                version = row[0]
            if version > len(MIGRATIONS):
                raise RuntimeError(f"{self.path} is schema version {version}, newer than this code "
                                   f"({len(MIGRATIONS)}): update the bot")
            for number in range(version + 1, len(MIGRATIONS) + 1):
                # executescript commits whatever is open first, then runs the
                # whole migration as one transaction.
                self._conn.executescript(
                    f"BEGIN;\n{MIGRATIONS[number - 1]}\n"
                    f"UPDATE schema_version SET version = {number};\nCOMMIT;"
                )
                log.info("Database %s migrated to version %d", self.path, number)

    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT version FROM schema_version").fetchone()[0]

    def on_change(self, callback: Callable[[str, int | None], None]) -> Callable:
        """Call `callback(table, guild_id)` after every committed change
        (guild_id None: could affect every server, e.g. a global voice or
        setting). It runs on the thread that made the change, outside the
        lock; hop to your own loop with call_soon_threadsafe if needed.
        Returns the callback, so it works as a decorator."""
        self._listeners.append(callback)
        return callback

    def off_change(self, callback: Callable) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    @contextmanager
    def batch(self):
        """Several writes as one transaction: all or nothing, and listeners
        hear about them once, at the end. Nests."""
        with self._lock:
            outer = self._depth == 0
            self._depth += 1
            try:
                yield self
            except BaseException:
                if outer:
                    self._conn.rollback()
                    self._pending.clear()
                raise
            else:
                if outer:
                    self._conn.commit()
            finally:
                self._depth -= 1
            changes = []
            if outer:
                changes, self._pending = list(dict.fromkeys(self._pending)), []
        for table, guild_id in changes:
            for callback in list(self._listeners):
                try:
                    callback(table, guild_id)
                except Exception:
                    log.exception("Store change listener failed for %s/%s", table, guild_id)

    def _write(self, sql: str, params: Iterable = (), *, changed: tuple[str, int | None] | None = None):
        with self.batch():
            cursor = self._conn.execute(sql, tuple(params))
            if changed is not None:
                self._pending.append(changed)
            return cursor

    def _query(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _scalar(self, sql: str, params: Iterable = ()):
        rows = self._query(sql, params)
        return rows[0][0] if rows else None

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        out = {}
        for key in row.keys():
            value = row[key]
            if key in _JSON and value is not None:
                value = json.loads(value)
            elif key in _BOOL:
                value = bool(value)
            out[key] = value
        return out

    @staticmethod
    def _encode(key: str, value):
        if key in _JSON and value is not None:
            return json.dumps(value, ensure_ascii=False)
        if key in _BOOL:
            return int(bool(value))
        return value

    def _insert(self, table: str, fields: dict, *, changed_guild) -> int:
        bad = set(fields) - set(_COLUMNS[table])
        if bad:
            raise ValueError(f"unknown {table} fields: {sorted(bad)}")
        stamp = now()
        fields = {**fields, "created_at": stamp}
        if table != "sounds":
            fields["updated_at"] = stamp
        cols = ", ".join(f'"{k}"' for k in fields)
        marks = ", ".join("?" for _ in fields)
        cursor = self._write(f"INSERT INTO {table} ({cols}) VALUES ({marks})",
                             [self._encode(k, v) for k, v in fields.items()], changed=(table, changed_guild))
        return cursor.lastrowid

    def _update(self, table: str, row_id: int, fields: dict, *, changed_guild) -> None:
        bad = set(fields) - set(_COLUMNS[table])
        if bad:
            raise ValueError(f"unknown {table} fields: {sorted(bad)}")
        if table != "sounds":
            fields = {**fields, "updated_at": now()}
        if not fields:
            return
        sets = ", ".join(f'"{k}" = ?' for k in fields)
        self._write(f"UPDATE {table} SET {sets} WHERE id = ?",
                    [*(self._encode(k, v) for k, v in fields.items()), row_id], changed=(table, changed_guild))

    def _get(self, table: str, row_id: int) -> dict | None:
        return self._decode(next(iter(self._query(f"SELECT * FROM {table} WHERE id = ?", (row_id,))), None))

    # ------------------------------------------------------------ settings
    def get_setting(self, guild_id: int, key: str, default=None):
        """The server's value, else the global one, else DEFAULT_SETTINGS[key], else `default`."""
        rows = self._query("SELECT guild_id, value FROM settings WHERE key = ? AND guild_id IN (?, ?)",
                           (key, guild_id, GLOBAL))
        values = {row["guild_id"]: json.loads(row["value"]) for row in rows}
        if guild_id in values:
            return values[guild_id]
        if GLOBAL in values:
            return values[GLOBAL]
        return DEFAULT_SETTINGS.get(key, default)

    def set_setting(self, guild_id: int, key: str, value) -> None:
        """guild_id 0 sets the global default."""
        self._write("INSERT INTO settings (guild_id, key, value) VALUES (?, ?, ?) "
                    "ON CONFLICT (guild_id, key) DO UPDATE SET value = excluded.value",
                    (guild_id, key, json.dumps(value, ensure_ascii=False)),
                    changed=("settings", None if guild_id == GLOBAL else guild_id))

    def delete_setting(self, guild_id: int, key: str) -> None:
        """Back to the global value (or the built-in default)."""
        self._write("DELETE FROM settings WHERE guild_id = ? AND key = ?", (guild_id, key),
                    changed=("settings", None if guild_id == GLOBAL else guild_id))

    def settings(self, guild_id: int, *, effective: bool = True) -> dict:
        """effective: every known key with the value that applies here.
        Otherwise only what this guild_id sets itself."""
        own = {row["key"]: json.loads(row["value"])
               for row in self._query("SELECT key, value FROM settings WHERE guild_id = ?", (guild_id,))}
        if not effective:
            return own
        merged = dict(DEFAULT_SETTINGS)
        if guild_id != GLOBAL:
            merged.update(self.settings(GLOBAL, effective=False))
        merged.update(own)
        return merged

    # ----------------------------------------------------------- reactions
    def add_reaction(self, guild_id: int, kind: str, name: str, triggers: list, options: list, *,
                     enabled: bool = True, status: str = "approved", created_by: int | None = None,
                     voice_id=None, by_users: list | None = None, cooldown_s: float | None = None,
                     chance: float = 1.0, position: int | None = None) -> int:
        """position None: after the server's last reaction (first match wins,
        so order matters)."""
        self._check_reaction(kind, status, triggers, options, chance)
        with self.batch():
            if position is None:
                position = (self._scalar("SELECT MAX(position) FROM reactions WHERE guild_id = ?",
                                         (guild_id,)) or 0) + 1
            return self._insert("reactions", dict(
                guild_id=guild_id, kind=kind, name=name, enabled=enabled, status=status,
                created_by=created_by, triggers=triggers, options=options, voice_id=voice_id,
                by_users=by_users, cooldown_s=cooldown_s, chance=chance, position=position,
            ), changed_guild=guild_id)

    def get_reaction(self, reaction_id: int) -> dict | None:
        return self._get("reactions", reaction_id)

    def list_reactions(self, guild_id: int, *, kind: str | None = None, status: str | None = None,
                       created_by: int | None = None) -> list[dict]:
        """In matching order."""
        sql, params = "SELECT * FROM reactions WHERE guild_id = ?", [guild_id]
        for column, value in (("kind", kind), ("status", status), ("created_by", created_by)):
            if value is not None:
                sql += f" AND {column} = ?"
                params.append(value)
        return [self._decode(r) for r in self._query(sql + " ORDER BY position, id", params)]

    def update_reaction(self, reaction_id: int, **fields) -> None:
        current = self.get_reaction(reaction_id)
        if current is None:
            raise KeyError(f"no reaction {reaction_id}")
        merged = {**current, **fields}
        self._check_reaction(merged["kind"], merged["status"], merged["triggers"], merged["options"],
                             merged["chance"])
        self._update("reactions", reaction_id, fields, changed_guild=current["guild_id"])

    def delete_reaction(self, reaction_id: int) -> None:
        guild_id = self._scalar("SELECT guild_id FROM reactions WHERE id = ?", (reaction_id,))
        self._write("DELETE FROM reactions WHERE id = ?", (reaction_id,), changed=("reactions", guild_id))

    def reorder_reactions(self, guild_id: int, ids: list[int]) -> None:
        """Put these reactions first, in this order; the rest keep theirs after them."""
        rest = [r["id"] for r in self.list_reactions(guild_id) if r["id"] not in set(ids)]
        with self.batch():
            for position, reaction_id in enumerate([*ids, *rest]):
                self._write("UPDATE reactions SET position = ? WHERE id = ? AND guild_id = ?",
                            (position, reaction_id, guild_id), changed=("reactions", guild_id))

    def record_use(self, reaction_id: int) -> None:
        """Count a reaction that actually played. No change notification:
        stats aren't worth recompiling over."""
        self._write("UPDATE reactions SET uses = uses + 1, last_used_at = ? WHERE id = ?", (now(), reaction_id))

    @staticmethod
    def _check_reaction(kind, status, triggers, options, chance) -> None:
        if kind not in REACTION_KINDS:
            raise ValueError(f"kind must be one of {REACTION_KINDS}, not {kind!r}")
        if status not in REACTION_STATUSES:
            raise ValueError(f"status must be one of {REACTION_STATUSES}, not {status!r}")
        if not 0 <= float(chance) <= 1:
            raise ValueError("chance must be between 0 and 1")
        reaction_rules.validate(triggers, options)

    # -------------------------------------------------------------- voices
    def add_voice(self, name: str, *, guild_id: int | None = None, kind: str = "clone", **fields) -> int:
        """guild_id None: a voice every server can use (e.g. the bot's own)."""
        if kind not in VOICE_KINDS:
            raise ValueError(f"kind must be one of {VOICE_KINDS}, not {kind!r}")
        if fields.get("status", "draft") not in VOICE_STATUSES:
            raise ValueError(f"status must be one of {VOICE_STATUSES}")
        return self._insert("voices", {"guild_id": guild_id, "name": name, "kind": kind, **fields},
                            changed_guild=guild_id)

    def get_voice(self, voice_id: int) -> dict | None:
        return self._get("voices", voice_id)

    def find_voice(self, guild_id: int, name: str) -> dict | None:
        """By name (any case): the server's own voice first, then a global one."""
        rows = self._query("SELECT * FROM voices WHERE name = ? COLLATE NOCASE AND (guild_id = ? OR guild_id IS NULL) "
                           "ORDER BY guild_id IS NULL, id", (name, guild_id))
        return self._decode(rows[0]) if rows else None

    def list_voices(self, guild_id: int | None, *, include_global: bool = True) -> list[dict]:
        if guild_id is None:
            rows = self._query("SELECT * FROM voices WHERE guild_id IS NULL ORDER BY name, id")
        elif include_global:
            rows = self._query("SELECT * FROM voices WHERE guild_id = ? OR guild_id IS NULL "
                               "ORDER BY guild_id IS NULL, name, id", (guild_id,))
        else:
            rows = self._query("SELECT * FROM voices WHERE guild_id = ? ORDER BY name, id", (guild_id,))
        return [self._decode(r) for r in rows]

    def find_voices(self, *, kind: str | None = None, owner_user_id: int | None = None,
                    statuses: Iterable[str] | None = None) -> list[dict]:
        """Voices in any server matching all the given filters (e.g. someone's speaker voices)."""
        sql, params = "SELECT * FROM voices WHERE 1", []
        if kind is not None:
            sql, params = sql + " AND kind = ?", [*params, kind]
        if owner_user_id is not None:
            sql, params = sql + " AND owner_user_id = ?", [*params, owner_user_id]
        if statuses is not None:
            statuses = list(statuses)
            sql += f" AND status IN ({', '.join('?' for _ in statuses)})"
            params += statuses
        return [self._decode(r) for r in self._query(sql + " ORDER BY guild_id IS NULL, id", params)]

    def update_voice(self, voice_id: int, **fields) -> None:
        current = self.get_voice(voice_id)
        if current is None:
            raise KeyError(f"no voice {voice_id}")
        if "status" in fields and fields["status"] not in VOICE_STATUSES:
            raise ValueError(f"status must be one of {VOICE_STATUSES}")
        self._update("voices", voice_id, fields, changed_guild=current["guild_id"])

    def delete_voice(self, voice_id: int) -> None:
        """Reactions that used it fall back to the next voice in line."""
        current = self.get_voice(voice_id)
        if current is None:
            return
        with self.batch():
            self._write("DELETE FROM voices WHERE id = ?", (voice_id,), changed=("voices", current["guild_id"]))
            self._scrub("voice_id", voice_id, current["guild_id"])

    # -------------------------------------------------------------- sounds
    def add_sound(self, guild_id: int, name: str, path: str, **fields) -> int:
        return self._insert("sounds", {"guild_id": guild_id, "name": name, "path": str(path), **fields},
                            changed_guild=guild_id)

    def get_sound(self, sound_id: int) -> dict | None:
        return self._get("sounds", sound_id)

    def find_sound(self, guild_id: int, name: str) -> dict | None:
        rows = self._query("SELECT * FROM sounds WHERE guild_id = ? AND name = ? COLLATE NOCASE", (guild_id, name))
        return self._decode(rows[0]) if rows else None

    def list_sounds(self, guild_id: int) -> list[dict]:
        return [self._decode(r) for r in self._query("SELECT * FROM sounds WHERE guild_id = ? ORDER BY name",
                                                     (guild_id,))]

    def update_sound(self, sound_id: int, **fields) -> None:
        current = self.get_sound(sound_id)
        if current is None:
            raise KeyError(f"no sound {sound_id}")
        self._update("sounds", sound_id, fields, changed_guild=current["guild_id"])

    def delete_sound(self, sound_id: int) -> None:
        """Steps that played it are removed from the reactions using it."""
        current = self.get_sound(sound_id)
        if current is None:
            return
        with self.batch():
            self._write("DELETE FROM sounds WHERE id = ?", (sound_id,), changed=("sounds", current["guild_id"]))
            self._scrub("sound_id", sound_id, current["guild_id"])

    def _scrub(self, key: str, value: int, guild_id: int | None) -> None:
        """Remove references to a deleted voice or sound from reaction rows,
        so nothing points at an id that a later row could reuse."""
        sql, params = "SELECT * FROM reactions", ()
        if guild_id is not None:
            sql, params = sql + " WHERE guild_id = ?", (guild_id,)
        for row in map(self._decode, self._query(sql, params)):
            changes = {}
            if key == "voice_id" and row["voice_id"] == value:
                changes["voice_id"] = None
            options = []
            for option in row["options"]:
                steps = []
                for step in option:
                    if step.get(key) == value:
                        if key == "sound_id":
                            continue  # a sound step without its sound is nothing
                        step = {**step, key: None}
                    steps.append(step)
                if steps:
                    options.append(steps)
            if options != row["options"]:
                changes["options"] = options
            if changes:
                if not options:
                    changes["enabled"] = False  # nothing left to play; keep the row for the admin
                    changes.pop("options")
                self._update("reactions", row["id"], changes, changed_guild=row["guild_id"])

    # -------------------------------------------------------------- people
    def set_person(self, guild_id: int, user_id: int, **fields) -> None:
        """Create or update; only the given fields (nickname, display_name, last_seen) change."""
        bad = set(fields) - {"nickname", "display_name", "last_seen"}
        if bad:
            raise ValueError(f"unknown people fields: {sorted(bad)}")
        cols = ["guild_id", "user_id", *fields]
        updates = ", ".join(f"{k} = excluded.{k}" for k in fields) or "user_id = user_id"
        self._write(f"INSERT INTO people ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
                    f"ON CONFLICT (guild_id, user_id) DO UPDATE SET {updates}",
                    (guild_id, user_id, *fields.values()), changed=("people", guild_id))

    def get_person(self, guild_id: int, user_id: int) -> dict | None:
        rows = self._query("SELECT * FROM people WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
        return self._decode(rows[0]) if rows else None

    def list_people(self, guild_id: int) -> list[dict]:
        return [self._decode(r) for r in self._query(
            "SELECT * FROM people WHERE guild_id = ? ORDER BY COALESCE(nickname, display_name), user_id", (guild_id,))]

    def delete_person(self, guild_id: int, user_id: int) -> None:
        self._write("DELETE FROM people WHERE guild_id = ? AND user_id = ?", (guild_id, user_id),
                    changed=("people", guild_id))

    def nickname_for(self, guild_id: int, user_id: int, display_name: str) -> str:
        """What the bot calls them: their nickname, or their display name
        without the emoji and symbols a voice can't say."""
        person = self.get_person(guild_id, user_id)
        if person and person["nickname"]:
            return person["nickname"]
        return reaction_rules.speakable_name(display_name)

    # ------------------------------------------------------------- consent
    # Consent is per person (global: a voice or words belong to the person, not
    # to a server) and per purpose: "voice" (cloning their voice) and
    # "transcripts" (keeping what they say).
    @staticmethod
    def _check_purpose(purpose: str) -> None:
        if purpose not in CONSENT_PURPOSES:
            raise ValueError(f"purpose must be one of {CONSENT_PURPOSES}, not {purpose!r}")

    def get_consent(self, user_id: int, purpose: str = "voice") -> dict | None:
        self._check_purpose(purpose)
        rows = self._query("SELECT * FROM consent WHERE user_id = ? AND purpose = ?", (user_id, purpose))
        return self._decode(rows[0]) if rows else None

    def set_consent(self, user_id: int, status: str, *, purpose: str = "voice",
                    text_version: str | None = None) -> None:
        """pending stamps asked_at (the question was put to them); anything
        else stamps decided_at."""
        self._check_purpose(purpose)
        if status not in CONSENT_STATUSES:
            raise ValueError(f"status must be one of {CONSENT_STATUSES}")
        stamp = now()
        asked, decided = (stamp, None) if status == "pending" else (None, stamp)
        self._write(
            "INSERT INTO consent (user_id, purpose, status, text_version, asked_at, decided_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (user_id, purpose) DO UPDATE SET status = excluded.status, "
            "text_version = COALESCE(excluded.text_version, text_version), "
            "asked_at = COALESCE(excluded.asked_at, asked_at), decided_at = excluded.decided_at",
            (user_id, purpose, status, None if text_version is None else str(text_version), asked, decided),
            changed=("consent", None))

    def has_consent(self, user_id: int, purpose: str = "voice") -> bool:
        consent = self.get_consent(user_id, purpose)
        return consent is not None and consent["status"] == "accepted"

    def list_consent(self, purpose: str | None = None) -> list[dict]:
        if purpose is None:
            rows = self._query("SELECT * FROM consent ORDER BY user_id, purpose")
        else:
            self._check_purpose(purpose)
            rows = self._query("SELECT * FROM consent WHERE purpose = ? ORDER BY user_id", (purpose,))
        return [self._decode(r) for r in rows]

    # -------------------------------------------------------------- quotas
    def quota_used(self, guild_id: int, user_id: int, resource: str) -> int:
        if resource == "gags":
            sql = ("SELECT COUNT(*) FROM reactions WHERE guild_id = ? AND created_by = ? AND kind = 'gag' "
                   "AND status IN ('pending', 'approved')")
        elif resource == "sounds":
            sql = "SELECT COUNT(*) FROM sounds WHERE guild_id = ? AND created_by = ?"
        elif resource == "voices":
            # Their own speaker voice is free.
            sql = "SELECT COUNT(*) FROM voices WHERE guild_id = ? AND created_by = ? AND kind != 'speaker'"
        else:
            raise ValueError(f"resource must be one of {QUOTA_RESOURCES}, not {resource!r}")
        return self._scalar(sql, (guild_id, user_id))

    def quota_limit(self, guild_id: int, user_id: int, resource: str) -> int:
        override = self._scalar('SELECT "limit" FROM quota_overrides WHERE guild_id = ? AND user_id = ? '
                                "AND resource = ?", (guild_id, user_id, resource))
        if override is not None:
            return override
        return int(self.get_setting(guild_id, f"quota.{resource}", 0))

    def quota(self, guild_id: int, user_id: int, resource: str) -> tuple[int, int]:
        """(used, limit)."""
        return self.quota_used(guild_id, user_id, resource), self.quota_limit(guild_id, user_id, resource)

    def can_create(self, guild_id: int, user_id: int, resource: str, count: int = 1) -> bool:
        used, limit = self.quota(guild_id, user_id, resource)
        return used + count <= limit

    def set_quota_override(self, guild_id: int, user_id: int, resource: str, limit: int | None) -> None:
        """limit None: back to the setting."""
        if resource not in QUOTA_RESOURCES:
            raise ValueError(f"resource must be one of {QUOTA_RESOURCES}")
        if limit is None:
            self._write("DELETE FROM quota_overrides WHERE guild_id = ? AND user_id = ? AND resource = ?",
                        (guild_id, user_id, resource), changed=("quota_overrides", guild_id))
        else:
            self._write('INSERT INTO quota_overrides (guild_id, user_id, resource, "limit") VALUES (?, ?, ?, ?) '
                        'ON CONFLICT (guild_id, user_id, resource) DO UPDATE SET "limit" = excluded."limit"',
                        (guild_id, user_id, resource, int(limit)), changed=("quota_overrides", guild_id))

    def list_quota_overrides(self, guild_id: int) -> list[dict]:
        return [self._decode(r) for r in self._query(
            "SELECT * FROM quota_overrides WHERE guild_id = ? ORDER BY user_id, resource", (guild_id,))]

    def request_quota(self, guild_id: int, user_id: int, resource: str, amount: int = 5, reason: str = "") -> int:
        """File a request for `amount` more. A person has at most one pending
        request per resource: asking again returns that one (updated)."""
        if resource not in QUOTA_RESOURCES:
            raise ValueError(f"resource must be one of {QUOTA_RESOURCES}")
        if int(amount) < 1:
            raise ValueError("amount must be at least 1")
        with self.batch():
            existing = self._scalar("SELECT id FROM quota_requests WHERE guild_id = ? AND user_id = ? "
                                    "AND resource = ? AND status = 'pending'", (guild_id, user_id, resource))
            if existing is not None:
                self._write("UPDATE quota_requests SET amount = ?, reason = ? WHERE id = ?",
                            (int(amount), reason, existing), changed=("quota_requests", guild_id))
                return existing
            return self._write(
                "INSERT INTO quota_requests (guild_id, user_id, resource, amount, reason, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (guild_id, user_id, resource, int(amount), reason, now()), changed=("quota_requests", guild_id),
            ).lastrowid

    def get_quota_request(self, request_id: int) -> dict | None:
        return self._get("quota_requests", request_id)

    def list_quota_requests(self, guild_id: int | None = None, *, status: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM quota_requests WHERE 1", []
        if guild_id is not None:
            sql, params = sql + " AND guild_id = ?", [*params, guild_id]
        if status is not None:
            sql, params = sql + " AND status = ?", [*params, status]
        return [self._decode(r) for r in self._query(sql + " ORDER BY id DESC", params)]

    def decide_quota_request(self, request_id: int, approve: bool, decided_by: int | None,
                             amount: int | None = None) -> dict:
        """Approve (raising their limit by `amount`, default what they asked
        for) or deny. Returns the updated request. ValueError if it was
        already decided."""
        with self.batch():
            request = self.get_quota_request(request_id)
            if request is None:
                raise KeyError(f"no quota request {request_id}")
            if request["status"] != "pending":
                raise ValueError(f"quota request {request_id} is already {request['status']}")
            amount = int(request["amount"] if amount is None else amount)
            g, u, resource = request["guild_id"], request["user_id"], request["resource"]
            if approve:
                self.set_quota_override(g, u, resource, self.quota_limit(g, u, resource) + amount)
            self._write("UPDATE quota_requests SET status = ?, amount = ?, decided_at = ?, decided_by = ? "
                        "WHERE id = ?", ("approved" if approve else "denied", amount, now(), decided_by, request_id),
                        changed=("quota_requests", g))
            return self.get_quota_request(request_id)

    # -------------------------------------------------------------- events
    def log_event(self, guild_id: int | None, outcome: str, *, user_id: int | None = None,
                  user_name: str | None = None, text: str | None = None, reaction_id: int | None = None,
                  voice_id=None, details: dict | None = None, time: str | None = None) -> int:
        """One line of history (feed, stats). No change notification: the
        live feed goes through events.bus. `text` is stored as given: the
        caller passes None unless that person agreed to have their words kept
        (consent "transcripts") in a server with transcripts.enabled."""
        return self._write(
            "INSERT INTO events (time, guild_id, user_id, user_name, text, reaction_id, voice_id, outcome, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time or now(), guild_id, user_id, user_name, text, reaction_id, voice_id, outcome,
             None if details is None else json.dumps(details, ensure_ascii=False)),
        ).lastrowid

    def list_events(self, guild_id: int | None = None, *, limit: int = 100, before_id: int | None = None) -> list[dict]:
        """Newest first. before_id pages back."""
        sql, params = "SELECT * FROM events WHERE 1", []
        if guild_id is not None:
            sql, params = sql + " AND guild_id = ?", [*params, guild_id]
        if before_id is not None:
            sql, params = sql + " AND id < ?", [*params, before_id]
        return [self._decode(r) for r in self._query(sql + " ORDER BY id DESC LIMIT ?", [*params, int(limit)])]

    def delete_events(self, guild_id: int, user_id: int) -> int:
        """Forget one person's history in one server. Returns how many went."""
        return self._write("DELETE FROM events WHERE guild_id = ? AND user_id = ?", (guild_id, user_id)).rowcount

    def delete_user_text(self, user_id: int, guild_id: int | None = None) -> int:
        """Forget what one person said (the rest of their history stays), in
        every server or in one. Returns how many lines lost their text."""
        sql, params = "UPDATE events SET text = NULL WHERE user_id = ? AND text IS NOT NULL", [user_id]
        if guild_id is not None:
            sql, params = sql + " AND guild_id = ?", [*params, guild_id]
        return self._write(sql, params).rowcount

    def prune_events(self, before: str) -> int:
        """Delete events older than `before` (an ISO time). Returns how many."""
        return self._write("DELETE FROM events WHERE time < ?", (before,)).rowcount
