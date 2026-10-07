import sqlite3
import threading

import pytest

import store as store_module
from store import DEFAULT_SETTINGS, MIGRATIONS, Store

SAY = [[{"type": "say", "text": "hola"}]]
PHRASE = [{"type": "phrase", "phrases": ["hola"]}]


# ------------------------------------------------------------ migrations
def test_fresh_database_runs_every_migration(tmp_path):
    path = tmp_path / "sub" / "bot.db"
    s = Store(path)
    assert s.schema_version() == len(MIGRATIONS)
    tables = {r[0] for r in s._query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"schema_version", "settings", "people", "consent", "voices", "sounds", "reactions",
            "quota_overrides", "quota_requests", "events"} <= tables
    assert s._scalar("PRAGMA journal_mode") == "wal"
    s.add_reaction(1, "gag", "hola", PHRASE, SAY)
    s.close()

    again = Store(path)  # reopening runs nothing twice and keeps the data
    assert again.schema_version() == len(MIGRATIONS)
    assert len(again.list_reactions(1)) == 1
    assert again._scalar("SELECT COUNT(*) FROM schema_version") == 1
    again.close()


def test_new_migrations_apply_on_open(tmp_path, monkeypatch):
    path = tmp_path / "bot.db"
    Store(path).close()
    monkeypatch.setattr(store_module, "MIGRATIONS", [*MIGRATIONS, "ALTER TABLE sounds ADD COLUMN tags TEXT;"])
    s = Store(path)
    assert s.schema_version() == len(MIGRATIONS) + 1
    assert "tags" in {r[1] for r in s._query("PRAGMA table_info(sounds)")}
    s.close()


def test_a_newer_database_is_refused(tmp_path):
    path = tmp_path / "bot.db"
    Store(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_version SET version = 999")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        Store(path)


# ------------------------------------------------------------- settings
def test_settings_precedence(store):
    assert store.get_setting(5, "quota.gags") == DEFAULT_SETTINGS["quota.gags"] == 15
    assert store.get_setting(5, "nonexistent", "fallback") == "fallback"

    store.set_setting(0, "quota.gags", 20)  # global
    assert store.get_setting(5, "quota.gags") == 20
    assert store.get_setting(6, "quota.gags") == 20

    store.set_setting(5, "quota.gags", 7)  # this server only
    assert store.get_setting(5, "quota.gags") == 7
    assert store.get_setting(6, "quota.gags") == 20

    store.set_setting(5, "bot.wake_words", ["robo", "robot"])  # any JSON value
    assert store.get_setting(5, "bot.wake_words") == ["robo", "robot"]

    effective = store.settings(5)
    assert effective["quota.gags"] == 7 and effective["quota.sounds"] == 15
    assert store.settings(5, effective=False) == {"quota.gags": 7, "bot.wake_words": ["robo", "robot"]}

    store.delete_setting(5, "quota.gags")
    assert store.get_setting(5, "quota.gags") == 20
    store.delete_setting(0, "quota.gags")
    assert store.get_setting(5, "quota.gags") == 15


def test_a_setting_can_be_set_to_null_over_the_global(store):
    store.set_setting(0, "voice.default", 3)
    store.set_setting(5, "voice.default", None)
    assert store.get_setting(5, "voice.default") is None
    assert store.get_setting(6, "voice.default") == 3


# --------------------------------------------------------------- quotas
def test_quota_counts_and_default_limits(store):
    g, u = 1, 42
    assert store.quota(g, u, "gags") == (0, 15)
    assert store.quota(g, u, "sounds") == (0, 15)
    assert store.quota(g, u, "voices") == (0, 5)

    store.add_reaction(g, "gag", "a", PHRASE, SAY, created_by=u)
    store.add_reaction(g, "gag", "b", PHRASE, SAY, created_by=u, status="pending")
    store.add_reaction(g, "gag", "c", PHRASE, SAY, created_by=u, status="rejected")  # doesn't count
    store.add_reaction(g, "response", "d", [{"type": "event", "event": "hello"}], SAY, created_by=u)  # not a gag
    store.add_reaction(g, "gag", "e", PHRASE, SAY)  # admin's
    store.add_reaction(2, "gag", "f", PHRASE, SAY, created_by=u)  # other server
    assert store.quota(g, u, "gags") == (2, 15)

    store.add_sound(g, "bruh", "a.ogg", created_by=u)
    assert store.quota(g, u, "sounds") == (1, 15)

    store.add_voice("mine", guild_id=g, kind="speaker", owner_user_id=u, created_by=u)  # free
    store.add_voice("pirate", guild_id=g, kind="designed", created_by=u)
    assert store.quota(g, u, "voices") == (1, 5)

    with pytest.raises(ValueError):
        store.quota(g, u, "nonsense")


def test_can_create_and_limits(store):
    g, u = 1, 42
    store.set_setting(g, "quota.gags", 2)
    assert store.can_create(g, u, "gags")
    store.add_reaction(g, "gag", "a", PHRASE, SAY, created_by=u)
    assert store.can_create(g, u, "gags")
    assert not store.can_create(g, u, "gags", count=2)
    store.add_reaction(g, "gag", "b", PHRASE, SAY, created_by=u)
    assert not store.can_create(g, u, "gags")

    store.set_quota_override(g, u, "gags", 3)  # an override beats the setting
    assert store.quota(g, u, "gags") == (2, 3)
    assert store.can_create(g, u, "gags")
    store.set_quota_override(g, u, "gags", None)
    assert store.quota(g, u, "gags") == (2, 2)


def test_quota_requests(store):
    g, u, admin = 1, 42, 7
    store.set_setting(g, "quota.sounds", 1)

    first = store.request_quota(g, u, "sounds", 3, "more memes")
    again = store.request_quota(g, u, "sounds", 4, "please")  # one pending per resource
    assert first == again
    request = store.get_quota_request(first)
    assert request["status"] == "pending" and request["amount"] == 4 and request["reason"] == "please"
    assert [r["id"] for r in store.list_quota_requests(g, status="pending")] == [first]

    decided = store.decide_quota_request(first, True, admin)
    assert decided["status"] == "approved" and decided["decided_by"] == admin and decided["decided_at"]
    assert store.quota(g, u, "sounds") == (0, 5)  # 1 + 4
    with pytest.raises(ValueError, match="already"):
        store.decide_quota_request(first, False, admin)

    second = store.request_quota(g, u, "sounds", 10)
    assert second != first
    store.decide_quota_request(second, True, admin, amount=2)  # the admin grants less
    assert store.quota(g, u, "sounds") == (0, 7)
    assert store.get_quota_request(second)["amount"] == 2

    third = store.request_quota(g, u, "sounds", 10)
    assert store.decide_quota_request(third, False, admin)["status"] == "denied"
    assert store.quota(g, u, "sounds") == (0, 7)

    with pytest.raises(ValueError):
        store.request_quota(g, u, "nonsense")
    with pytest.raises(KeyError):
        store.decide_quota_request(9999, True, admin)


# ----------------------------------------------------------------- CRUD
def test_reactions_crud_and_order(store):
    a = store.add_reaction(1, "gag", "a", PHRASE, SAY)
    b = store.add_reaction(1, "gag", "b", PHRASE, SAY, by_users=[123, "Pepe"], voice_id="@speaker")
    c = store.add_reaction(1, "command", "c", [{"type": "command", "phrases": ["vete"]}],
                           [[{"type": "builtin", "action": "leave"}]])
    row = store.get_reaction(b)
    assert row["by_users"] == [123, "Pepe"] and row["voice_id"] == "@speaker"
    assert row["enabled"] is True and row["triggers"] == PHRASE and isinstance(row["id"], int)
    assert [r["id"] for r in store.list_reactions(1)] == [a, b, c]
    assert [r["id"] for r in store.list_reactions(1, kind="gag")] == [a, b]

    store.reorder_reactions(1, [c, a])
    assert [r["id"] for r in store.list_reactions(1)] == [c, a, b]

    store.update_reaction(a, enabled=False, chance=0.5)
    assert store.get_reaction(a)["enabled"] is False and store.get_reaction(a)["chance"] == 0.5
    with pytest.raises(ValueError):
        store.update_reaction(a, options=[])
    with pytest.raises(ValueError):
        store.add_reaction(1, "gag", "bad", [{"type": "phrase"}], SAY)
    with pytest.raises(ValueError):
        store.add_reaction(1, "nonsense", "bad", PHRASE, SAY)
    with pytest.raises(ValueError):
        store.add_reaction(1, "gag", "bad", PHRASE, [[{"type": "builtin", "action": "explode"}]])

    store.record_use(b)
    store.record_use(b)
    assert store.get_reaction(b)["uses"] == 2 and store.get_reaction(b)["last_used_at"]

    store.delete_reaction(a)
    assert store.get_reaction(a) is None


def test_deleting_a_voice_or_sound_scrubs_references(store):
    v = store.add_voice("pirate", guild_id=1, kind="designed", instruct="arr")
    snd = store.add_sound(1, "bruh", "bruh.ogg")
    r = store.add_reaction(1, "gag", "skit", PHRASE, [
        [{"type": "say", "text": "hola", "voice_id": v}, {"type": "sound", "sound_id": snd}],
        [{"type": "sound", "sound_id": snd}],
    ], voice_id=v)
    store.delete_voice(v)
    row = store.get_reaction(r)
    assert row["voice_id"] is None and row["options"][0][0]["voice_id"] is None
    store.delete_sound(snd)
    assert store.get_reaction(r)["options"] == [[{"type": "say", "text": "hola", "voice_id": None}]]


def test_voices_and_sounds(store):
    bot_voice = store.add_voice("Bot", kind="clone")  # global
    own = store.add_voice("pirate", guild_id=1, kind="designed", tags=["fun"], status="ready")
    assert store.get_voice(own)["tags"] == ["fun"]
    assert store.find_voice(1, "bot")["id"] == bot_voice
    assert store.find_voice(1, "PIRATE")["id"] == own
    assert store.find_voice(2, "pirate") is None
    assert [v["id"] for v in store.list_voices(1)] == [own, bot_voice]
    assert [v["id"] for v in store.list_voices(1, include_global=False)] == [own]
    store.update_voice(own, status="failed", error="out of memory")
    assert store.get_voice(own)["error"] == "out of memory"
    with pytest.raises(ValueError):
        store.update_voice(own, status="nonsense")

    s = store.add_sound(1, "bruh", "data/bruh.ogg", duration_s=1.5)
    assert store.find_sound(1, "BRUH")["duration_s"] == 1.5
    with pytest.raises(sqlite3.IntegrityError):
        store.add_sound(1, "bruh", "other.ogg")  # names are unique per server
    store.update_sound(s, enabled=False)
    assert store.get_sound(s)["enabled"] is False


def test_people_and_nicknames(store):
    store.set_person(1, 42, display_name="Pepe 🎉")
    assert store.nickname_for(1, 42, "Pepe 🎉") == "Pepe"  # no nickname: display name, speakable
    store.set_person(1, 42, nickname="Pepito")
    person = store.get_person(1, 42)
    assert person["nickname"] == "Pepito" and person["display_name"] == "Pepe 🎉"  # untouched
    assert store.nickname_for(1, 42, "whatever") == "Pepito"
    assert store.nickname_for(2, 42, "🎉") == "🎉"  # nothing speakable: as is
    assert [p["user_id"] for p in store.list_people(1)] == [42]
    store.delete_person(1, 42)
    assert store.get_person(1, 42) is None


def test_consent(store):
    assert not store.has_consent(42)
    store.set_consent(42, "pending", text_version="v1")
    c = store.get_consent(42)
    assert c["status"] == "pending" and c["purpose"] == "voice" and c["asked_at"] and c["decided_at"] is None
    store.set_consent(42, "accepted")
    c2 = store.get_consent(42)
    assert c2["status"] == "accepted" and c2["asked_at"] == c["asked_at"] and c2["text_version"] == "v1"
    assert store.has_consent(42)
    store.set_consent(42, "revoked")
    assert not store.has_consent(42)
    with pytest.raises(ValueError):
        store.set_consent(42, "maybe")


def test_consent_per_purpose(store):
    store.set_consent(42, "accepted")  # voice, the default
    assert store.has_consent(42, "voice") and not store.has_consent(42, "transcripts")
    store.set_consent(42, "accepted", purpose="transcripts", text_version=2)
    store.set_consent(42, "revoked", purpose="voice")
    assert store.has_consent(42, purpose="transcripts") and not store.has_consent(42)
    assert store.get_consent(42, "transcripts")["text_version"] == "2"
    store.set_consent(43, "declined", purpose="transcripts")
    assert [(c["user_id"], c["purpose"]) for c in store.list_consent()] == [
        (42, "transcripts"), (42, "voice"), (43, "transcripts")]
    assert [c["user_id"] for c in store.list_consent("transcripts")] == [42, 43]
    with pytest.raises(ValueError):
        store.set_consent(42, "accepted", purpose="selfies")
    with pytest.raises(ValueError):
        store.has_consent(42, "selfies")


def test_migration_3_keeps_voice_consent(tmp_path, monkeypatch):
    path = tmp_path / "bot.db"
    monkeypatch.setattr(store_module, "MIGRATIONS", MIGRATIONS[:2])  # a database from before migration 3
    old = Store(path)
    old._write("INSERT INTO consent (user_id, status, text_version, asked_at, decided_at) "
               "VALUES (42, 'accepted', '1', 'a', 'b'), (43, 'declined', NULL, NULL, 'c')")
    old.close()
    monkeypatch.setattr(store_module, "MIGRATIONS", MIGRATIONS)
    new = Store(path)
    assert new.schema_version() == len(MIGRATIONS)
    assert new.has_consent(42) and not new.has_consent(42, "transcripts")
    assert new.get_consent(43)["status"] == "declined" and new.get_consent(43)["decided_at"] == "c"
    new.close()


def test_events_log(store):
    first = store.log_event(1, "played", user_id=42, user_name="Pepe", text="hola", details={"ms": 300})
    second = store.log_event(1, "no match", user_id=42, text=None)
    store.log_event(2, "played")
    events = store.list_events(1)
    assert [e["id"] for e in events] == [second, first]
    assert events[0]["text"] is None and events[1]["text"] == "hola" and events[1]["details"] == {"ms": 300}
    assert [e["id"] for e in store.list_events(1, before_id=second)] == [first]
    assert store.prune_events("9999") == 3


def test_delete_user_text(store):
    store.log_event(1, "played", user_id=42, text="uno")
    store.log_event(2, "played", user_id=42, text="dos")
    store.log_event(1, "played", user_id=43, text="tres")
    assert store.delete_user_text(42, guild_id=1) == 1
    assert [e["text"] for e in store.list_events(2)] == ["dos"]
    assert store.delete_user_text(42) == 1
    assert [e["text"] for e in store.list_events()] == ["tres", None, None]
    assert len(store.list_events()) == 3  # the history stays, without the words


def test_transcript_settings_default_off(store):
    assert store.get_setting(1, "transcripts.enabled") is False
    assert store.get_setting(1, "language") is None
    assert "events.save_text" not in store.settings(1)


# ---------------------------------------------------- notifications, batches
def test_change_notifications(store):
    heard = []
    store.on_change(lambda table, guild: heard.append((table, guild)))
    r = store.add_reaction(1, "gag", "a", PHRASE, SAY)
    store.set_setting(0, "quota.gags", 3)
    store.set_setting(2, "quota.gags", 3)
    store.add_voice("global")
    store.record_use(r)  # stats: no notification
    assert heard == [("reactions", 1), ("settings", None), ("settings", 2), ("voices", None)]

    heard.clear()
    with store.batch():
        store.add_reaction(1, "gag", "b", PHRASE, SAY)
        store.add_reaction(1, "gag", "c", PHRASE, SAY)
        assert heard == []  # only once it's committed
    assert heard == [("reactions", 1)]


def test_batch_rolls_back_everything(store):
    heard = []
    store.on_change(lambda *a: heard.append(a))
    with pytest.raises(ValueError):
        with store.batch():
            store.add_reaction(1, "gag", "a", PHRASE, SAY)
            store.add_reaction(1, "gag", "bad", [], SAY)
    assert store.list_reactions(1) == [] and heard == []


def test_a_failing_listener_does_not_break_writes(store):
    store.on_change(lambda *a: 1 / 0)
    store.add_reaction(1, "gag", "a", PHRASE, SAY)
    assert len(store.list_reactions(1)) == 1


def test_threads_share_one_store(store):
    def work(n):
        for i in range(25):
            store.add_reaction(n, "gag", f"{n}-{i}", PHRASE, SAY)
            store.list_reactions(n)
            store.log_event(n, "played")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(1, 5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(len(store.list_reactions(n)) == 25 for n in range(1, 5))
    assert len(store.list_events(limit=1000)) == 100


def test_timers(store):
    a = store.add_timer(1, 5, "Ana", "10 minutes", "pizza", 2000.0)
    b = store.add_timer(1, 6, "Bo", "5 PM", "", 1000.0)
    store.add_timer(2, 5, "Ana", "1 hour", "", 3000.0)
    assert [t["id"] for t in store.list_timers(1)] == [b, a]  # soonest first
    assert [t["guild_id"] for t in store.list_timers()] == [1, 1, 2]
    assert [t["said"] for t in store.list_timers(1, user_id=5)] == ["10 minutes"]
    assert store.get_timer(a)["message"] == "pizza" and store.get_timer(a)["ends_at"] == 2000.0
    assert store.delete_timer(a) and not store.delete_timer(a)
    assert store.get_timer(a) is None
