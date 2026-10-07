"""Stats and history routes on a seeded events table (real Store).

    .venv/bin/python -m pytest tests/test_web_stats.py -q
"""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events import EventBus  # noqa: E402
from web.dev_server import FakeController  # noqa: E402
from web.routes_stats import first_id_since  # noqa: E402
from web.server import create_app  # noqa: E402

pytestmark = pytest.mark.asyncio

TOKEN = "stats-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GID = 1100000000000000001
OTHER = 2200000000000000002
ANA, BEN = 111111111111111111, 222222222222222222
G = f"/api/g/{GID}"


def ago(**kw) -> str:
    return (datetime.now().astimezone() - timedelta(**kw)).isoformat(timespec="seconds")


@pytest.fixture
def seeded(store):
    """Reactions + events at known ages. Returns the reaction ids."""
    phrase = lambda w: [{"type": "phrase", "phrases": [w]}]
    say = [[{"type": "say", "text": "x"}]]
    pizza = store.add_reaction(GID, "gag", "pizza", phrase("pizza"), say, created_by=ANA)
    chamba = store.add_reaction(GID, "gag", "chamba", phrase("chamba"), say, cooldown_s=60)
    stop = store.add_reaction(GID, "command", "stop", [{"type": "command", "phrases": ["para"]}],
                              [[{"type": "builtin", "action": "stop"}]])
    store.add_reaction(GID, "gag", "dead gag", phrase("nunca"), say)
    store.add_reaction(GID, "gag", "disabled gag", phrase("off"), say, enabled=False)
    store.add_reaction(GID, "response", "hello", [{"type": "event", "event": "hello"}], say)  # no sentences
    store.set_person(GID, ANA, display_name="Ana R.", nickname="Anita")

    def log(age, user, outcome, reaction=None, text="algo", name="Someone"):
        store.log_event(GID, outcome, user_id=user, user_name=name, text=text, reaction_id=reaction,
                        details={"matched": None, "lang": "es", "duration": 1.0}, time=ago(**age))

    log({"days": 40}, ANA, "matched", pizza, "pizza vieja")         # only in 90d
    log({"days": 3}, BEN, "matched", chamba, "hay chamba", "Ben")    # 7d+
    log({"days": 3}, BEN, "skipped (cooldown)", chamba, "chamba otra vez", "Ben")
    log({"days": 2}, BEN, "skipped (cooldown)", chamba, "más chamba", "Ben")
    log({"hours": 5}, ANA, "matched", pizza, "quiero pizza")         # 24h+
    log({"hours": 4}, ANA, "matched", pizza, "pizza 100% buena")
    log({"hours": 3}, ANA, "no match", None, "hola a todos")
    log({"hours": 2}, ANA, "matched", stop, "bot para")
    log({"hours": 1}, BEN, "skipped (chance)", pizza, "pizza?", "Ben")
    log({"minutes": 30}, BEN, "ignored (echo)", None, "eco", "Ben")
    store.log_event(OTHER, "matched", user_id=ANA, text="otro server", time=ago(minutes=5))
    return {"pizza": pizza, "chamba": chamba, "stop": stop}


@pytest_asyncio.fixture
async def client(aiohttp_client, store):
    bus = EventBus()
    app = create_app(FakeController(bus), bus, token=TOKEN, store=store, status_interval=60)
    return await aiohttp_client(app, headers=AUTH)


async def ok(response, status=200):
    body = await response.json()
    assert response.status == status, body
    return body


# ---------------------------------------------------------------- auth

@pytest.mark.parametrize("method,path", [("GET", f"{G}/stats"), ("GET", f"{G}/history"),
                                         ("DELETE", f"{G}/history/{ANA}")])
async def test_auth(client, method, path):
    assert (await client.request(method, path, headers={"Authorization": ""})).status == 401


async def test_delete_refuses_other_origins(client):
    r = await client.delete(f"{G}/history/{ANA}", headers={"Origin": "https://evil.example"})
    assert r.status == 403


# ---------------------------------------------------------------- the id window

async def test_first_id_since_matches_a_full_scan(store):
    random.seed(7)
    base = datetime.now().astimezone() - timedelta(days=10)
    stamps = sorted(base + timedelta(minutes=random.randint(0, 14000)) for _ in range(300))
    for t in stamps:  # two servers interleaved: ids aren't contiguous per server
        store.log_event(random.choice((GID, OTHER)), "no match", time=t.isoformat(timespec="seconds"))
    rows = store._query("SELECT id, time FROM events WHERE guild_id = ? ORDER BY id", (GID,))
    for days in (0.01, 0.5, 1, 3, 7, 9.99, 11):
        since = datetime.now().astimezone() - timedelta(days=days)
        expected = next((r["id"] for r in rows if datetime.fromisoformat(r["time"]) >= since), None)
        assert first_id_since(store, GID, since) == expected, days
    assert first_id_since(store, 999, datetime.now().astimezone()) is None


# ---------------------------------------------------------------- stats

async def test_stats_ranges_and_totals(client, seeded):
    day = await ok(await client.get(f"{G}/stats?range=24h"))
    t = day["totals"]
    assert t["heard"] == 6 and t["matched"] == 3 and t["no_match"] == 1 and t["echo"] == 1
    assert t["skipped"] == {"skipped (chance)": 1}
    week = await ok(await client.get(f"{G}/stats?range=7d"))
    assert week["totals"]["heard"] == 9 and week["totals"]["skipped"] == {"skipped (chance)": 1, "skipped (cooldown)": 2}
    assert (await ok(await client.get(f"{G}/stats?range=30d")))["totals"]["heard"] == 9
    assert (await ok(await client.get(f"{G}/stats?range=90d")))["totals"]["heard"] == 10  # other server not counted
    assert (await client.get(f"{G}/stats?range=1y")).status == 400

    assert len(day["hours"]) == 24 and sum(h["heard"] for h in day["hours"]) == 6
    assert sum(h["matched"] for h in day["hours"]) == 3


async def test_top_never_fired_and_skips(client, seeded):
    week = await ok(await client.get(f"{G}/stats?range=7d"))
    top = {r["name"]: r for r in week["top_reactions"]}
    assert top["pizza"]["fires"] == 2 and top["pizza"]["creator"] == "Anita" and top["pizza"]["kind"] == "gag"
    assert top["chamba"]["fires"] == 1 and top["chamba"]["creator"] == "admin"
    assert [r["name"] for r in week["top_reactions"]][0] == "pizza"

    never = [r["name"] for r in week["never_fired"]]
    assert "dead gag" in never and "disabled gag" not in never and "hello" not in never and "pizza" not in never

    skips = {r["name"]: r for r in week["skipped_reactions"]}
    assert skips["chamba"]["skips"] == {"skipped (cooldown)": 2} and skips["chamba"]["skip_share"] == pytest.approx(2 / 3, abs=0.001)
    assert skips["pizza"]["skips"] == {"skipped (chance)": 1}


async def test_people(client, seeded):
    week = await ok(await client.get(f"{G}/stats?range=7d"))
    people = {p["name"]: p for p in week["people"]}
    assert set(people) == {"Anita", "Ben"}  # nickname wins; else the name the events saw
    anita = people["Anita"]
    assert anita["user_id"] == str(ANA) and anita["heard"] == 4 and anita["matched"] == 3 and anita["gags"] == 2
    assert anita["top_gag"] == {"name": "pizza", "count": 2}
    assert people["Ben"]["gags"] == 1 and people["Ben"]["top_gag"]["name"] == "chamba"


async def test_empty_server(client):
    out = await ok(await client.get("/api/g/42/stats?range=24h"))
    assert out["totals"]["heard"] == 0 and out["people"] == [] and out["top_reactions"] == []


async def test_deleted_reaction_still_counted(client, store, seeded):
    store.delete_reaction(seeded["chamba"])
    week = await ok(await client.get(f"{G}/stats?range=7d"))
    assert any(r.get("deleted") for r in week["top_reactions"])


# ---------------------------------------------------------------- history

async def test_history_paging(client, seeded):
    first = await ok(await client.get(f"{G}/history?limit=4"))
    assert [e["text"] for e in first["events"]][:2] == ["eco", "pizza?"]  # newest first
    assert {p["name"] for p in first["people"]} == {"Anita", "Ben"}
    assert "matched" in first["outcomes"]
    seen, page = list(first["events"]), first
    while page["next_before_id"]:
        page = await ok(await client.get(f"{G}/history?limit=4&before_id={page['next_before_id']}"))
        assert "people" not in page
        seen += page["events"]
    assert len(seen) == 10 and len({e["id"] for e in seen}) == 10
    assert [e["id"] for e in seen] == sorted((e["id"] for e in seen), reverse=True)
    stop = next(e for e in seen if e["text"] == "bot para")
    assert stop["reaction"] == "stop" and stop["kind"] == "command" and stop["user"] == "Anita"


async def test_history_filters(client, seeded):
    ben = await ok(await client.get(f"{G}/history?user_id={BEN}"))
    assert len(ben["events"]) == 5 and all(e["user_id"] == str(BEN) for e in ben["events"])
    cool = await ok(await client.get(f"{G}/history?outcome=skipped%20(cooldown)"))
    assert len(cool["events"]) == 2
    pizza = await ok(await client.get(f"{G}/history?q=PIZZA"))
    assert len(pizza["events"]) == 4  # case-insensitive; the other server's isn't included
    pct = await ok(await client.get(f"{G}/history?q=100%25"))
    assert [e["text"] for e in pct["events"]] == ["pizza 100% buena"]  # % is literal
    both = await ok(await client.get(f"{G}/history?user_id={ANA}&outcome=matched&q=pizza"))
    assert len(both["events"]) == 3
    assert (await client.get(f"{G}/history?user_id=abc")).status == 400
    assert (await client.get(f"{G}/history?limit=x")).status == 400


async def test_text_not_saved(client, store):
    store.log_event(GID, "no match", user_id=ANA, text=None)  # no consent to keep their words
    out = await ok(await client.get(f"{G}/history"))
    assert out["events"][0]["text"] is None


async def test_delete_a_persons_history(client, store, seeded):
    out = await ok(await client.delete(f"{G}/history/{ANA}"))
    assert out["deleted"] == 5
    left = await ok(await client.get(f"{G}/history?limit=100"))
    assert all(e["user_id"] == str(BEN) for e in left["events"]) and len(left["events"]) == 5
    # their history in another server is untouched
    assert store._query("SELECT COUNT(*) FROM events WHERE guild_id = ? AND user_id = ?", (OTHER, ANA))[0][0] == 1
    assert (await ok(await client.delete(f"{G}/history/{ANA}")))["deleted"] == 0
