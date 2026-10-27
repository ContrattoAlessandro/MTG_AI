"""`mz lan`: the lobby, the game file, the session's seats and the HTTP routes (with a fake engine)."""

import threading
import time

import pytest

from magezero import webplay
from magezero.tabletop.app import EXTENSIONS_MARK, create_app
from magezero.tabletop.feed import ChangeFeed
from magezero.tabletop.lobby import DeckShelf, Lobby, SeatKind
from magezero.tabletop.push import SEAT_COOKIE, delta, seat_token
from magezero.tabletop.session import TableSession, game_file, playing_message


def write_deck(folder, name, commander=None, cards=60):
    lines = [f"{cards - (1 if commander else 0)} [SNC:280] Forest"]
    if commander:
        lines.append(f"SB: 1 [IKO:216] {commander}")
    (folder / f"{name}.dck").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def shelf(tmp_path):
    write_deck(tmp_path, "winota", "Winota, Joiner of Forces", 100)
    write_deck(tmp_path, "kinnan", "Kinnan, Bonder Prodigy", 100)
    write_deck(tmp_path, "burn")
    return DeckShelf(tmp_path)


@pytest.fixture
def feed():
    return ChangeFeed()


@pytest.fixture
def lobby(shelf, feed):
    return Lobby(shelf, feed.bump)


def test_the_feed_wakes_a_waiting_reader(feed):
    seen = feed.version
    threading.Timer(0.05, feed.bump).start()
    assert feed.wait(seen, timeout=5) == seen + 1
    assert feed.wait(seen + 1, timeout=0.01) == seen + 1  # nothing new: returns after the timeout


def test_people_take_open_seats_and_keep_their_token(lobby):
    ann = lobby.join("Ann")
    bob = lobby.join("  Bob  ", index=2)
    assert lobby.seat_of(ann).index == 0 and lobby.seat_of(bob).name == "Bob"
    assert lobby.join("Annie", token=ann) == ann  # a rename keeps the seat
    assert lobby.seat_of(ann).name == "Annie"
    with pytest.raises(ValueError, match="already at the table"):
        lobby.join("bob")
    with pytest.raises(ValueError, match="sits there"):
        lobby.join("Cid", index=2)
    view = lobby.view(bob)
    assert view["you"] == 2
    assert "token" not in str(view["seats"])  # nobody sees another person's secret


def test_a_person_may_replace_a_bot(lobby):
    assert lobby.seats[3].kind == SeatKind.BOT
    token = lobby.join("Dee", index=3)
    assert lobby.seat_of(token).kind == SeatKind.PERSON


def test_only_seated_people_change_the_table(lobby):
    with pytest.raises(ValueError, match="Take a seat"):
        lobby.set_seat(None, 1, "bot")
    ann = lobby.join("Ann")
    lobby.set_seat(ann, 1, "bot", skill=5)
    assert lobby.seats[1].kind == SeatKind.BOT and lobby.seats[1].skill == 5
    with pytest.raises(ValueError, match="sits there"):
        lobby.set_seat(ann, 0, "closed")
    with pytest.raises(ValueError, match="Choose a deck for your seat or for a bot"):
        lobby.choose_deck(lobby.join("Bob", index=2), 0, "burn")


def test_the_format_follows_the_decks_and_the_number_of_players(lobby):
    ann = lobby.join("Ann")
    lobby.choose_deck(ann, 0, "winota")
    lobby.set_seat(ann, 1, "closed")
    lobby.set_seat(ann, 2, "closed")
    lobby.choose_deck(ann, 3, "kinnan")
    assert lobby.view(ann)["format"] == "commander-duel"
    lobby.set_seat(ann, 2, "bot")
    lobby.choose_deck(ann, 2, "kinnan")
    assert lobby.view(ann)["format"] == "commander-free-for-all"
    lobby.choose_deck(ann, 2, "burn")
    assert lobby.view(ann)["format"] is None
    assert "cannot play together" in " ".join(lobby.problems())


def test_a_game_needs_every_seat_settled(lobby):
    ann = lobby.join("Ann")
    assert any("open seats" in problem for problem in lobby.problems())
    lobby.set_seat(ann, 1, "closed")
    lobby.set_seat(ann, 2, "closed")
    assert lobby.problems() == []
    with pytest.raises(ValueError, match="Nobody plays"):
        lobby.choose_first(ann, 1)


def test_the_plan_lists_the_players_in_turn_order_from_the_first(lobby, shelf):
    ann, bob = lobby.join("Ann"), lobby.join("Bob")
    lobby.set_seat(ann, 2, "bot", skill=2)
    lobby.choose_first(ann, 2)
    plan = lobby.plan()
    assert [seat.name for seat in plan.seats] == ["Bot 3", "Bot 4", "Ann", "Bob"]
    spec = game_file(plan, shelf, "http://127.0.0.1:8765/engine")
    # Everyone has the first deck of the shelf until they choose: burn, 60 cards.
    assert spec["format"] == "free-for-all" and spec["goes_first"] == 0
    assert spec["seats"][0] == {"name": "Bot 3", "deckPath": str(shelf.path("burn")), "type": "xmage-ai", "skill": 2}
    assert spec["seats"][2]["type"] == "web"
    assert spec["seats"][3]["endpoint"] == "http://127.0.0.1:8765/engine/seat/1"
    assert bob


def test_the_seats_are_locked_during_a_game(lobby):
    ann = lobby.join("Ann")
    lobby.lock_for_game(True)
    with pytest.raises(ValueError, match="in progress"):
        lobby.set_seat(ann, 1, "bot")
    with pytest.raises(ValueError, match="in progress"):
        lobby.join("Bob")


def test_seat_token_reads_the_cookie():
    assert seat_token(f"theme=dark; {SEAT_COOKIE}=abc123") == "abc123"
    assert seat_token("theme=dark") is None
    assert seat_token(None) is None


def test_playing_message_names_whose_move_it_is():
    state = {
        "turn": 4,
        "priority_id": "b",
        "players": [{"id": "a", "side": "you", "name": "Ann"}, {"id": "b", "side": "opponent", "name": "Bob"}],
    }
    assert playing_message({"state": state, "decision": None}) == "Turn 4: Bob is playing"
    assert playing_message({"state": state, "decision": {"kind": "cast"}}) == "Turn 4: your move"
    state["players"][0]["lost"] = True
    assert playing_message({"state": state, "decision": None}).startswith("You are out")


class FakeEngine:
    """Plays a table game file by asking each person's seat once, as the engine relays decisions."""

    def __init__(self):
        self.files = []
        self.session = None
        self.release = threading.Event()

    def run(self, path):
        self.files.append(path)
        self.release.wait(5)
        return (1, 1, 0, 0, 0)

    def kill(self):
        self.release.set()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def session(lobby, feed, engine, tmp_path):
    session = TableSession(lobby, feed, lambda: engine, tmp_path, "http://127.0.0.1:1/engine")
    session.boot()
    yield session
    engine.release.set()


def seated(lobby):
    ann, bob = lobby.join("Ann"), lobby.join("Bob")
    lobby.set_seat(ann, 2, "bot")
    return ann, bob


def test_a_game_gives_each_person_their_own_decision_table(session, lobby, engine, monkeypatch):
    monkeypatch.setattr(webplay, "HOLD_SECONDS", 0.05)
    ann, bob = seated(lobby)
    session.start(ann)
    assert session.status == "playing" and lobby.locked
    assert set(session.tables) == {0, 1}
    assert session.table_of(ann) is session.engine_table(0) and session.table_of(bob) is session.engine_table(1)
    with pytest.raises(KeyError):
        session.engine_table(2)  # a bot's seat has no table
    state = {"turn": 1, "log": [], "players": []}
    options = [{"id": "p", "type": "pass", "label": "Pass"}, {"id": "c", "type": "cast", "label": "Cast"}]
    decision = {"decision": "d1", "kind": "cast", "prompt": "Your turn", "state": state, "options": options}
    assert session.engine_table(1).decide(decision) == {"wait": True}
    assert session.view(bob)["decision"]["decision"] == "d1"
    assert session.view(ann)["decision"] is None  # Ann does not see Bob's choice
    engine.release.set()
    deadline = time.monotonic() + 5
    while session.status == "playing" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.status == "over" and not lobby.locked


def test_starting_needs_a_seat_and_a_ready_table(session, lobby):
    with pytest.raises(ValueError, match="Take a seat"):
        session.start(None)
    ann = lobby.join("Ann")
    with pytest.raises(ValueError, match="open seats"):
        session.start(ann)


def test_abandoning_stops_every_seat(session, lobby, engine):
    ann, _ = seated(lobby)
    session.start(ann)
    tables = list(session.tables.values())
    engine.release.set()  # the fake engine returns once asked to stop
    session.abandon(ann)
    assert all(table.abort for table in tables)
    assert session.status == "ready" and not lobby.locked


def client_for(session):
    app = create_app(session, 8766)
    app.testing = True
    return app.test_client()


def test_the_page_loads_the_table_script(session, monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text(f"<html>{EXTENSIONS_MARK}<script></script></html>", encoding="utf-8")
    monkeypatch.setattr("magezero.tabletop.app.WEB_DIR", tmp_path)
    page = client_for(session).get("/").get_data(as_text=True)
    assert '<script src="/lan.js"></script>' in page


def test_joining_sets_the_seat_cookie_and_the_view_follows_it(session):
    client = client_for(session)
    assert client.get("/api/config").get_json() == {"mode": "lan", "push_port": 8766, "share_url": None}
    reply = client.post("/api/lobby/join", json={"name": "Ann"})
    assert reply.status_code == 200 and SEAT_COOKIE in reply.headers["Set-Cookie"]
    view = client.get("/api/state").get_json()
    assert view["mode"] == "lan" and view["lobby"]["you"] == 0
    assert client.post("/api/lobby/seat", json={"seat": 1, "kind": "bot", "skill": 2}).status_code == 200
    refused = client.post("/api/lobby/seat", json={"seat": 0, "kind": "closed"})
    assert refused.get_json()["error"] == "Ann sits there."
    stranger = client_for(session)
    assert stranger.get("/api/state").get_json()["lobby"]["you"] is None
    assert stranger.post("/api/answer", json={"decision": "d", "action_id": "x"}).status_code == 400


def test_the_engine_routes_answer_this_computer_only(session):
    client = client_for(session)
    remote = client.post("/engine/seat/0/decide", json={}, environ_base={"REMOTE_ADDR": "192.168.1.20"})
    assert remote.status_code == 403
    assert client.post("/engine/seat/0/decide", json={}).status_code == 404  # no game: no seat


def patched(base, change):
    """The browser's side of `delta` (lan.js: patch)."""
    view = {**base, **change.get("set", {})}
    if "state" in change:
        view["state"] = change["state"]
    elif "patch" in change or "log" in change:
        state = {**base["state"], **change.get("patch", {})}
        if "log" in change:
            state["log"] = base["state"]["log"][change["log"]["drop"] :] + change["log"]["add"]
        view["state"] = state
    return view


def test_a_push_carries_only_what_changed():
    lines = [f"line {i}" for i in range(400)]
    before = {"version": 1, "status": "playing", "lobby": {"seats": []},
              "state": {"turn": 3, "stack": [], "players": [{"life": 40}], "log": lines}}  # fmt: skip
    after = {**before, "version": 2,
             "state": {**before["state"], "stack": [{"id": "s"}], "log": lines[2:] + ["new 1", "new 2"]}}  # fmt: skip
    change = delta(before, after)
    assert change == {"set": {"version": 2}, "patch": {"stack": [{"id": "s"}]},
                      "log": {"drop": 2, "add": ["new 1", "new 2"]}}  # fmt: skip
    assert patched(before, change) == after
    # Only the version moved (another seat's change): nothing to send.
    assert delta(after, {**after, "version": 3}) is None


def test_a_push_sends_the_whole_state_when_it_appears_or_goes():
    lobby_only = {"version": 1, "status": "ready", "state": None, "decision": None}
    playing = {"version": 2, "status": "playing", "state": {"turn": 1, "log": ["a"]}, "decision": {"id": "d"}}
    for before, after in ((lobby_only, playing), (playing, lobby_only)):
        assert patched(before, delta(before, after)) == after
    # A log that starts over (a new game) is replaced whole.
    later = {**playing, "version": 3, "state": {"turn": 1, "log": ["x", "y"]}}
    assert delta(playing, later)["log"] == {"drop": 1, "add": ["x", "y"]}
    assert patched(playing, delta(playing, later)) == later
