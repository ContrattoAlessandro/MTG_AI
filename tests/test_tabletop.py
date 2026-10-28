"""`mz lan`: the lobby, the game file, the session's seats and the HTTP routes (with a fake engine)."""

import threading
import time

import pytest

from magezero import webplay
from magezero.tabletop.app import EXTENSIONS_MARK, create_app
from magezero.tabletop.feed import ChangeFeed
from magezero.tabletop.lobby import DeckShelf, Lobby, SeatKind
from magezero.tabletop.paste import CardNames, DeckPaster, parse
from magezero.tabletop.push import delta, seat_cookie, seat_token
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
    spec = game_file(plan, shelf.path, "http://127.0.0.1:8765/engine")
    # Everyone has the first deck of the shelf until they choose: burn, 60 cards.
    assert spec["format"] == "free-for-all" and spec["goes_first"] == 0
    # A bot's board goes to its seat's endpoint too, for whoever watches it.
    assert spec["seats"][0] == {"name": "Bot 3", "deckPath": str(shelf.path("burn")), "type": "xmage-ai", "skill": 2,
                                "endpoint": "http://127.0.0.1:8765/engine/seat/2"}  # fmt: skip
    assert spec["seats"][2]["type"] == "web"
    assert spec["seats"][3]["endpoint"] == "http://127.0.0.1:8765/engine/seat/1"
    assert bob


def test_the_seats_are_locked_during_a_game(lobby):
    ann = lobby.join("Ann")
    lobby.lock_for_game(True)
    with pytest.raises(ValueError, match="in progress"):
        lobby.set_seat(ann, 1, "bot")
    with pytest.raises(ValueError, match="in progress"):
        lobby.join("Annie", token=ann)  # no renaming during a game
    bob = lobby.join("Bob")  # someone arriving watches
    assert lobby.seat_of(bob) is None and lobby.watching(bob)
    assert lobby.view(bob)["watcher"] == "Bob" and lobby.view(ann)["watchers"] == ["Bob"]
    with pytest.raises(ValueError, match="already at the table"):
        lobby.join("ann")
    lobby.lock_for_game(False)
    assert lobby.join("Bob", token=bob) == bob  # and sits down for the next game with the same token
    assert lobby.seat_of(bob).name == "Bob" and not lobby.watching(bob)


def test_seat_token_reads_the_cookie_of_its_own_table():
    assert seat_token(f"theme=dark; {seat_cookie(8766)}=abc123", seat_cookie(8766)) == "abc123"
    # Another table on the same computer has its own seat cookie.
    assert seat_token(f"{seat_cookie(8796)}=abc123", seat_cookie(8766)) is None
    assert seat_token("theme=dark", seat_cookie(8766)) is None
    assert seat_token(None, seat_cookie(8766)) is None


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
    assert session.engine_table(2) is session.eyes[2]  # a bot's seat only receives its board
    with pytest.raises(KeyError):
        session.engine_table(1 + 2 + 1 + 1)
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


def test_moving_to_another_seat_keeps_the_cookie_and_the_deck(session, lobby):
    client = client_for(session)
    client.post("/api/lobby/join", json={"name": "Ann", "seat": 0})
    token = client.get_cookie(seat_cookie(8766)).value
    deck = list(lobby.decks())[-1]
    client.post("/api/lobby/deck", json={"seat": 0, "deck": deck})
    assert client.post("/api/lobby/join", json={"name": "Ann", "seat": 1}).status_code == 200
    assert client.get_cookie(seat_cookie(8766)).value == token
    view = client.get("/api/state").get_json()["lobby"]
    assert view["you"] == 1 and view["seats"][0]["kind"] == "open" and view["seats"][1]["deck"] == deck


def test_joining_sets_the_seat_cookie_and_the_view_follows_it(session):
    client = client_for(session)
    assert client.get("/api/config").get_json() == {"mode": "lan", "push_port": 8766, "share_url": None}
    reply = client.post("/api/lobby/join", json={"name": "Ann"})
    assert reply.status_code == 200 and seat_cookie(8766) in reply.headers["Set-Cookie"]
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


def test_a_state_that_loses_a_key_goes_whole_with_its_version():
    # Declaring attackers: the state offers can_attack, then an attack trigger goes on the stack.
    declaring = {"version": 7, "decision": {"decision": "a"},
                 "state": {"turn": 3, "stack": [], "can_attack": ["x"], "log": ["a"]}}  # fmt: skip
    triggered = {"version": 9, "decision": {"decision": "t"},
                 "state": {"turn": 3, "stack": [{"id": "s"}], "log": ["a"]}}  # fmt: skip
    change = delta(declaring, triggered)
    assert change["set"]["version"] == 9  # without it the page takes the view for one it has and drops it
    assert patched(declaring, change) == triggered


def priority(name, turn=5):
    state = {"turn": turn, "log": [], "players": []}
    return {"decision": name, "kind": "pass", "prompt": "", "state": state,
            "options": [{"id": "pass", "type": "pass", "label": "Pass"}]}  # fmt: skip


def test_going_back_waits_for_everyone_still_playing(session, lobby, engine, monkeypatch):
    monkeypatch.setattr(webplay, "HOLD_SECONDS", 0.05)
    ann, bob = seated(lobby)  # and two bots, who never vote
    session.start(ann)
    for index in (0, 1):
        session.engine_table(index).publish({"state": priority("x")["state"]})
    session.ask_rewind(ann, 1)
    assert session.view(ann)["rewind"] == {"by": "Ann", "turn": 4, "status": "asking", "waiting": ["Bob"],
                                           "refused_by": None, "mine": True, "your_vote": False}  # fmt: skip
    assert session.view(bob)["rewind"]["your_vote"]
    with pytest.raises(ValueError, match="already asked"):
        session.ask_rewind(bob, 0)
    # Until Bob agrees, a priority is asked as usual.
    ann_table = session.engine_table(0)
    ann_table.pass_mode = "manual"
    assert ann_table.decide(priority("p1")) == {"wait": True}
    # Ann waits at her priority; Bob's yes reaches her there.
    monkeypatch.setattr(webplay, "HOLD_SECONDS", 5)
    replies = []
    waiting = threading.Thread(target=lambda: replies.append(ann_table.decide(priority("p1"))))
    waiting.start()
    time.sleep(0.1)
    session.answer_rewind(bob, True)
    waiting.join(5)
    assert replies == [{"rollback": 1}]
    assert session.view(bob)["rewind"] is None  # done once
    monkeypatch.setattr(webplay, "HOLD_SECONDS", 0.05)
    target = {**priority("p2"), "kind": "target", "options": [{"id": "t", "type": "target", "label": "Bob"}]}
    assert ann_table.decide(target) == {"wait": True}


def test_going_back_is_off_when_someone_refuses(session, lobby, engine):
    ann, bob = seated(lobby)
    session.start(ann)
    session.engine_table(0).publish({"state": priority("x")["state"]})
    session.ask_rewind(ann, 0)
    session.answer_rewind(bob, False)
    assert session.view(ann)["rewind"]["status"] == "refused" and session.view(ann)["rewind"]["refused_by"] == "Bob"
    assert session._take_rewind(priority("p")) is None
    with pytest.raises(ValueError, match="Nothing to answer"):
        session.answer_rewind(bob, True)


def test_alone_with_bots_going_back_needs_no_vote(session, lobby, engine):
    ann = lobby.join("Ann")
    for index in (1, 2):
        lobby.set_seat(ann, index, "bot")
    session.start(ann)
    session.engine_table(0).publish({"state": priority("x")["state"]})
    session.ask_rewind(ann, 0)
    assert session.view(ann)["rewind"]["status"] == "approved"
    assert session._take_rewind({**priority("t"), "options": [{"id": "t", "type": "target"}]}) is None
    assert session._take_rewind(priority("p")) == {"rollback": 0}


def board(me, turn=3, lost=()):
    """A board as the engine sends it to seat `me`: players by name, a hand."""
    names = ["Ann", "Bob", "Bot 3", "Bot 4"]
    players = [{"id": n, "name": n, "side": "you" if n == me else "opponent", "lost": n in lost} for n in names]
    hand = [{"name": f"{me}'s card"}]
    return {"state": {"turn": turn, "log": [], "players": players, "priority_id": "Bob", "hand": hand}}


def test_someone_arriving_during_a_game_watches_a_player_of_their_choice(session, lobby, engine):
    ann, bob = seated(lobby)
    lobby.choose_first(ann, 0)
    session.start(ann)
    for index, name in enumerate(["Ann", "Bob", "Bot 3", "Bot 4"]):
        session.engine_table(index).publish(board(name))
    cid = lobby.join("Cid")
    view = session.view(cid)
    # The first person still playing, as they see the game, without their decisions.
    assert view["watching"]["seat"] == 0 and view["state"]["hand"] == [{"name": "Ann's card"}]
    assert view["decision"] is None and view["controller"] is None
    assert view["message"] == "Watching Ann · Turn 3: Bob is playing"
    assert [p["name"] for p in view["watching"]["players"]] == ["Ann", "Bob", "Bot 3", "Bot 4"]
    session.watch(cid, 2)  # a bot, hand included
    assert session.view(cid)["state"]["hand"] == [{"name": "Bot 3's card"}]
    with pytest.raises(ValueError, match="Nobody plays"):
        session.watch(cid, 7)
    with pytest.raises(ValueError, match="You are playing"):
        session.watch(ann, 1)
    assert session.view(ann)["watching"] is None
    with pytest.raises(ValueError, match="not playing"):
        session.ask_rewind(cid, 0)


def test_a_player_out_of_the_game_watches_the_others(session, lobby, engine):
    ann, bob = seated(lobby)
    session.start(ann)
    session.engine_table(0).publish(board("Ann", lost=["Ann"]))
    session.engine_table(1).publish(board("Bob", turn=4, lost=["Ann"]))
    view = session.view(ann)
    assert view["watching"] == {**view["watching"], "seat": 0, "out": True}  # their last board first
    session.watch(ann, 1)
    assert session.view(ann)["state"]["turn"] == 4 and session.view(ann)["state"]["hand"] == [{"name": "Bob's card"}]
    session.watch(ann, 0)  # a player out of the game shows their last board
    assert session.view(ann)["state"]["turn"] == 3
    assert session.view(bob)["watching"] is None


MOXFIELD = "\n".join(["1 Sol Ring", "1 Lim-Dûl's Vault (ALL) 23 *F*", "1 Fire // Ice", "1 Valakut Awakening",
                      *["95 Forest"], "", "1 Winota, Joiner of Forces"])  # fmt: skip


@pytest.fixture
def paster(tmp_path):
    names = tmp_path / "card-names.txt"
    known = ["Sol Ring", "Lim-Dûl's Vault", "Fire", "Ice", "Valakut Awakening", "Valakut Stoneforge", "Forest",
             "Winota, Joiner of Forces", "Kinnan, Bonder Prodigy", "Mountain"]  # fmt: skip
    names.write_text("\n".join(known) + "\n", encoding="utf-8")
    return DeckPaster(CardNames(names), tmp_path / "pasted")


def test_a_pasted_moxfield_list_has_its_commander_after_the_blank_line(paster):
    main, commanders, sideboard, unknown = parse(MOXFIELD)
    assert commanders == [(1, "Winota, Joiner of Forces")] and sideboard == [] and unknown == []
    assert (1, "Lim-Dûl's Vault") in main  # set code and foil mark dropped
    pasted = paster.paste(MOXFIELD.replace("Sol Ring", "sol ring"), "Ann (pasted)", "seat-0")
    assert pasted.deck.cards == 99 and pasted.deck.commander == "Winota, Joiner of Forces"
    lines = pasted.path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "1 Sol Ring" and "1 Fire // Ice" in lines  # the engine's spelling
    assert lines[-1] == "SB: 1 Winota, Joiner of Forces"


def test_arena_headers_and_a_sideboard_are_understood():
    text = "Commander\n1 Kinnan, Bonder Prodigy\n\nDeck\n99 Forest\n"
    assert parse(text)[:2] == ([(99, "Forest")], [(1, "Kinnan, Bonder Prodigy")])
    sixty = "20 Mountain\n40 Forest\n\n2 Sol Ring\n"
    main, commanders, sideboard, _ = parse(sixty)
    assert commanders == [] and sideboard == [(2, "Sol Ring")] and sum(n for n, _ in main) == 60


def test_a_pasted_list_with_unknown_cards_or_a_wrong_size_is_refused(paster):
    with pytest.raises(ValueError, match="Not in XMage's card database: Black Lotus"):
        paster.paste(MOXFIELD.replace("Sol Ring", "Black Lotus"), "x", "k")
    with pytest.raises(ValueError, match="100 cards with its commander; this one has 101"):
        paster.paste(MOXFIELD.replace("95 Forest", "96 Forest"), "x", "k")
    with pytest.raises(ValueError, match="Not a card line"):
        paster.paste("Sol Ring and friends", "x", "k")
    with pytest.raises(ValueError, match="only 10 cards"):
        paster.paste("10 Forest", "x", "k")


def test_a_pasted_deck_is_its_owners_for_the_session(shelf, feed, paster):
    lobby = Lobby(shelf, feed.bump, paster=paster)
    ann, bob = lobby.join("Ann"), lobby.join("Bob")
    deck = lobby.paste_deck(ann, MOXFIELD)
    assert lobby.seat_of(ann).deck == deck.name == "Ann (pasted)"
    assert lobby.deck_path(deck.name).suffix == ".txt" and lobby.view(bob)["pasted"][0]["commander"] == deck.commander
    with pytest.raises(ValueError, match="whoever pasted it"):
        lobby.choose_deck(bob, 1, deck.name)
    lobby.set_seat(ann, 2, "closed")
    lobby.set_seat(ann, 3, "closed")
    lobby.choose_deck(bob, 1, "kinnan")
    assert lobby.problems() == [] and lobby._format() == "commander-duel"
    lobby.leave(ann)
    assert lobby.view(bob)["pasted"] == []  # gone with its owner
