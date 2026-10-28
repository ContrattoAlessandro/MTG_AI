"""The game in progress: the engine plays it, and each person at the table has a decision table.

Each bot has a table too, which only receives its board (the engine's eye on it). Whoever is
not playing (someone who came to watch, a player out of the game) sees the board of a player
of their choice, hand included, as that player sees it, without the decisions.
"""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ..engine import write_game_file
from ..webplay import Table
from .feed import ChangeFeed
from .lobby import Lobby, SeatKind, TablePlan

MAX_TURNS = 500
MAX_MINUTES = 12 * 60


class Engine(Protocol):
    """What the session needs of the XMage engine (PersistentEngine)."""

    def run(self, config_path) -> tuple[int, int, int, int, int]: ...

    def kill(self) -> None: ...


class SeatTable(Table):
    """`mz web`'s decision hand-off for one seat; every change also wakes the table's feed."""

    def __init__(self, feed: ChangeFeed, rewind: Callable[[dict], dict | None] = lambda payload: None):
        super().__init__()
        self.feed = feed
        self.status = "playing"
        self.rewind = rewind

    def changed(self):
        super().changed()
        self.feed.bump()

    def interrupt(self, payload: dict) -> dict | None:
        return self.rewind(payload)

    def playing(self) -> bool:
        """Still in the game: neither conceded nor out (as this seat's last board shows)."""
        with self.cond:
            if self.concede or self.result:
                return False
            players = (self.state or {}).get("players", [])
        return not any(p.get("side") == "you" and p.get("lost") for p in players)


class Rewind:
    """A request to go back to the start of a turn: every person still playing must agree (bots always do).

    asking -> approved (the next priority of a person unwinds the turn) -> gone, or refused.
    """

    def __init__(self, by: int, names: dict[int, str], turns: int, turn: int, voters: set[int]):
        self.by, self.names, self.turns, self.turn = by, names, turns, turn
        self.voters, self.votes = voters, {by}
        self.status = "asking" if voters - self.votes else "approved"
        self.refused_by: str | None = None

    def public(self, seat: int | None) -> dict:
        waiting = sorted(self.voters - self.votes)
        return {
            "by": self.names[self.by],
            "turn": self.turn,
            "status": self.status,
            "waiting": [self.names[i] for i in waiting],
            "refused_by": self.refused_by,
            "mine": seat == self.by,
            "your_vote": self.status == "asking" and seat in waiting,
        }


def game_file(plan: TablePlan, deck_path: Callable[[str], Path], engine_url: str) -> dict:
    """The engine's table game file (TableGame.java): people relay through `engine_url`/seat/<index>,
    where the bots' boards arrive as well."""
    seats = []
    for seat in plan.seats:
        entry = {
            "name": seat.name,
            "deckPath": str(deck_path(seat.deck)),
            "endpoint": f"{engine_url}/seat/{seat.index}",
        }
        if seat.kind == SeatKind.PERSON:
            entry |= {"type": "web"}
        else:
            entry |= {"type": "xmage-ai", "skill": seat.skill}
        seats.append(entry)
    # The plan lists the seats in turn order, the first player first.
    return {"format": plan.format, "goes_first": 0, "max_turns": MAX_TURNS, "max_minutes": MAX_MINUTES, "seats": seats}


class TableSession:
    """Starts, follows and stops the games of one table; one game at a time."""

    def __init__(self, lobby: Lobby, feed: ChangeFeed, start_engine: Callable[[], Engine], workdir: Path,
                 engine_url: str):  # fmt: skip
        self.lobby = lobby
        self.feed = feed
        self.start_engine = start_engine
        self.workdir = workdir
        self.engine_url = engine_url
        self.engine: Engine | None = None
        self.runner: threading.Thread | None = None
        self.tables: dict[int, SeatTable] = {}  # by seat index, the people of the current game
        self.eyes: dict[int, SeatTable] = {}  # by seat index, the bots' boards
        self.order: list[int] = []  # the seats of the current game in turn order
        self.watching: dict[str, int] = {}  # by token: the seat someone not playing watches
        self.game = 0
        self.rewind: Rewind | None = None  # a request to go back to the start of a turn
        self.status = "starting"  # starting | ready | playing | over | error
        self.message = "Starting the engine…"
        self.state_lock = threading.Lock()
        self.control = threading.Lock()  # one start or stop at a time

    # --- lifecycle

    def boot(self) -> None:
        try:
            self.engine = self.start_engine()
        except Exception as error:  # shown in the browsers
            self._set("error", f"The engine did not start: {error}")
            return
        self._set("ready", "Take a seat, choose decks and start.")

    def start(self, token: str | None) -> None:
        if self.lobby.seat_of(token) is None:
            raise ValueError("Take a seat first.")
        with self.control:
            if self.engine is None:
                raise ValueError("The engine is not ready yet.")
            self._stop_current()
            plan = self.lobby.plan()
            with self.state_lock:
                self.game += 1
                game = self.game
                self.rewind = None
                self.tables = {seat.index: SeatTable(self.feed, self._take_rewind) for seat in plan.seats
                               if seat.kind == SeatKind.PERSON}  # fmt: skip
                self.eyes = {seat.index: SeatTable(self.feed) for seat in plan.seats if seat.kind != SeatKind.PERSON}
                self.order = [seat.index for seat in plan.seats]
                self.watching = {}
            self.lobby.lock_for_game(True)
            spec = game_file(plan, self.lobby.deck_path, self.engine_url)
            path = write_game_file(self.workdir / f"lan-game-{game}.json", spec)
            self._set("playing", "Shuffling up…")
            self.runner = threading.Thread(target=self._play, args=(path, game), name="lan-game", daemon=True)
            self.runner.start()

    def abandon(self, token: str | None) -> None:
        """End the game for everyone (a new game will be set up)."""
        if self.lobby.seat_of(token) is None:
            raise ValueError("Take a seat first.")
        with self.control:
            if self.runner is None or not self.runner.is_alive():
                raise ValueError("No game in progress.")
            self._stop_current()
            with self.state_lock:
                self.tables, self.eyes, self.order = {}, {}, []  # every browser goes back to the lobby
                self.rewind = None
            self._set("ready", "Game abandoned.")

    def concede(self, token: str | None) -> None:
        """The person leaves the game; the others play on (the engine ends it when no person is left)."""
        table = self.table_of(token)
        if table is None or self.runner is None or not self.runner.is_alive():
            raise ValueError("No game in progress.")
        with table.cond:
            table.concede = True
            table.changed()

    # --- going back to the start of a turn

    def ask_rewind(self, token: str | None, turns: int) -> None:
        """Ask to go back to the start of this turn (0) or the previous one (1)."""
        table = self.table_of(token)
        if table is None or self.runner is None or not self.runner.is_alive() or not table.playing():
            raise ValueError("You are not playing in a game.")
        if turns not in (0, 1):
            raise ValueError("Go back to the start of this turn or of the previous one.")
        turn = ((table.view()["state"] or {}).get("turn")) or 0
        if turn - turns < 1:
            raise ValueError("There is no earlier turn to go back to.")
        me = self.lobby.seat_of(token).index
        with self.state_lock:
            tables = dict(self.tables)
        voters = {index for index, other in tables.items() if other.playing()}
        names = {entry["index"]: entry["name"] for entry in self.lobby.view(token)["seats"]}
        with self.state_lock:
            if self.rewind is not None and self.rewind.status in ("asking", "approved"):
                raise ValueError("Someone already asked to go back.")
            self.rewind = Rewind(me, names, turns, turn - turns, voters | {me})
        self._rewind_changed()

    def answer_rewind(self, token: str | None, accept: bool) -> None:
        seat = self.lobby.seat_of(token)
        with self.state_lock:
            rewind = self.rewind
            if rewind is None or rewind.status != "asking" or seat is None or seat.index not in rewind.voters:
                raise ValueError("Nothing to answer.")
            if accept:
                rewind.votes.add(seat.index)
                if not rewind.voters - rewind.votes:
                    rewind.status = "approved"
            else:
                rewind.status, rewind.refused_by = "refused", rewind.names[seat.index]
        self._rewind_changed()

    def _take_rewind(self, payload: dict) -> dict | None:
        """The engine's reply to a person's priority once the table agreed (once, at a priority only)."""
        options = payload.get("options") or []
        if not options or options[0].get("type") != "pass":
            return None  # in the middle of a choice: wait for a priority
        with self.state_lock:
            rewind = self.rewind
            if rewind is None or rewind.status != "approved":
                return None
            self.rewind = None
        self.feed.bump()
        return {"rollback": rewind.turns}

    def _rewind_changed(self) -> None:
        """Wake every seat: a person waiting at a priority gives way to an agreed rewind."""
        with self.state_lock:
            tables = list(self.tables.values())
        for table in tables:
            with table.cond:
                table.changed()

    def close(self) -> None:
        if self.engine is not None:
            self.engine.kill()

    def _play(self, path: Path, game: int) -> None:
        try:
            counts = self.engine.run(path)
            error = None if counts[0] or counts[2] else "the game ended with an engine error (see the engine log)"
        except Exception as failure:
            error = str(failure)
        if self.game != game:
            return  # a newer game replaced this one
        aborted = any(table.abort for table in self.tables.values())
        self.lobby.lock_for_game(False)
        if aborted:
            return  # whoever abandoned the game reports it
        if error and not any(table.result for table in self.tables.values()):
            self._set("error", f"Game stopped: {error}.")
        else:
            self._set("over", "Game over.")

    def _stop_current(self) -> None:
        """Abandon the running game; restart the engine if it does not stop in time."""
        if self.runner is None or not self.runner.is_alive():
            return
        for table in [*self.tables.values(), *self.eyes.values()]:
            with table.cond:
                table.abort = True
                table.changed()
        self.runner.join(timeout=30)
        if self.runner.is_alive():
            # The engine is busy with the bots' turns: restart it.
            self.engine.kill()
            self.runner.join(timeout=10)
            self.engine = self.start_engine()
        self.lobby.lock_for_game(False)

    def _set(self, status: str, message: str) -> None:
        with self.state_lock:
            self.status, self.message = status, message
        self.feed.bump()

    # --- seats

    def table_of(self, token: str | None) -> SeatTable | None:
        seat = self.lobby.seat_of(token)
        with self.state_lock:
            return None if seat is None else self.tables.get(seat.index)

    def engine_table(self, index: int) -> SeatTable:
        with self.state_lock:
            table = self.tables.get(index) or self.eyes.get(index)
        if table is None:
            raise KeyError(index)
        return table

    # --- watching

    def watch(self, token: str | None, index: int) -> None:
        """Someone not playing chooses whose board they see."""
        if self._watched(token) is None:
            raise ValueError("You are playing: your own board is the one you see.")
        with self.state_lock:
            if index not in self.order:
                raise ValueError("Nobody plays in that seat.")
            self.watching[token] = index
        self.feed.bump()

    def _watched(self, token: str | None) -> int | None:
        """The seat whose board `token` sees instead of their own, or None (playing, or no game)."""
        seat = self.lobby.seat_of(token)
        with self.state_lock:
            order, status = list(self.order), self.status
            tables = {**self.eyes, **self.tables}
            own = None if seat is None else self.tables.get(seat.index)
            chosen = self.watching.get(token) if token else None
        if not order:
            return None
        if own is None and not self.lobby.watching(token):
            return None  # seated after the game ended: the lobby
        if own is not None and (status != "playing" or own.playing()):
            return None
        if chosen in tables:
            return chosen
        if own is not None:
            return seat.index  # out of the game: their last board until they choose another
        # The first person still playing, else whoever is.
        alive = [i for i in order if tables[i].playing()]
        return next((i for i in alive if i in self.tables), alive[0] if alive else order[0])

    def _watchable(self) -> list[dict]:
        """The players of the game in turn order, for choosing whose board to see."""
        with self.lobby.lock:
            names = {seat.index: seat.name for seat in self.lobby.seats}
        with self.state_lock:
            order = list(self.order)
            tables = {**self.eyes, **self.tables}
            people = set(self.tables)
        return [
            {"seat": i, "name": names.get(i, "?"), "person": i in people, "playing": tables[i].playing()} for i in order
        ]

    def view(self, token: str | None) -> dict:
        """Everything one browser may see: its own seat's game (or the one it watches), and the lobby."""
        table = self.table_of(token)
        watched = self._watched(token)
        if watched is not None:
            # As the watched player sees it, without their decisions (nobody here may answer them).
            base = {**self.engine_table(watched).view(), "decision": None, "result": None, "controller": None}
        else:
            base = table.view() if table else {"state": None, "decision": None, "result": None, "controller": None}
        with self.state_lock:
            status, message, game = self.status, self.message, self.game
        watching = None
        if watched is not None:
            watchable = self._watchable()
            name = next(p["name"] for p in watchable if p["seat"] == watched)
            watching = {"seat": watched, "name": name, "out": table is not None, "players": watchable}
        if status == "playing" and base["state"]:
            message = playing_message(base) if watching is None else watching_message(base["state"], name)
        return {
            **base,
            "mode": "lan",
            "status": status,
            "message": message,
            "game": game,
            "version": self.feed.version,
            "decks": None,
            "lobby": self.lobby.view(token),
            "rewind": self._rewind_view(token),
            "watching": watching,
        }

    def _rewind_view(self, token: str | None) -> dict | None:
        seat = self.lobby.seat_of(token)
        with self.state_lock:
            return None if self.rewind is None else self.rewind.public(None if seat is None else seat.index)


def playing_message(view: dict) -> str:
    """The status line of a seat during a game: whose move it is."""
    state = view["state"]
    me = next((p for p in state.get("players", []) if p.get("side") == "you"), {})
    if me.get("lost"):
        return "You are out of the game; the others play on."
    if view["decision"]:
        return f"Turn {state.get('turn')}: your move"
    waiting = next((p for p in state.get("players", []) if p.get("id") == state.get("priority_id")), None)
    if waiting is None:
        return f"Turn {state.get('turn')}"
    doing = "passing" if waiting.get("side") == "you" else f"{waiting['name']} is playing"
    return f"Turn {state.get('turn')}: {doing}"


def watching_message(state: dict, name: str) -> str:
    """The status line of someone watching `name`'s board: whose move it is."""
    holder = next((p for p in state.get("players", []) if p.get("id") == state.get("priority_id")), None)
    turn = f"Turn {state.get('turn')}" + (f": {holder['name']} is playing" if holder else "")
    return f"Watching {name} · {turn}"
