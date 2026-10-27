"""The game in progress: the engine plays it, and each person at the table has a decision table."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ..engine import write_game_file
from ..webplay import Table
from .feed import ChangeFeed
from .lobby import DeckShelf, Lobby, SeatKind, TablePlan

MAX_TURNS = 500
MAX_MINUTES = 12 * 60


class Engine(Protocol):
    """What the session needs of the XMage engine (PersistentEngine)."""

    def run(self, config_path) -> tuple[int, int, int, int, int]: ...

    def kill(self) -> None: ...


class SeatTable(Table):
    """`mz web`'s decision hand-off for one seat; every change also wakes the table's feed."""

    def __init__(self, feed: ChangeFeed):
        super().__init__()
        self.feed = feed
        self.status = "playing"

    def changed(self):
        super().changed()
        self.feed.bump()


def game_file(plan: TablePlan, shelf: DeckShelf, engine_url: str) -> dict:
    """The engine's table game file (TableGame.java): people relay through `engine_url`/seat/<index>."""
    seats = []
    for seat in plan.seats:
        entry = {"name": seat.name, "deckPath": str(shelf.path(seat.deck))}
        if seat.kind == SeatKind.PERSON:
            entry |= {"type": "web", "endpoint": f"{engine_url}/seat/{seat.index}"}
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
        self.game = 0
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
                self.tables = {seat.index: SeatTable(self.feed) for seat in plan.seats if seat.kind == SeatKind.PERSON}
            self.lobby.lock_for_game(True)
            spec = game_file(plan, self.lobby.shelf, self.engine_url)
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
                self.tables = {}  # every browser goes back to the lobby
            self._set("ready", "Game abandoned.")

    def concede(self, token: str | None) -> None:
        """The person leaves the game; the others play on (the engine ends it when no person is left)."""
        table = self.table_of(token)
        if table is None or self.runner is None or not self.runner.is_alive():
            raise ValueError("No game in progress.")
        with table.cond:
            table.concede = True
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
        for table in self.tables.values():
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
            table = self.tables.get(index)
        if table is None:
            raise KeyError(index)
        return table

    def view(self, token: str | None) -> dict:
        """Everything one browser may see: its own seat's game, and the lobby."""
        table = self.table_of(token)
        base = table.view() if table else {"state": None, "decision": None, "result": None, "controller": None}
        with self.state_lock:
            status, message, game = self.status, self.message, self.game
        return {
            **base,
            "mode": "lan",
            "status": status,
            "message": playing_message(base) if status == "playing" and base["state"] else message,
            "game": game,
            "version": self.feed.version,
            "decks": None,
            "lobby": self.lobby.view(token),
        }


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
