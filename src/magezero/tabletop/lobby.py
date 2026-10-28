"""The seats of the table before a game: who sits where, with which deck, and what is played.

Someone who arrives once a game has started watches it (a watcher) and takes a seat for the next one.
"""

import random
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from ..webplay import read_deck

MAX_SEATS = 4
MAX_NAME = 24
SKILLS = range(1, 11)


class SeatKind(str, Enum):
    OPEN = "open"  # waiting for a person
    PERSON = "person"  # a person in a browser
    BOT = "bot"  # XMage's AI
    CLOSED = "closed"  # nobody plays there


@dataclass(frozen=True)
class Deck:
    name: str
    cards: int
    commander: str | None
    names: tuple[str, ...]

    def public(self) -> dict:
        return {"name": self.name, "cards": self.cards, "commander": self.commander, "names": list(self.names)}


class DeckShelf:
    """The .dck files of a folder, read again when the folder changes."""

    def __init__(self, folder: Path):
        self.folder = folder

    def decks(self) -> dict[str, Deck]:
        found = {}
        for path in sorted(self.folder.glob("*.dck")):
            cards, names, sideboard = read_deck(path)
            if names:
                commander = sideboard[0] if cards >= 99 and len(sideboard) == 1 else None
                found[path.stem] = Deck(path.stem, cards, commander, tuple(sorted(names)))
        return found

    def path(self, name: str) -> Path:
        return (self.folder / f"{name}.dck").resolve()


class Paster(Protocol):
    """Checks a pasted deck list and writes it for the engine (paste.DeckPaster)."""

    def paste(self, text: str, name: str, key: str): ...


@dataclass
class Seat:
    index: int
    kind: SeatKind = SeatKind.OPEN
    name: str = ""
    deck: str | None = None
    skill: int = 3
    token: str | None = None  # the person's secret; never shown to other browsers

    @property
    def plays(self) -> bool:
        return self.kind in (SeatKind.PERSON, SeatKind.BOT)

    def public(self) -> dict:
        return {"index": self.index, "kind": self.kind.value, "name": self.name, "deck": self.deck, "skill": self.skill}


@dataclass(frozen=True)
class TablePlan:
    """What the engine plays: a format and the playing seats in turn order, the first one first."""

    format: str
    seats: tuple[Seat, ...]


class Lobby:
    """The seats, changed by the people at the table. `on_change` runs after every change."""

    def __init__(self, shelf: DeckShelf, on_change: Callable[[], None], size: int = MAX_SEATS,
                 paster: Paster | None = None):  # fmt: skip
        if not 2 <= size <= MAX_SEATS:
            raise ValueError(f"a table has 2 to {MAX_SEATS} seats")
        self.shelf = shelf
        self.paster = paster
        self.pasted: dict[str, object] = {}  # by token: each person's pasted deck (paste.Pasted), for this session
        self.watchers: dict[str, str] = {}  # by token: the names of the people watching the game
        self.on_change = on_change
        self.lock = threading.RLock()
        self.seats = [Seat(i) for i in range(size)]
        # Until people arrive the table offers one bot, so two people can start at once.
        self.seats[-1] = Seat(size - 1, SeatKind.BOT, self._bot_name(size - 1), self._default_deck())
        self.first: int | None = None  # the seat that starts; None: random
        self.locked = False  # a game is being played: the seats cannot change

    # --- reading

    def seat_of(self, token: str | None) -> Seat | None:
        with self.lock:
            return next((seat for seat in self.seats if token and seat.token == token), None)

    def watching(self, token: str | None) -> bool:
        """Whether `token` is someone who came to watch (not seated)."""
        with self.lock:
            return bool(token) and token in self.watchers

    def decks(self) -> dict[str, Deck]:
        """The shelf's decks and the decks pasted at the table."""
        with self.lock:
            return self.shelf.decks() | {p.deck.name: p.deck for p in self.pasted.values()}

    def deck_path(self, name: str) -> Path:
        with self.lock:
            pasted = next((p for p in self.pasted.values() if p.deck.name == name), None)
        return pasted.path if pasted else self.shelf.path(name)

    def view(self, token: str | None) -> dict:
        with self.lock:
            mine = self.seat_of(token)
            return {
                "seats": [seat.public() for seat in self.seats],
                "you": None if mine is None else mine.index,
                "watcher": self.watchers.get(token) if token else None,
                "watchers": sorted(self.watchers.values(), key=str.casefold),
                "pasted": [p.deck.public() | {"seat": self.seat_of(t).index} for t, p in self.pasted.items()],
                "can_paste": self.paster is not None,
                "first": self.first,
                "locked": self.locked,
                "format": self._format(),
                "problems": self.problems(),
            }

    def problems(self) -> list[str]:
        """Why a game cannot start now (empty: it can)."""
        with self.lock:
            playing = [seat for seat in self.seats if seat.plays]
            found = []
            if len(playing) < 2:
                found.append("A game needs at least two players.")
            if not any(seat.kind == SeatKind.PERSON for seat in playing):
                found.append("Nobody has taken a seat yet.")
            if any(seat.kind == SeatKind.OPEN for seat in self.seats):
                found.append("Close the open seats, give them to a bot, or wait for your friends.")
            decks = self.decks()
            missing = [seat.name for seat in playing if seat.deck not in decks]
            if missing:
                found.append("Choose a deck for " + ", ".join(missing) + ".")
            elif self._format() is None and len(playing) >= 2:
                found.append("Commander decks and 60-card decks cannot play together.")
            names = [seat.name.casefold() for seat in playing]
            if len(set(names)) != len(names):
                found.append("Two players have the same name.")
            if self.first is not None and not self.seats[self.first].plays:
                found.append("The player chosen to start is not at the table.")
            return found

    def plan(self) -> TablePlan:
        with self.lock:
            problems = self.problems()
            if problems:
                raise ValueError(problems[0])
            playing = [seat for seat in self.seats if seat.plays]
            first = self.seats[self.first] if self.first is not None else random.choice(playing)
            start = playing.index(first)
            return TablePlan(self._format(), tuple(playing[start:] + playing[:start]))

    def _format(self) -> str | None:
        """Commander when every deck has a commander, constructed when none has; free-for-all above two."""
        decks = self.decks()
        chosen = [decks[seat.deck] for seat in self.seats if seat.plays and seat.deck in decks]
        if not chosen:
            return None
        commanders = {deck.commander is not None for deck in chosen}
        if len(commanders) > 1:
            return None
        players = sum(seat.plays for seat in self.seats)
        base = "commander-" if commanders.pop() else ""
        return base + ("duel" if players == 2 else "free-for-all")

    # --- changing

    def join(self, name: str, index: int | None = None, token: str | None = None) -> str:
        """Seat a person (in the first open seat unless `index` is given); returns their token.

        During a game, someone without a seat watches it instead.
        """
        name = " ".join(str(name).split())[:MAX_NAME]
        with self.lock:
            current = self.seat_of(token)
            if self.locked and current is None:
                return self._watch(name, token)
            self._unlocked()
            if not name:
                raise ValueError("Choose a name.")
            if any(seat.name.casefold() == name.casefold() and seat is not current for seat in self.seats):
                raise ValueError(f"{name} is already at the table.")
            if index is None:
                open_seat = next((seat.index for seat in self.seats if seat.kind == SeatKind.OPEN), None)
                index = current.index if current else open_seat
                if index is None:
                    raise ValueError("The table is full.")
            seat = self._seat(index)
            if seat is not current and seat.kind == SeatKind.PERSON:
                raise ValueError(f"{seat.name} sits there.")
            # Someone moving keeps their token (vacating the old seat forgets it); so does a
            # watcher, who sits down for the next game.
            known = token if current or token in self.watchers else None
            # Sitting in a bot's seat takes its deck; an open seat gets the deck the person had.
            deck = seat.deck if seat.kind == SeatKind.BOT or seat is current else current and current.deck
            if current is not None and current is not seat:
                self._vacate(current)
            self.watchers.pop(token, None)
            seat.token = known or secrets.token_urlsafe(18)
            seat.kind, seat.name = SeatKind.PERSON, name
            seat.deck = deck or self._default_deck()
            self.on_change()
            return seat.token

    def leave(self, token: str) -> None:
        with self.lock:
            if self.watchers.pop(token, None) is not None:
                self.on_change()
                return
            self._unlocked()
            seat = self._own(token)
            self._vacate(seat)
            self.pasted.pop(token, None)
            self.on_change()

    def set_seat(self, token: str, index: int, kind: str, skill: int | None = None) -> None:
        """A seated person opens, closes or gives a bot an empty seat (not someone else's)."""
        with self.lock:
            self._unlocked()
            self._own(token)
            seat = self._seat(index)
            if seat.kind == SeatKind.PERSON:
                raise ValueError(f"{seat.name} sits there.")
            kind = SeatKind(kind)
            if kind == SeatKind.PERSON:
                raise ValueError("People take a seat themselves.")
            seat.kind = kind
            seat.name = self._bot_name(index) if kind == SeatKind.BOT else ""
            seat.deck = (seat.deck or self._default_deck()) if kind == SeatKind.BOT else None
            if skill is not None:
                if skill not in SKILLS:
                    raise ValueError("Bot skill goes from 1 to 10.")
                seat.skill = skill
            if self.first == index and not seat.plays:
                self.first = None
            self.on_change()

    def choose_deck(self, token: str, index: int, deck: str) -> None:
        """A person chooses their own deck, or a bot's."""
        with self.lock:
            self._unlocked()
            mine = self._own(token)
            seat = self._seat(index)
            if seat is not mine and seat.kind != SeatKind.BOT:
                raise ValueError("Choose a deck for your seat or for a bot.")
            if deck not in self.decks():
                raise ValueError("Unknown deck.")
            owner = next((t for t, p in self.pasted.items() if p.deck.name == deck), None)
            if owner is not None and owner != seat.token:
                raise ValueError("A pasted deck is played by whoever pasted it.")
            seat.deck = deck
            self.on_change()

    def paste_deck(self, token: str | None, text: str) -> Deck:
        """A seated person plays a deck list they paste (kept for this session only)."""
        if self.paster is None:
            raise ValueError("Pasting a deck is not available here.")
        with self.lock:
            self._unlocked()
            seat = self._own(token)
            key = f"seat-{seat.index}-{secrets.token_hex(4)}"
            pasted = self.paster.paste(str(text), f"{seat.name} (pasted)", key)
            if pasted.deck.name in self.shelf.decks():
                raise ValueError("A deck of the shelf has that name.")
            self.pasted[token] = pasted
            seat.deck = pasted.deck.name
            self.on_change()
            return pasted.deck

    def choose_first(self, token: str, index: int | None) -> None:
        with self.lock:
            self._unlocked()
            self._own(token)
            if index is not None and not self._seat(index).plays:
                raise ValueError("Nobody plays in that seat.")
            self.first = index
            self.on_change()

    def lock_for_game(self, locked: bool) -> None:
        with self.lock:
            self.locked = locked
            self.on_change()

    def _own(self, token: str | None) -> Seat:
        seat = self.seat_of(token)
        if seat is None:
            raise ValueError("Take a seat first.")
        return seat

    def _seat(self, index: int) -> Seat:
        if not isinstance(index, int) or not 0 <= index < len(self.seats):
            raise ValueError("No such seat.")
        return self.seats[index]

    def _unlocked(self) -> None:
        if self.locked:
            raise ValueError("A game is in progress: the seats change when it ends.")

    def _watch(self, name: str, token: str | None) -> str:
        """Someone watches the game in progress; returns their token."""
        if not name:
            raise ValueError("Choose a name.")
        taken = [seat.name for seat in self.seats if seat.plays]
        taken += [n for t, n in self.watchers.items() if t != token]
        if any(other.casefold() == name.casefold() for other in taken):
            raise ValueError(f"{name} is already at the table.")
        token = token if token in self.watchers else secrets.token_urlsafe(18)
        self.watchers[token] = name
        self.on_change()
        return token

    def _vacate(self, seat: Seat) -> None:
        seat.kind, seat.name, seat.deck, seat.token = SeatKind.OPEN, "", None, None
        if self.first == seat.index:
            self.first = None

    def _default_deck(self) -> str | None:
        # The shelf's first deck (never someone's pasted one).
        return next(iter(self.shelf.decks()), None)

    @staticmethod
    def _bot_name(index: int) -> str:
        return f"Bot {index + 1}"
