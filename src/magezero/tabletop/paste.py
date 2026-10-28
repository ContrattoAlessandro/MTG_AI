"""Deck lists pasted in the lobby, checked against the card names the engine knows.

Moxfield's plain text is the usual source: `1 Sol Ring` lines, and after a blank line the
commander (or two partners). Arena and MTGO lists work too: section headers (Commander,
Deck, Sideboard), `SB:` lines, set codes and foil marks after the name are understood.
A pasted deck lives for the session only: it is written where the engine reads it, as
a plain list the engine imports by name (TxtDeckImporter), and forgotten when the server stops.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from ..webplay import fold
from .lobby import Deck

MAX_TEXT = 20_000
# A section header and where its cards go (None: left out, like a maybeboard).
HEADERS = {
    "commander": "commander", "commanders": "commander", "deck": "main", "main": "main", "mainboard": "main",
    "main deck": "main", "sideboard": "side", "companion": "side", "maybeboard": None, "considering": None,
    "tokens": None,
}  # fmt: skip
CARD_LINE = re.compile(r"^(\d+)\s*x?\s+(.+)$", re.IGNORECASE)
# What some exports write after the name: "(SNC) 280", "[SNC]", "*F*", "#tags".
TRAILER = re.compile(r"(\s+\([A-Za-z0-9]{2,8}\)(\s+[\w-]+)?|\s+\[[A-Za-z0-9]{2,8}\]|\s+\*[A-Za-z]\*|\s+#.*)+\s*$")


class CardNames:
    """The card names the engine wrote at start (one per line), looked up without case or accents."""

    def __init__(self, path: Path):
        self.path = path
        self._names: dict[str, str] | None = None

    def lookup(self, name: str) -> str | None:
        """The engine's spelling of `name`, or None when it does not know the card."""
        if self._names is None:
            if not self.path.is_file():
                raise ValueError("The engine is still starting: paste the deck again in a moment.")
            names = self.path.read_text(encoding="utf-8").splitlines()
            self._names = {fold(name): name for name in names if name}
        # A card of two faces or halves (Fire // Ice, an adventure, a modal double-faced card)
        # is known by each half.
        halves = [self._names.get(fold(half)) for half in name.split("//")]
        if not all(halves):
            return None
        return " // ".join(halves)


@dataclass(frozen=True)
class Pasted:
    deck: Deck
    path: Path


def parse(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]], list[str]]:
    """(main, commanders, sideboard, lines not understood) of a deck list.

    Without headers, the last group after a blank line is the commander when it holds one or
    two cards next to a main deck of about a hundred; otherwise it is the sideboard.
    """
    groups: dict[str, list[tuple[int, str]]] = {"main": [], "commander": [], "side": [], "after": []}
    current: str | None = "main"
    headers, unknown = False, []
    for raw in text.splitlines():
        line = raw.replace("﻿", "").strip()
        # "Commander", "// Sideboard", "Deck:", "Mainboard (99)"
        header = re.sub(r"\s*\(\d+\)$", "", line.strip("/ ").rstrip(":").strip()).lower()
        if header in HEADERS and not CARD_LINE.match(line):
            headers, current = True, HEADERS[header]
            continue
        if not line:
            if not headers and groups["main"] and current in ("main", "after"):
                # A new group: the one before it was part of the main deck.
                groups["main"] += groups["after"]
                groups["after"], current = [], "after"
            continue
        if line.startswith(("//", "#")):
            continue
        side = line.upper().startswith("SB:")
        match = CARD_LINE.match(line[3:].strip() if side else line)
        if not match:
            unknown.append(line)
            continue
        if current is None:
            continue
        name = TRAILER.sub("", match.group(2)).strip()
        groups["side" if side else current].append((int(match.group(1)), name))
    main, after = groups["main"], groups["after"]
    commanders = groups["commander"]
    if not commanders and 1 <= sum(n for n, _ in after) <= 2 and sum(n for n, _ in main) >= 90:
        commanders, after = after, []
    return main, commanders, groups["side"] + after, unknown


class DeckPaster:
    """Checks a pasted list and writes it for the engine: Pasted, or ValueError saying what is wrong."""

    def __init__(self, names: CardNames, folder: Path):
        self.names = names
        self.folder = folder

    def paste(self, text: str, name: str, key: str) -> Pasted:
        if len(text) > MAX_TEXT:
            raise ValueError("That is too long for a deck list.")
        main, commanders, sideboard, unknown = parse(text)
        if unknown:
            raise ValueError(f"Not a card line: {unknown[0]!r} (write each card as “1 Sol Ring”).")
        if not main:
            raise ValueError("Paste a deck list: one card per line, as “1 Sol Ring”.")
        missing, spelled = [], {}
        for _, card in main + commanders + sideboard:
            known = self.names.lookup(card)
            if known is None:
                missing.append(card)
            spelled[card] = known
        if missing:
            shown = ", ".join(dict.fromkeys(missing[:8]))
            more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
            raise ValueError(f"Not in XMage's card database: {shown}{more}.")
        cards = sum(n for n, _ in main)
        if commanders:
            total = cards + sum(n for n, _ in commanders)
            if total != 100:
                raise ValueError(f"A Commander deck has 100 cards with its commander; this one has {total}.")
            sideboard = []  # a Commander game takes the commander from the sideboard: nothing else goes there
        elif cards < 40:
            raise ValueError(f"The deck has only {cards} cards.")
        lines = [f"{n} {spelled[card]}" for n, card in main]
        lines += [f"SB: {n} {spelled[card]}" for n, card in commanders + sideboard]
        self.folder.mkdir(parents=True, exist_ok=True)
        path = (self.folder / f"{key}.txt").resolve()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        names = {spelled[card] for _, card in main + commanders + sideboard}
        commander = spelled[commanders[0][1]] if commanders else None
        return Pasted(Deck(name, cards, commander, tuple(sorted(names))), path)
