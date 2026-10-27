# Playing with friends on the local network

`mz lan` hosts a game of Magic for two to four people on the same network, in the browser, with the table of [`mz web`](web-play.md). Seats nobody takes can go to XMage's own AI. The MageZero policy does not play here: it was trained for two-player games.

## Starting a table

On the computer that hosts (it needs the XMage distribution in `xmage/` and a JDK, as for `mz web`):

```bash
mz lan
```

It prints the address the others open, for example `http://192.168.1.17:8765/`, and opens the host's own browser. The others need only a browser on the same network. The first start compiles the Java adapter; booting the engine takes up to a minute.

| Option | Default | Meaning |
|---|---|---|
| `--host` | 0.0.0.0 | address to listen on (`127.0.0.1`: this computer only) |
| `--port` | 8765 | web server port; the updates use the next port (8766) |
| `--seats` | 4 | seats at the table (2 to 4) |
| `--xmage` | xmage | XMage distribution folder |
| `--heap` | 6G | engine Java heap |
| `--no-browser` | off | do not open a browser window |

On macOS the firewall asks once whether Python may accept incoming connections: allow it. Card images come from Scryfall, so every browser needs internet access.

## The lobby

Everyone types a name and takes a seat. Anyone seated can then:

- choose their own deck, and the deck and skill of each bot (skill 1 is quick; 6 is stronger and slower);
- give an empty seat to a bot, close it, or open it again;
- choose who goes first (or leave it random);
- start the game once every seat is settled.

The format follows the decks: Commander when every deck has a commander, constructed when none has; a duel with two players, free-for-all with more. Commander and 60-card decks cannot play together. The decks are those in `xmage/decks/`.

A seat belongs to the browser that took it (a cookie). Reloading the page keeps the seat; to sit two people on one computer, use a second browser or a private window.

## The table

Your half of the table is the same as in `mz web`. The opponent in view fills the other half, as in a duel. Down the top-right corner, one square per opponent in turn order shows their commander, name and life, with a gold dot on whose turn it is, a white one on who has priority and ⚔ on whoever is attacking you. Hover a square for hand, library, graveyard, poison and commander damage. The arrow at the top of the column shrinks the squares to narrow bars with name and life (and back); the page remembers the choice.

- **Click a square** to see that player's board. Your choice holds until the turn changes.
- **Otherwise the view follows the game**: whoever's turn it is, and the attacking player when you declare blockers.
- **A glowing square** means your pending decision can act on that player or one of their permanents (a target, for example): click it to see their board. The decision panel also lists every option.
- **Attacking**: click a creature that can attack several players (or planeswalkers) and their cards fan out above it: commander, name, life (loyalty for a planeswalker) the damage the creatures you have already sent there would deal (their total power), and how many untapped creatures could block. The cards are sized to the attacking creature's card. Click one to attack it; click elsewhere or Esc to keep the creature home. **All attack** sends every creature at the opponent in view.
- **Who attacks whom**: in front of every attacking creature (and every creature you have picked while declaring) a label, never wider than the card, shows the commander and name of the player it attacks, or the planeswalker.
- **The stack's lines** to something of an opponent out of view (a target, or the source of an ability) end at that opponent's square, which lights up.
- **Conceding** takes you out of the game; the others play on. When every person is out, the game ends.
- **New game** during a game shows the lobby; **Abandon the game** there ends it for everyone.

## How it works

```
 Browser A ─┐   HTTP (commands)        ┌─ WebPlayer seat A ─┐
 Browser B ─┼─────────────── server ───┼─ WebPlayer seat B ─┼─ XMage engine (one game)
 Browser C ─┘   WebSocket (updates)    ├─ XMage AI         ─┤
                                       └─ XMage AI         ─┘
```

The host is the only authority over the game: one XMage engine plays it, and each browser receives only what its seat may see (its own hand, never the others'). This is why it is client-server and not peer-to-peer: the rules engine cannot be replicated on every computer, and a peer would receive every player's hidden cards.

- **Engine** (`TableGame` in the adapter): plays a table game file, `{"format", "goes_first", "seats": [...]}`. Formats are `TableFormat` (duel, commander-duel, free-for-all, commander-free-for-all); seat kinds are `TableSeat.Kind` (`web`, `xmage-ai`). Each person is a `WebPlayer` that relays its decisions to `/engine/seat/<index>/`, which only answers the host itself.
- **Server** (`src/magezero/tabletop/`): `Lobby` (seats, decks, format), `TableSession` (the game and one decision table per person, the same hand-off as `mz web`), `ChangeFeed` (wakes whoever waits for a change), `PushServer` (WebSocket) and the Flask routes. Commands go over HTTP; every change is pushed to each browser as its seat's view.
- **Page**: `mz web`'s page plus `web/lan.js`, which plugs in through `window.mzTable`: the WebSocket transport (long polling if it cannot connect), the lens that turns the table into a duel view of the opponent in focus, the seat column, the attack fan, the attack labels and the lobby.

## Limits

- XMage's AI takes a while per decision in a four-player game; keep bot skill low for a brisk game.
- A person who is out of the game sees the board as it was when they left until the game ends.
- Nobody can watch without a seat, and nobody can join a game in progress.
