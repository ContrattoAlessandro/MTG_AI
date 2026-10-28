"""A game among friends on the local network (`mz lan`).

One computer hosts: it runs the XMage engine and this web server. Everyone, the host
included, plays in a browser at http://<host>:<port>/ with the page of `mz web`; empty
seats can be taken by XMage's AI. The engine is the only authority over the game, and
each player's browser only receives what their seat may see. Whoever is not playing (someone
who arrived during the game, a player out of it) watches: they choose a player and see the
game as that player does, hand included.

    feed.py     ChangeFeed: a version counter that wakes whoever waits for a change
    lobby.py    Lobby: the seats (person, bot, open, closed), names, decks, format, watchers
    paste.py    DeckPaster: deck lists pasted in the lobby, checked against the engine's card names
    session.py  TableSession: the engine and the game in progress, one decision table per person
    push.py     PushServer: WebSocket updates, one view per seat
    app.py      the HTTP routes (browser and engine)
    server.py   run(): wires the pieces together
"""
