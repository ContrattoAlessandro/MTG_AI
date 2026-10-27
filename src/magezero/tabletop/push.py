"""WebSocket updates: each browser receives its seat's view whenever the table changes.

The browser sends its commands (answers, lobby changes) over HTTP; this channel only
pushes. It listens on its own port beside the web server, and knows the browser by the
seat cookie, which the browser sends to every port of the host.

To spare the network (Wi-Fi, phones), a burst of changes (priority going round the
table, a bot's spell resolving) goes out as one message once the table is quiet for a
moment, and each message after the first carries only what changed (see `delta`).
A decision for the seat goes out at once.
"""

import json
import threading
import time
from collections.abc import Callable
from http.cookies import SimpleCookie

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from .feed import ChangeFeed

SEAT_COOKIE = "mz_seat"
# A view is sent at least this often, so a dead connection is noticed.
HEARTBEAT_SECONDS = 15.0
# A burst of changes is sent once nothing changed for QUIET seconds, at most HOLD after it began.
QUIET_SECONDS = 0.08
HOLD_SECONDS = 0.3


def seat_token(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    cookie = SimpleCookie()
    cookie.load(cookie_header)
    return cookie[SEAT_COOKIE].value if SEAT_COOKIE in cookie else None


def delta(old: dict, new: dict) -> dict | None:
    """What turns view `old` into `new`, or None when only the version differs.

    {"set": top-level keys that changed, "patch": state keys that changed, "log": {"drop": n,
    "add": lines}} (the log is a sliding window: drop n lines from its start, append the
    new ones), or {"set": ..., "state": whole state} when a state appears or goes.
    """
    changes = {k: v for k, v in new.items() if k not in ("state", "version") and old.get(k) != v}
    out: dict = {"set": changes} if changes else {}
    before, after = old.get("state"), new.get("state")
    if before is None or after is None:
        if before != after:
            out["state"] = after
    elif before != after:
        patch = {k: v for k, v in after.items() if k != "log" and before.get(k) != v}
        if before.keys() - after.keys():
            out["state"] = after
            return out
        if patch:
            out["patch"] = patch
        old_log, new_log = before.get("log", []), after.get("log", [])
        if old_log != new_log:
            drop = next(d for d in range(len(old_log) + 1) if new_log[: len(old_log) - d] == old_log[d:])
            out["log"] = {"drop": drop, "add": new_log[len(old_log) - drop :]}
    if not out:
        return None
    out["set"] = {**out.get("set", {}), "version": new.get("version")}
    return out


class PushServer:
    """Serves `view_for(seat token)` to every connected browser, again after each change of `feed`."""

    def __init__(self, feed: ChangeFeed, view_for: Callable[[str | None], dict], host: str, port: int):
        self.feed = feed
        self.view_for = view_for
        self.server = serve(self._serve, host, port)
        self.thread = threading.Thread(target=self.server.serve_forever, name="lan-push", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()

    def _settle(self, seen: int, token: str | None) -> tuple[int, dict]:
        """The seat's view once the table is quiet (or holds a decision for the seat)."""
        view = self.view_for(token)
        deadline = time.monotonic() + HOLD_SECONDS
        while not view.get("decision"):
            left = deadline - time.monotonic()
            if left <= 0:
                break
            newer = self.feed.wait(seen, min(QUIET_SECONDS, left))
            if newer == seen:
                break
            seen, view = newer, self.view_for(token)
        return seen, view

    def _serve(self, connection: ServerConnection) -> None:
        token = seat_token(connection.request.headers.get("Cookie"))
        seen, sent = None, None
        try:
            while True:
                seen = self.feed.wait(-1 if seen is None else seen, HEARTBEAT_SECONDS)
                seen, view = self._settle(seen, token)
                # The version changes with every seat's moves; send only what this seat sees.
                change = None if sent is None else delta(sent, view)
                if sent is None:
                    connection.send(json.dumps(view))
                elif change is not None:
                    connection.send(json.dumps({"delta": change}))
                else:
                    connection.ping()
                    continue
                sent = view
        except ConnectionClosed:
            pass
