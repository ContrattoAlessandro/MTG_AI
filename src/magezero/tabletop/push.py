"""WebSocket updates: each browser receives its seat's view whenever the table changes.

The browser sends its commands (answers, lobby changes) over HTTP; this channel only
pushes. It listens on its own port beside the web server, and knows the browser by the
seat cookie, which the browser sends to every port of the host.
"""

import json
import threading
from collections.abc import Callable
from http.cookies import SimpleCookie

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from .feed import ChangeFeed

SEAT_COOKIE = "mz_seat"
# A view is sent at least this often, so a dead connection is noticed.
HEARTBEAT_SECONDS = 15.0


def seat_token(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    cookie = SimpleCookie()
    cookie.load(cookie_header)
    return cookie[SEAT_COOKIE].value if SEAT_COOKIE in cookie else None


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

    def _serve(self, connection: ServerConnection) -> None:
        token = seat_token(connection.request.headers.get("Cookie"))
        seen, sent = None, None
        try:
            while True:
                seen = self.feed.wait(-1 if seen is None else seen, HEARTBEAT_SECONDS)
                view = self.view_for(token)
                # The version changes with every seat's moves; resend only what this seat sees.
                body = json.dumps({**view, "version": None}, sort_keys=True)
                if body != sent:
                    connection.send(json.dumps(view))
                    sent = body
                else:
                    connection.ping()
        except ConnectionClosed:
            pass
