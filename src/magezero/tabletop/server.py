"""`mz lan`: wire the lobby, the engine, the web server and the push channel together."""

import socket
import sys
import tempfile
import threading
from pathlib import Path

from ..engine import PersistentEngine
from ..webplay import compile_adapter, engine_launcher
from .app import create_app
from .feed import ChangeFeed
from .lobby import DeckShelf, Lobby
from .push import PushServer
from .session import TableSession


def lan_address() -> str:
    """This computer's address on the local network (the route a packet out would take; nothing is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def run(*, xmage="xmage", host="0.0.0.0", port=8765, seats=4, heap="6G", open_browser=True) -> None:
    """Start the engine, the web server and the push channel; serve until Ctrl+C."""
    import waitress

    repo = Path(__file__).resolve().parents[3]
    xmage = Path(xmage).resolve()
    if not (xmage / "mz" / "build.json").is_file():
        sys.exit(f"no XMage distribution in {xmage}: build it with python integration/xmage/build.py")
    workdir = Path(tempfile.mkdtemp(prefix="mz-lan-"))
    classes = xmage / "mz" / "web-classes"
    compile_adapter(repo, xmage, classes)
    launcher = engine_launcher(workdir, xmage, classes, heap)
    engine_log = workdir / "engine.log"

    feed = ChangeFeed()
    lobby = Lobby(DeckShelf(xmage / "decks"), feed.bump, seats)
    session = TableSession(lobby, feed, lambda: PersistentEngine(launcher, engine_log), workdir,
                           f"http://127.0.0.1:{port}/engine")  # fmt: skip
    push = PushServer(feed, session.view, host, port + 1)
    try:
        threading.Thread(target=session.boot, name="engine-boot", daemon=True).start()
        push.start()
        url = f"http://{lan_address()}:{port}/"
        print(f"[lan] friends on this network open {url}  (this computer: http://127.0.0.1:{port}/)")
        print(f"[lan] updates on port {port + 1}; logs in {workdir}; Ctrl+C stops everything")
        if open_browser:
            import webbrowser

            threading.Timer(1.0, webbrowser.open, args=(f"http://127.0.0.1:{port}/",)).start()
        # Each held engine request and browser long poll occupies a thread (four people, four engine seats).
        waitress.serve(create_app(session, port + 1, url), host=host, port=port, threads=32, channel_timeout=120)
    except KeyboardInterrupt:
        pass
    finally:
        push.stop()
        session.close()
