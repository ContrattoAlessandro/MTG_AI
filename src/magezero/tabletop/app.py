"""The HTTP routes of `mz lan`: the page, the browsers' commands and the engine's relay.

The browser routes are those of `mz web` (the page is the same), applied to the seat of
the person's cookie, plus the lobby's. The engine relays each person's decisions through
/engine/seat/<index>/.
"""

from flask import Flask, Response, jsonify, request, send_from_directory

from ..webplay import WEB_DIR
from .push import seat_cookie
from .session import SeatTable, TableSession

# The page of `mz web` loads the table's script where it marks extensions.
EXTENSIONS_MARK = "<!-- mz:extensions -->"


def create_app(session: TableSession, push_port: int, share_url: str | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    lobby = session.lobby
    cookie = seat_cookie(push_port)

    def token() -> str | None:
        return request.cookies.get(cookie)

    def body() -> dict:
        return request.get_json(force=True, silent=True) or {}

    def my_table() -> SeatTable:
        table = session.table_of(token())
        if table is None:
            raise ValueError("You are not playing in this game.")
        return table

    def ok():
        return jsonify(ok=True)

    @app.before_request
    def engine_is_local():
        # The engine runs on this computer; the other browsers on the network never speak for it.
        if request.path.startswith("/engine/") and request.remote_addr not in ("127.0.0.1", "::1"):
            return jsonify(error="the engine routes answer this computer only"), 403
        return None

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(KeyError)
    def unknown(error):
        return jsonify(error="no such seat in this game"), 404

    # --- the page

    @app.get("/")
    def index():
        page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        page = page.replace(EXTENSIONS_MARK, '<script src="/lan.js"></script>', 1)
        return Response(page, mimetype="text/html")

    @app.get("/lan.js")
    def script():
        return send_from_directory(WEB_DIR, "lan.js")

    @app.get("/api/config")
    def config():
        # share_url: the address friends open (the host's own browser may be on 127.0.0.1).
        return jsonify(mode="lan", push_port=push_port, share_url=share_url)

    @app.get("/api/state")
    def state():
        # Long poll, for a browser whose WebSocket cannot connect.
        since = request.args.get("since", type=int)
        if since is not None:
            session.feed.wait(since, timeout=10)
        return jsonify(session.view(token()))

    @app.get("/api/decks")
    def decks():
        return jsonify(decks=[deck.public() for deck in lobby.shelf.decks().values()])

    # --- the lobby

    @app.post("/api/lobby/join")
    def join():
        data = body()
        seat = data.get("seat")
        secret = lobby.join(str(data.get("name", "")), None if seat is None else int(seat), token())
        reply = jsonify(ok=True)
        reply.set_cookie(cookie, secret, httponly=True, samesite="Lax", max_age=7 * 24 * 3600)
        return reply

    @app.post("/api/lobby/leave")
    def leave():
        lobby.leave(token())
        reply = jsonify(ok=True)
        reply.delete_cookie(cookie)
        return reply

    @app.post("/api/lobby/seat")
    def seat():
        data = body()
        skill = data.get("skill")
        lobby.set_seat(token(), int(data.get("seat", -1)), str(data.get("kind")), None if skill is None else int(skill))
        return ok()

    @app.post("/api/lobby/deck")
    def deck():
        data = body()
        lobby.choose_deck(token(), int(data.get("seat", -1)), str(data.get("deck")))
        return ok()

    @app.post("/api/lobby/paste")
    def paste():
        deck = lobby.paste_deck(token(), str(body().get("text", "")))
        return jsonify(ok=True, deck=deck.public())

    @app.post("/api/lobby/first")
    def first():
        index = body().get("seat")
        lobby.choose_first(token(), None if index is None else int(index))
        return ok()

    @app.post("/api/new")
    def new_game():
        session.start(token())
        return ok()

    @app.post("/api/abandon")
    def abandon():
        session.abandon(token())
        return ok()

    # --- the person's own game (the routes of `mz web`)

    @app.post("/api/answer")
    def answer():
        data = body()
        client = None if data.get("client") is None else str(data["client"])
        my_table().respond(str(data.get("decision")), str(data.get("action_id")), client)
        return ok()

    @app.get("/api/search")
    def search():
        limit = min(request.args.get("limit", 60, type=int), 200)
        return jsonify(names=my_table().search(request.args.get("q", ""), limit))

    @app.get("/api/history")
    def history():
        table = my_table()
        with table.cond:
            return jsonify(list(table.history))

    @app.post("/api/claim")
    def claim():
        table = session.table_of(token())
        if table is not None:  # before a game there is nothing to claim
            table.claim(str(body().get("client")))
        return ok()

    @app.post("/api/mode")
    def mode():
        value = str(body().get("mode"))
        if value not in ("manual", "auto", "fast"):
            raise ValueError("mode must be manual, auto or fast")
        table = session.table_of(token())
        if table is not None:
            with table.cond:
                table.pass_mode = value
        return ok()

    @app.post("/api/rewind")
    def rewind():
        # Back to the start of this turn (0) or the previous one (1), once everyone still playing agrees.
        session.ask_rewind(token(), int(body().get("turns", 0)))
        return ok()

    @app.post("/api/rewind/answer")
    def rewind_answer():
        session.answer_rewind(token(), bool(body().get("accept")))
        return ok()

    @app.post("/api/watch")
    def watch():
        # Someone not playing chooses whose board they see.
        session.watch(token(), int(body().get("seat", -1)))
        return ok()

    @app.post("/api/concede")
    def concede():
        session.concede(token())
        return ok()

    # --- the engine (one WebPlayer per person)

    @app.post("/engine/seat/<int:index>/decide")
    def engine_decide(index):
        return jsonify(session.engine_table(index).decide(request.get_json(force=True)))

    @app.post("/engine/seat/<int:index>/state")
    def engine_state(index):
        return jsonify(session.engine_table(index).publish(request.get_json(force=True)))

    @app.post("/engine/seat/<int:index>/finish")
    def engine_finish(index):
        return jsonify(session.engine_table(index).finish(request.get_json(force=True)))

    return app
