"""Play against a policy checkpoint in the browser (`mz web`).

Three processes: this web server, a policy service for the bot, and an XMage
engine (`PolicyGames --serve`) that plays one game at a time. The person's seat
is a `WebPlayer` in the engine. It sends every decision it would otherwise make
itself (the same candidate lists the policy chooses from), with readable labels
and a snapshot of what the person may see, to `/engine/decide`. The browser polls
`/api/state`, shows the board and the pending decision, and answers through
`/api/answer`. Mana is paid automatically, as in training.
"""

import collections
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .engine import PersistentEngine, write_game_file

WEB_DIR = Path(__file__).parent / "web"
# An engine decision request is held this long before the engine is told to ask again.
HOLD_SECONDS = 20.0
SOURCES = ("src", "src-ma")  # adapter source folders under integration/xmage


def strip_tags(message: str) -> str:
    """XMage log lines carry client HTML (colours, card links); keep the text."""
    return re.sub(r"<[^>]+>", "", message).replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")


def fold(text: str) -> str:
    """Lower case without accents or punctuation, for searching card names ("Lim-Dûl" ~ "lim dul")."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def read_deck(path: Path) -> tuple[int, set[str], list[str]]:
    """Main-deck card count, every card name, and the sideboard of an XMage .dck file.

    Lines are `N [SET:NUM] Name`, sideboard lines (a Commander deck's commander) `SB: N [SET:NUM] Name`.
    """
    total, names, sideboard = 0, set(), []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*(SB:\s*)?(\d+)\s+\[[^\]]*\]\s*(.+?)\s*$", line)
        if not match:
            continue
        names.add(match.group(3))
        if match.group(1):
            sideboard.append(match.group(3))
        else:
            total += int(match.group(2))
    return total, names, sideboard


def compile_adapter(repo: Path, xmage: Path, classes: Path) -> None:
    """Compile the adapter sources into `classes` when they are newer than the last build.

    The classes come first on the engine's classpath, so they override the
    adapter jar (xmage/mz) without touching the jars that training runs use.
    """
    sources = [p for folder in SOURCES for p in (repo / "integration" / "xmage" / folder).rglob("*.java")]
    stamp = classes / ".built"
    if stamp.exists() and all(p.stat().st_mtime <= stamp.stat().st_mtime for p in sources):
        return
    print(f"[web] compiling the XMage adapter into {classes}")
    classes.mkdir(parents=True, exist_ok=True)
    separator = ";" if sys.platform == "win32" else ":"
    result = subprocess.run(
        ["javac", "-nowarn", "-encoding", "UTF-8", "-cp", f"{xmage / 'lib' / '*'}{separator}{xmage / 'plugins' / '*'}",
         "-d", str(classes), *map(str, sources)],
        capture_output=True, text=True,
    )  # fmt: skip
    if result.returncode != 0:
        sys.exit(f"javac failed:\n{result.stdout}\n{result.stderr}")
    stamp.touch()


def engine_launcher(folder: Path, xmage: Path, classes: Path, heap: str) -> Path:
    """A PolicyGames launcher with the web classes ahead of the distribution's jars."""
    folder.mkdir(parents=True, exist_ok=True)
    main = "org.mage.magezero.PolicyGames"
    if sys.platform == "win32":
        script = folder / "web-engine.bat"
        script.write_text(
            "@echo off\r\n"
            f'pushd "{xmage}"\r\n'
            f'java -Xmx{heap} -Dh2.bindAddress=127.0.0.1 -cp "{classes};mz/*;lib/*;plugins/*" {main} %*\r\n'
            "popd\r\n",
            encoding="utf-8",
        )
    else:
        script = folder / "web-engine.sh"
        script.write_text(
            "#!/bin/sh\n"
            f'cd "{xmage}" || exit 2\n'
            f'exec java -Xmx{heap} -Dh2.bindAddress=127.0.0.1 -cp "{classes}:mz/*:lib/*:plugins/*" {main} "$@"\n',
            encoding="utf-8",
        )
        script.chmod(0o755)
    return script


class Table:
    """The one game in progress, shared by the engine-facing and browser-facing routes."""

    def __init__(self):
        self.cond = threading.Condition()
        self.status = "starting"  # starting | ready | playing | over | error
        self.message = "Starting the engine and the bot…"
        self.state = None
        self.decision = None
        self.answer = None
        self.abort = False
        self.concede = False
        self.result = None
        self.decks = None  # the running game's decks: {"you": deck, "opponent": deck}
        # The browser tab that plays: the last one to claim the table. Other tabs watch,
        # so two open tabs never answer (or auto-pass) the same decision.
        self.controller = None
        # Recent decisions and answers, for diagnosing the flow (GET /api/history).
        self.history = collections.deque(maxlen=300)
        self.game = 0
        self.version = 0

    def changed(self):
        self.version += 1
        self.cond.notify_all()

    def view(self) -> dict:
        with self.cond:
            state = dict(self.state) if self.state else None
            if state:
                state["log"] = [strip_tags(line) for line in state.get("log", [])]
            return {
                "status": self.status,
                "message": self.message,
                "game": self.game,
                "version": self.version,
                "state": state,
                # A searched choice's names (every card in XMage) stay here: GET /api/search.
                "decision": None
                if self.decision is None
                else {
                    **{k: v for k, v in self.decision.items() if k not in ("state", "names")},
                    **({"count": len(self.decision["names"])} if "names" in self.decision else {}),
                },
                "result": self.result,
                "decks": self.decks,
                "controller": self.controller,
            }

    def search(self, query: str, limit: int = 60) -> list[str]:
        """Names of the pending searched choice that match `query`: whole name, then word starts, then anywhere."""
        with self.cond:
            names = self.decision.get("names", []) if self.decision else []
        words = fold(query).split()
        if not words:
            return []
        ranked = []
        for name in names:
            folded = fold(name)
            if not all(word in folded for word in words):
                continue
            starts = folded.split()
            if folded.startswith(" ".join(words)):
                rank = 0
            elif all(any(s.startswith(w) for s in starts) for w in words):
                rank = 1
            else:
                rank = 2
            ranked.append((rank, len(name), name))
        ranked.sort()
        return [name for _, _, name in ranked[:limit]]

    # --- engine side

    def decide(self, payload: dict) -> dict:
        deadline = time.monotonic() + HOLD_SECONDS
        with self.cond:
            if self.abort:
                return {"abort": True}
            if self.concede:
                return {"concede": True}
            if self.decision is None or self.decision["decision"] != payload["decision"]:
                self.decision, self.answer = payload, None
                self.state = payload["state"]
                state = self.state
                self.history.append(
                    {
                        "t": round(time.time(), 2),
                        "asked": payload["kind"],
                        "turn": state.get("turn"),
                        "step": state.get("step"),
                        "active": state.get("active"),
                        "stack": len(state.get("stack", [])),
                        "options": [o["label"] for o in payload["options"]],
                    }
                )
                self.message = f"Turn {self.state['turn']}: your move"
                self.changed()
            while self.answer is None and not self.abort and not self.concede:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"wait": True}
                self.cond.wait(remaining)
            if self.abort:
                return {"abort": True}
            if self.concede:
                self.decision = None
                self.changed()
                return {"concede": True}
            answer, self.decision, self.answer = self.answer, None, None
            self.message = f"Turn {self.state['turn']}: the bot is playing"
            self.changed()
            return {"action_id": answer}

    def publish(self, payload: dict) -> dict:
        with self.cond:
            if self.abort:
                return {"abort": True}
            self.state = payload["state"]
            self.changed()
            return {"concede": True} if self.concede else {"ok": True}

    def finish(self, payload: dict) -> dict:
        with self.cond:
            self.state = payload["state"]
            self.result = payload["result"]
            self.decision = None
            self.changed()
            return {"ok": True}

    # --- browser side

    def claim(self, client: str) -> None:
        with self.cond:
            if client != self.controller:
                self.controller = client
                self.changed()

    def respond(self, decision: str, action_id: str, client: str | None = None) -> None:
        with self.cond:
            if client is not None and self.controller is not None and client != self.controller:
                raise ValueError("this game is being played in another browser tab")
            if self.decision is None or self.decision["decision"] != decision:
                raise ValueError("that decision is no longer pending")
            offered = {option["id"]: option["label"] for option in self.decision["options"]}
            if self.decision["kind"] == "search":
                if action_id not in self.decision["names"]:
                    raise ValueError("not one of the names offered")
                label = action_id
            elif self.decision["kind"] == "triggers":
                # The answer orders every offered trigger: ids joined by commas, first on the stack first.
                order = action_id.split(",")
                if sorted(order) != sorted(offered):
                    raise ValueError("the order must list every trigger once")
                label = "order: " + " | ".join(offered[i] for i in order)
            elif self.decision["kind"] in ("attackers", "blockers"):
                # The whole declaration: the chosen pairs' ids joined by commas (empty: none).
                pairs = [i for i in action_id.split(",") if i]
                if len(set(pairs)) != len(pairs) or any(i not in offered for i in pairs):
                    raise ValueError("not a declaration of offered pairs")
                none = "no attackers" if self.decision["kind"] == "attackers" else "no blockers"
                label = " | ".join(offered[i] for i in pairs) or none
            elif action_id not in offered:
                raise ValueError("not an offered option")
            else:
                label = offered[action_id]
            self.answer = action_id
            self.history.append({"t": round(time.time(), 2), "answered": label, "client": client})
            self.changed()


class WebGame:
    """Owns the engine and plays one game at a time for the browser."""

    def __init__(self, xmage: Path, launcher: Path, policy_port: int, web_port: int, workdir: Path, log: Path,
                 catalog: set[str]):  # fmt: skip
        self.xmage = xmage
        self.catalog = catalog
        self.launcher = launcher
        self.policy_port = policy_port
        self.web_port = web_port
        self.workdir = workdir
        self.log = log
        self.table = Table()
        self.engine = None
        self.runner = None
        self.lock = threading.Lock()

    def decks(self) -> list[dict]:
        """Decks in xmage/decks whose cards are all in the bot's catalog (it must know both lists)."""
        decks = []
        for path in sorted((self.xmage / "decks").glob("*.dck")):
            cards, names, sideboard = read_deck(path)
            if names and names <= self.catalog:
                commander = sideboard[0] if cards >= 99 and len(sideboard) == 1 else None
                decks.append({"name": path.stem, "cards": cards, "commander": commander, "names": sorted(names)})
        return decks

    def boot(self) -> None:
        try:
            self.engine = PersistentEngine(self.launcher, self.log)
        except Exception as error:  # surfaced in the browser
            with self.table.cond:
                self.table.status, self.table.message = "error", f"The engine did not start: {error}"
                self.table.changed()
            return
        with self.table.cond:
            self.table.status, self.table.message = "ready", "Choose decks and start a game."
            self.table.changed()

    def start(self, human_deck: str, bot_deck: str, starts: str, greedy: bool) -> None:
        names = {deck["name"]: deck for deck in self.decks()}
        if human_deck not in names or bot_deck not in names:
            raise ValueError("unknown deck")
        if starts not in ("random", "you", "bot"):
            raise ValueError("starts must be random, you or bot")
        with self.lock:
            if self.engine is None:
                raise ValueError("the engine is not ready yet")
            self._stop_current()
            commander = names[human_deck]["cards"] >= 99 and names[bot_deck]["cards"] >= 99
            config = {
                "goes_first": {"random": "random", "you": "player_b", "bot": "player_a"}[starts],
                "game_mode": "commander" if commander else "normal",
                "player_a": self._player(bot_deck, "policy"),
                "player_b": self._player(human_deck, "web"),
                # One game; no wall-clock limit per turn, since a person may think.
                "training": {"games": 1, "threads": 1, "max_turns": 200, "max_minutes": 720, "max_turn_seconds": 0},
                "server": {
                    "host": "127.0.0.1",
                    "port": self.policy_port,
                    "opponent_port": self.policy_port,
                    "sample": not greedy,
                    "web_endpoint": f"http://127.0.0.1:{self.web_port}/engine",
                },
                "logging": {"log_feature_hash": False, "save_final_wr": False},
            }
            table = self.table
            with table.cond:
                table.game += 1
                table.status, table.message = "playing", "Shuffling up…"
                table.state, table.decision, table.answer, table.result = None, None, None, None
                table.abort = table.concede = False
                table.decks = {"you": names[human_deck], "opponent": names[bot_deck]}
                table.changed()
                game = table.game
            path = write_game_file(self.workdir / f"web-game-{game}.json", config)
            self.runner = threading.Thread(target=self._play, args=(path, game), name="web-game", daemon=True)
            self.runner.start()

    def _player(self, deck: str, kind: str) -> dict:
        return {
            "deckPath": str((self.xmage / "decks" / f"{deck}.dck").resolve()),
            "type": kind,
            "hiddenInfo": {"see_opponent_hand": False},
            "gameplay": {"mulligans_enabled": True, "manual_tapping": False},
        }

    def _play(self, path: Path, game: int) -> None:
        try:
            counts = self.engine.run(path)
            error = None if counts[0] or counts[2] else "the game ended with an engine error (see the engine log)"
        except Exception as failure:
            counts, error = None, str(failure)
        with self.table.cond:
            if self.table.game != game:
                return  # a newer game replaced this one
            self._save_log(game)
            self.table.decision = None
            if self.table.abort:
                self.table.status, self.table.message = "ready", "Game abandoned."
            elif error and not self.table.result:
                self.table.status, self.table.message = "error", f"Game stopped: {error}. Log: {self.log}"
            else:
                self.table.status = "over"
                self.table.message = f"Game over: you {self.table.result}." if self.table.result else "Game over."
            self.table.changed()

    def _save_log(self, game: int) -> None:
        """Keep the game's log and the decisions asked and answered (a new game replaces them on the page)."""
        lines = [strip_tags(line) for line in (self.table.state or {}).get("log", [])]
        decisions = [json.dumps(entry, ensure_ascii=False) for entry in self.table.history]
        text = "\n".join([*lines, "", "--- decisions", *decisions]) + "\n"
        (self.workdir / f"web-game-{game}.log").write_text(text, encoding="utf-8")

    def concede(self) -> None:
        """Concede the running game: the bot wins, as in XMage."""
        with self.table.cond:
            if self.runner is None or not self.runner.is_alive():
                raise ValueError("no game in progress")
            self.table.concede = True
            self.table.message = "Conceding…"
            self.table.changed()

    def _stop_current(self) -> None:
        """Abandon the running game; restart the engine if it does not stop in time."""
        if self.runner is None or not self.runner.is_alive():
            return
        with self.table.cond:
            self.table.abort = True
            self.table.message = "Abandoning the game…"
            self.table.changed()
        self.runner.join(timeout=30)
        if self.runner.is_alive():
            # The engine is stuck in the bot's turn: restart it.
            self.engine.kill()
            self.runner.join(timeout=10)
            self.engine = PersistentEngine(self.launcher, self.log)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.kill()


def create_app(web: WebGame) -> Flask:
    app = Flask(__name__, static_folder=None)
    table = web.table

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/state")
    def state():
        since = request.args.get("since", type=int)
        if since is not None:
            # Long poll: return as soon as anything changes, or after a few seconds.
            with table.cond:
                table.cond.wait_for(lambda: table.version != since, timeout=10)
        return jsonify(table.view())

    @app.get("/api/decks")
    def decks():
        return jsonify(decks=web.decks())

    @app.post("/api/new")
    def new_game():
        body = request.get_json(force=True)
        web.start(str(body.get("human_deck")), str(body.get("bot_deck")), str(body.get("starts", "random")),
                  bool(body.get("greedy", True)))  # fmt: skip
        return jsonify(ok=True)

    @app.post("/api/answer")
    def answer():
        body = request.get_json(force=True)
        client = body.get("client")
        table.respond(str(body.get("decision")), str(body.get("action_id")), None if client is None else str(client))
        return jsonify(ok=True)

    @app.get("/api/search")
    def search():
        return jsonify(names=table.search(request.args.get("q", ""), min(request.args.get("limit", 60, type=int), 200)))

    @app.get("/api/history")
    def history():
        with table.cond:
            return jsonify(list(table.history))

    @app.post("/api/claim")
    def claim():
        table.claim(str(request.get_json(force=True).get("client")))
        return jsonify(ok=True)

    @app.post("/api/concede")
    def concede():
        web.concede()
        return jsonify(ok=True)

    @app.post("/engine/decide")
    def engine_decide():
        return jsonify(table.decide(request.get_json(force=True)))

    @app.post("/engine/state")
    def engine_state():
        return jsonify(table.publish(request.get_json(force=True)))

    @app.post("/engine/finish")
    def engine_finish():
        return jsonify(table.finish(request.get_json(force=True)))

    return app


def run(checkpoint, *, xmage="xmage", port=8765, policy_port=50071, device="cpu", heap="4G", torch_threads=2,
        open_browser=True) -> None:  # fmt: skip
    """Start the bot, the engine and the web server; serve until Ctrl+C."""
    import waitress

    from .checkpoint import read_policy_version
    from .service_client import ServiceProcess

    repo = Path(__file__).resolve().parents[2]
    xmage = Path(xmage).resolve()
    if not (xmage / "mz" / "build.json").is_file():
        sys.exit(f"no XMage distribution in {xmage}: build it with python integration/xmage/build.py")
    workdir = Path(tempfile.mkdtemp(prefix="mz-web-"))
    classes = xmage / "mz" / "web-classes"
    compile_adapter(repo, xmage, classes)
    launcher = engine_launcher(workdir, xmage, classes, heap)
    policy_log = workdir / "policy.log"
    print(f"[web] starting the bot ({checkpoint}) on port {policy_port}")
    web = None
    service = ServiceProcess(Path(checkpoint).resolve(), policy_port, None, device, policy_log, torch_threads)
    try:
        service.wait_ready(read_policy_version(checkpoint), timeout=120)
        with urllib.request.urlopen(f"http://127.0.0.1:{policy_port}/catalog", timeout=30) as reply:
            catalog = {card["name"] for card in json.load(reply)["catalog"]["cards"]}
        web = WebGame(xmage, launcher, policy_port, port, workdir, workdir / "engine.log", catalog)
        threading.Thread(target=web.boot, name="engine-boot", daemon=True).start()
        url = f"http://127.0.0.1:{port}/"
        print(f"[web] open {url}  (logs in {workdir}; Ctrl+C stops everything)")
        if open_browser:
            import webbrowser

            threading.Timer(1.0, webbrowser.open, args=(url,)).start()
        # Each held engine request and browser long poll occupies a thread.
        waitress.serve(create_app(web), host="127.0.0.1", port=port, threads=16, channel_timeout=120)
    except KeyboardInterrupt:
        pass
    finally:
        if web is not None:
            web.close()
        service.stop()
