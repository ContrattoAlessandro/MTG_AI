"""The XMage engine process and the game-batch configurations it plays.

A long-lived `PolicyGames --serve` JVM plays one batch per request, so the card
database and the warmed JIT survive across generations. Requests and replies are
single stdout lines tagged MZ-POLICY; every other line is engine log.
"""

import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

TAG = "MZ-POLICY "
# Per-batch game counts in an engine RESULT reply, in order.
RESULT_FIELDS = ("terminal", "requested", "truncated", "errors", "timeouts")


class PersistentEngine:
    """One XMage JVM serving game batches over stdin/stdout."""

    def __init__(self, script, log_path, ready_timeout: float = 600):
        script = Path(script).resolve()
        command = ["cmd", "/c", str(script), "--serve"] if sys.platform == "win32" else [str(script), "--serve"]
        self.log = Path(log_path).open("a", encoding="utf-8")  # noqa: SIM115 (closed by close/kill)
        self.replies: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # A process group lets close() reach the JVM behind the shell launcher.
            **(
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if sys.platform == "win32"
                else {"start_new_session": True}
            ),
        )
        self.reader = threading.Thread(target=self._read, name="engine-output", daemon=True)
        self.reader.start()
        try:
            reply = self._reply(ready_timeout)
            if reply != "READY":
                raise RuntimeError(f"engine did not start: {reply}")
        except Exception:
            self.close()
            raise

    def _read(self) -> None:
        for line in self.process.stdout:
            self.log.write(line)
            self.log.flush()
            if line.startswith(TAG):
                self.replies.put(line[len(TAG) :].strip())
        self.replies.put("EXITED")

    def _reply(self, timeout: float | None = None) -> str:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                reply = self.replies.get(timeout=1)
                if reply == "EXITED":
                    raise RuntimeError(f"engine exited; inspect {self.log.name}")
                return reply
            except queue.Empty:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"engine gave no reply; inspect {self.log.name}") from None

    def run(self, config_path, timeout: float | None = None) -> tuple[int, int, int, int, int]:
        """Play the batch in config_path.

        Returns the RESULT_FIELDS game counts. A batch that exceeds `timeout`
        seconds raises TimeoutError; the caller must kill() the engine, since the
        hung JVM may still be playing.
        """
        if self.process.poll() is not None:
            raise RuntimeError(f"engine is not running; inspect {self.log.name}")
        self.process.stdin.write(f"{Path(config_path).resolve()}\n")
        self.process.stdin.flush()
        # The engine enforces per-game time limits (training.max_minutes).
        reply = self._reply(timeout)
        if not reply.startswith("RESULT "):
            raise RuntimeError(f"engine batch failed: {reply}; inspect {self.log.name}")
        fields = [int(x) for x in reply.split()[1:]]
        if len(fields) == 2:  # engines built before per-game outcome counts
            fields += [0, 0, fields[1] - fields[0]]
        if len(fields) != len(RESULT_FIELDS):
            raise RuntimeError(f"malformed engine reply: {reply}")
        return tuple(fields)

    def close(self) -> None:
        """Ask the engine to leave its command loop; kill it if it does not."""
        if self.process.poll() is None:
            try:
                self.process.stdin.close()  # EOF: the engine leaves its command loop
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self._kill()
        self.reader.join(timeout=5)
        self.log.close()

    def kill(self) -> None:
        """Hard stop for a failed or hung engine; safe to call more than once."""
        if self.process.poll() is None:
            self._kill()
        self.reader.join(timeout=5)
        if not self.log.closed:
            self.log.close()

    def _kill(self) -> None:
        if sys.platform == "win32":
            # Terminating cmd.exe alone would orphan the JVM.
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait(timeout=10)


def game_config(base: dict, games: int, ports, opponent_ports, *, keep_deck: bool = False, sample: bool = True) -> dict:
    """A batch of `games` policy-versus-policy games from the base game YAML.

    Odd games swap seats, and each controller starts with each deck. Without
    keep_deck the decks stay with the seats, so against a fixed opponent the
    learner plays both decks of an asymmetric matchup; with it the learner always
    plays player_a's deck. Shard i of the learner listens on ports[i], its
    opponent on opponent_ports[i]; game g goes to shard g mod len(ports).
    """
    config = json.loads(json.dumps(base))
    config["training"]["games"] = games
    config["goes_first"] = "random"
    config["server"] = {
        "host": "127.0.0.1",
        "port": ports[0],
        "opponent_port": opponent_ports[0],
        "ports": ports,
        "opponent_ports": opponent_ports,
        "alternate_seats": True,
        "learner_keeps_deck": bool(keep_deck or (base.get("server") or {}).get("learner_keeps_deck")),
        "sample": sample,
    }
    for player in ("player_a", "player_b"):
        config[player]["hiddenInfo"] = {"see_opponent_hand": False}
    return config


def gauntlet_config(
    base: dict, games: int, ports, deck, ai_skill: int, *, sample: bool = True, think_seconds: int = 0
) -> dict:
    """The learner's deck (player_a) against `deck` piloted by XMage's AI.

    think_seconds > 0 caps the AI's search per decision (XMage: 3 s per skill level).
    """
    config = game_config(base, games, ports, ports, keep_deck=True, sample=sample)
    config["player_b"]["deckPath"] = str(deck)
    config["player_b"]["type"] = "xmage-ai"
    config["opponent_ai"] = {"skill": ai_skill, **({"think_seconds": think_seconds} if think_seconds else {})}
    return config


def batch_timeout(config: dict) -> float:
    """Seconds the engine may take for one batch before it is presumed hung."""
    training = config["training"]
    rounds = math.ceil(training["games"] / training.get("threads", 2))
    return 60 * (rounds * (training.get("max_minutes", 20) + 1) + 5)
