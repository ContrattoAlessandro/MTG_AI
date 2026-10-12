"""A long-lived XMage collector: `PolicyGames --serve` plays one batch per request.

The JVM, card database and warmed JIT survive across generations. Requests and
replies are single stdout lines tagged MZ-POLICY; every other line is engine log.
"""

import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

TAG = "MZ-POLICY "


class PersistentEngine:
    def __init__(self, script, log_path, ready_timeout=600):
        script = Path(script).resolve()
        command = (["cmd", "/c", str(script), "--serve"] if sys.platform == "win32"
                   else [str(script), "--serve"])
        self.log = Path(log_path).open("a", encoding="utf-8")
        self.replies: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            # A process group lets close() reach the JVM behind the shell launcher.
            **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
               else {"start_new_session": True}))
        self.reader = threading.Thread(target=self._read, name="engine-output", daemon=True)
        self.reader.start()
        try:
            reply = self._reply(ready_timeout)
            if reply != "READY":
                raise RuntimeError(f"engine did not start: {reply}")
        except Exception:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.log.write(line)
            self.log.flush()
            if line.startswith(TAG):
                self.replies.put(line[len(TAG):].strip())
        self.replies.put("EXITED")

    def _reply(self, timeout=None) -> str:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                reply = self.replies.get(timeout=1)
                if reply == "EXITED":
                    raise RuntimeError(f"engine exited; inspect {self.log.name}")
                return reply
            except queue.Empty:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"engine gave no reply; inspect {self.log.name}")

    def run(self, config_path, timeout=None) -> tuple[int, int, int, int, int]:
        """Play the batch in config_path.

        Returns (terminal, requested, truncated, errors, timeouts) game counts. A
        batch that exceeds `timeout` seconds raises TimeoutError; the caller must
        kill() the engine, since the hung JVM may still be playing.
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
        if len(fields) != 5:
            raise RuntimeError(f"malformed engine reply: {reply}")
        return tuple(fields)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()  # EOF: the engine leaves its command loop
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self._kill()
        self.reader.join(timeout=5)
        self.log.close()

    def kill(self):
        """Hard stop for a failed or hung engine; safe to call more than once."""
        if self.process.poll() is None:
            self._kill()
        self.reader.join(timeout=5)
        if not self.log.closed:
            self.log.close()

    def _kill(self):
        if sys.platform == "win32":
            # Terminating cmd.exe alone would orphan the JVM.
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait(timeout=10)
