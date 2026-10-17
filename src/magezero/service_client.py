"""Client side of the policy service: `mz serve` subprocesses, health checks, hot reload and counters."""

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .schema import SCHEMA_VERSION

# Per-player game outcomes counted by a service's /finish since its start or last reload.
OUTCOMES = ("win", "loss", "draw", "truncated", "aborted")


class ServiceProcess:
    """A policy service subprocess on a local port, logging to `log_path`."""

    def __init__(
        self, checkpoint, port: int, episodes, device: str, log_path, torch_threads: int = 1, allow_reload: bool = False
    ):
        # An existing server must never silently collect data for the wrong process.
        # A just-stopped service can hold the port for a moment, so retry briefly.
        deadline = time.monotonic() + 15
        while True:
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                    break
                except OSError as error:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"policy port {port} is unavailable: {error}") from error
            time.sleep(0.5)
        command = [
            sys.executable, "-m", "magezero.cli", "serve", "--checkpoint", str(checkpoint),
            "--port", str(port), "--device", device,
            # The engine reads only action_id; skip diagnostic belief payloads.
            "--no-belief", "--torch-threads", str(torch_threads),
        ]  # fmt: skip
        if episodes:
            command += ["--episodes", str(episodes)]
        if allow_reload:
            command.append("--allow-reload")
        self.port = port
        self.log = Path(log_path).open("w", encoding="utf-8")  # noqa: SIM115 (closed by stop)
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    def wait_ready(self, policy_version: str, timeout: float = 60) -> "ServiceProcess":
        """Block until /healthz reports this exact policy version."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.running:
                raise RuntimeError(f"policy service exited; inspect {self.log.name}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=1) as reply:
                    health = json.load(reply)
                if health.get("schema_version") != SCHEMA_VERSION:
                    raise RuntimeError("engine port belongs to a legacy service")
                if health.get("policy_version") != policy_version:
                    raise RuntimeError("policy service checkpoint version does not match this generation")
                return self
            except (OSError, ValueError):
                time.sleep(0.2)
        raise TimeoutError(f"policy service was not ready after {timeout} seconds; inspect {self.log.name}")

    def reload(self, checkpoint, episodes, policy_version: str, abandon_sessions: bool = False) -> None:
        """Swap the served checkpoint in place (POST /reload; needs allow_reload).

        abandon_sessions: drop sessions a failed or interrupted game left open. Only
        safe between batches, when the engine plays no game against this service.
        """
        payload = {"checkpoint": str(checkpoint), "episodes": str(episodes) if episodes else None}
        if abandon_sessions:
            payload["abandon_sessions"] = True
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/reload",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as reply:
            if json.load(reply).get("policy_version") != policy_version:
                raise RuntimeError("reloaded policy service reports the wrong checkpoint version")

    def stop(self) -> None:
        if sys.platform == "win32" and self.running:
            # A venv's python.exe is a launcher whose child is the real interpreter;
            # terminating only the launcher would leave the server holding the port.
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.process.pid)], capture_output=True)
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.log.close()


def start_service(
    checkpoint, port: int, episodes, device: str, log_path, policy_version: str, torch_threads: int = 1
) -> ServiceProcess:
    """Launch a service and wait until it serves `policy_version`; stop it on failure."""
    service = ServiceProcess(checkpoint, port, episodes, device, log_path, torch_threads)
    try:
        return service.wait_ready(policy_version)
    except Exception:
        service.stop()
        raise


def service_stats(port: int) -> dict | None:
    """The service's /stats, or None when it does not answer."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=5) as reply:
            return json.load(reply)
    except (OSError, ValueError):
        return None


def collect_results(ports) -> dict[str, int]:
    """Summed per-player outcomes that /finish recorded on these services since their last reload."""
    totals = dict.fromkeys(OUTCOMES, 0)
    for port in ports:
        stats = service_stats(port) or {}
        for key, value in stats.get("results", {}).items():
            totals[key] += value
    return totals


class ServicePool:
    """Policy services kept across generations and reloaded in place.

    Restarting a service costs about 2.5 s (Python, torch, checkpoint). A service
    that cannot reload (e.g. one with a session left open by a failed game) is
    restarted.
    """

    def __init__(self, root: Path, device: str, torch_threads: int):
        self.root, self.device, self.torch_threads = root, device, torch_threads
        self.services: dict[int, ServiceProcess] = {}

    def prepare(self, targets: dict) -> None:
        """targets: {port: (checkpoint, episodes or None, policy_version, log name)}."""
        fresh = []
        for port, (checkpoint, episodes, version, name) in targets.items():
            service = self.services.get(port)
            if service is not None and service.running:
                try:
                    # Between batches no game is running, so open sessions are stale.
                    service.reload(checkpoint, episodes, version, abandon_sessions=True)
                    continue
                except (OSError, ValueError, RuntimeError) as error:
                    print(f"Restarting policy service on port {port}: {error}")
                    service.stop()
            service = ServiceProcess(
                checkpoint,
                port,
                episodes,
                self.device,
                self.root / f"{name}.log",
                self.torch_threads,
                allow_reload=True,
            )
            self.services[port] = service
            fresh.append((service, version))
        for service, version in fresh:  # launched together, awaited together
            service.wait_ready(version)

    def close(self) -> None:
        for service in self.services.values():
            service.stop()
        self.services.clear()
