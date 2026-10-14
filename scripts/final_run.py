"""Supervise an unattended `mz selfplay` run: restart after crashes, keep the PC awake.

    python scripts/final_run.py --output artifacts/final/run [--max-restarts 20] -- <mz selfplay arguments>

The selfplay arguments are everything after `--`, without `--output` and `--resume`
(the supervisor adds both). The supervisor

- runs `mz selfplay`, adding `--resume` once the output folder holds a run state;
- restarts it after a crash, pausing `--pause` seconds, at most `--max-restarts`
  times in a row without a completed generation (progress resets the count);
- stops on a clean finish, on the disk guard, and on the failure guard (more than
  `--max-failure-fraction` of a generation's games failed), which need a person;
- keeps Windows from sleeping while it runs (SetThreadExecutionState; a closed lid,
  a manual sleep or a Windows Update restart still stop the run, so --resume it);
- appends selfplay's output to `<output>/supervisor.log` and, every `--status-every`
  seconds, one progress line to `<output>/status.log`.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
FAILURE_GUARD = "games failed in generation"
DISK_GUARD = "Disk guard: stopped"


def keep_awake(on: bool) -> None:
    if sys.platform == "win32":
        import ctypes

        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


def completed(root: Path) -> int:
    state = root / "run-state.json"
    return json.loads(state.read_text(encoding="utf-8"))["next_generation"] if state.exists() else 0


def status_line(root: Path) -> str:
    """Generation, best score, and the last generation's throughput and failures."""
    line = f"{datetime.now():%Y-%m-%d %H:%M} generations={completed(root)}"
    summary = root / "run-summary.json"
    if summary.exists():
        best = json.loads(summary.read_text(encoding="utf-8")).get("best") or {}
        if best:
            line += f" best={best.get('score', 0):.3f}@{best.get('generation')}"
    generations = sorted(root.glob("generation_[0-9][0-9][0-9][0-9]/metrics.json"))
    if generations:
        metrics = json.loads(generations[-1].read_text(encoding="utf-8"))
        line += (
            f" last: games/h={metrics.get('collection_games_per_hour', 0):.0f}"
            f" failures={metrics.get('failure_fraction', 0):.3f}"
            f" truncations={metrics.get('truncation_fraction', 0):.3f}"
            f" entropy={metrics.get('entropy', float('nan')):.3f}"
            f" reference_kl={metrics.get('reference_kl', float('nan')):.3f}"
        )
    return line


def log(path: Path, text: str) -> None:
    print(text, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def run_once(command: list[str], log_path: Path) -> tuple[int, str]:
    """Run selfplay, tee its output to the log, and return the exit code and last lines."""
    tail: list[str] = []
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
            handle.flush()
            tail = [*tail, line][-200:]
        return process.wait(), "".join(tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mz", default=str(Path(sys.executable).with_name("mz")), help="the mz executable")
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--pause", type=float, default=120)
    parser.add_argument("--status-every", type=float, default=1800)
    parser.add_argument("selfplay", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    selfplay = args.selfplay[1:] if args.selfplay[:1] == ["--"] else args.selfplay
    if any(a in ("--output", "--resume") for a in selfplay):
        parser.error("pass --output to the supervisor and leave out --resume")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    supervisor_log, status_log = root / "supervisor.log", root / "status.log"

    stop = threading.Event()

    def report() -> None:
        while not stop.wait(args.status_every):
            log(status_log, status_line(root))

    threading.Thread(target=report, daemon=True).start()
    keep_awake(True)
    restarts, code = 0, 1
    try:
        while True:
            before = completed(root)
            command = [args.mz, "selfplay", *selfplay, "--output", str(root)]
            if (root / "run-state.json").exists():
                command.append("--resume")
            started = f"== {datetime.now():%Y-%m-%d %H:%M:%S} start at generation {before}"
            log(supervisor_log, f"{started}: {' '.join(command)}")
            code, tail = run_once(command, supervisor_log)
            after = completed(root)
            log(supervisor_log, f"== {datetime.now():%Y-%m-%d %H:%M:%S} exit {code} at generation {after}")
            if code == 0:
                reason = "disk guard" if DISK_GUARD in tail else "finished"
                log(supervisor_log, f"== stopped: {reason}")
                break
            if FAILURE_GUARD in tail:
                log(supervisor_log, "== stopped: failure guard; inspect engine.log, then rerun this command")
                break
            restarts = 0 if after > before else restarts + 1
            if restarts > args.max_restarts:
                log(supervisor_log, f"== stopped: {restarts} restarts without progress")
                break
            log(supervisor_log, f"== restart {restarts}/{args.max_restarts} in {args.pause:.0f}s")
            time.sleep(args.pause)
    finally:
        stop.set()
        keep_awake(False)
        log(status_log, status_line(root) + " (supervisor exit)")
    sys.exit(code)


if __name__ == "__main__":
    main()
