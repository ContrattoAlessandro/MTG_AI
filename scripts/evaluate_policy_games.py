"""Fixed-opponent arena: a checkpoint against another checkpoint or upstream MageZero's MCTS.

Compiles integration/xmage/src/org/mage/magezero/PolicyArena.java against the
installed XMage distribution (xmage/lib) and plays a deck mirror with paired
seeds and alternating seats. Writes results.json (per game) and summary.json.

    python scripts/evaluate_policy_games.py --checkpoint a.pt --opponent-checkpoint b.pt \
        --deck xmage/decks/simplegreen.dck --games 64 --output artifacts/arena-a-vs-b
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from magezero.checkpoint import read_policy_version
from magezero.runner import wilson_interval
from magezero.service_client import start_service

LEARNER_PORT, OPPONENT_PORT = 50054, 50055


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent-checkpoint", help="required for --opponent policy")
    parser.add_argument("--opponent", choices=["policy", "mcts-public", "mcts-raw"], default="policy")
    parser.add_argument("--deck", default="xmage/decks/simplegreen.dck", help="deck used by both players")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--budget", type=int, default=300, help="MCTS visit cap per decision")
    parser.add_argument("--timeout", type=float, default=1.0, help="MCTS seconds per decision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--xmage", default="xmage", help="XMage distribution with the patched lib/ jars")
    parser.add_argument(
        "--record-episodes",
        action="store_true",
        help="save evaluation trajectories for fixed-history behaviour diagnostics; never train on them",
    )
    args = parser.parse_args()
    if args.games <= 0 or not 1 <= args.threads <= 16:
        raise ValueError("games must be positive and threads must be in 1..16")
    if args.opponent == "policy" and not args.opponent_checkpoint:
        raise ValueError("a policy opponent requires --opponent-checkpoint")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "results.json").exists():
        raise ValueError("choose a fresh output directory")
    xmage = Path(args.xmage).resolve()
    classpath = os.pathsep.join(str(p) for p in sorted((xmage / "lib").glob("*.jar")))
    source = Path(__file__).resolve().parents[1] / "integration/xmage/src/org/mage/magezero/PolicyArena.java"
    subprocess.run(["javac", "--release", "8", "-proc:none", "-Xlint:-options", "-cp", classpath,
                    "-d", str(root / "classes"), str(source)], check=True)  # fmt: skip

    services, started = [], time.perf_counter()
    try:
        players = [(args.checkpoint, LEARNER_PORT, "learner")]
        if args.opponent == "policy":
            players.append((args.opponent_checkpoint, OPPONENT_PORT, "opponent"))
        for checkpoint, port, name in players:
            episodes = root / f"episodes-{name}" if args.record_episodes else None
            services.append(
                start_service(
                    Path(checkpoint).resolve(),
                    port,
                    episodes,
                    "cpu",
                    root / f"{name}.log",
                    read_policy_version(checkpoint),
                )
            )
        command = [
            "java", "-Xmx8G", "-Dh2.bindAddress=127.0.0.1", "-cp", str(root / "classes") + os.pathsep + classpath,
            "org.mage.magezero.PolicyArena", "--deck", str(Path(args.deck).resolve()), "--games", str(args.games),
            "--threads", str(args.threads), "--seed", str(args.seed), "--opponent", args.opponent,
            "--port-a", str(LEARNER_PORT), "--port-b", str(OPPONENT_PORT), "--budget", str(args.budget),
            "--timeout", str(args.timeout), "--output", str(root / "results.json"),
        ]  # fmt: skip
        with (root / "engine.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=xmage, check=True, stdout=log, stderr=log, timeout=max(600, args.games * 15))
    finally:
        for service in services:
            service.stop()

    games = json.loads((root / "results.json").read_text(encoding="utf-8"))["games"]
    terminal = [g for g in games if g.get("terminal") and "finish_error" not in g]
    wins = sum(g["result"] == 1 for g in terminal)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "opponent": args.opponent,
        "opponent_checkpoint": args.opponent_checkpoint,
        "deck": str(Path(args.deck).resolve()),
        "requested_games": args.games,
        "terminal_games": len(terminal),
        "wins": wins,
        "losses": sum(g["result"] == -1 for g in terminal),
        "draws": sum(g["result"] == 0 for g in terminal),
        "errors_or_truncations": len(games) - len(terminal),
        "win_rate": wins / len(terminal) if terminal else 0,
        "win_rate_wilson_95": wilson_interval(wins, len(terminal)),
        "wall_seconds": time.perf_counter() - started,
        "mean_game_seconds": sum(g["seconds"] for g in games) / len(games),
        "mcts_visit_cap": args.budget,
        "mcts_seconds_cap": args.timeout,
        "seed": args.seed,
        "threads": args.threads,
        "recorded_evaluation_episodes": args.record_episodes,
        "scope": "fixed-opponent evaluation; paired seeds/seats do not ensure identical UUID ordering "
        "or stochastic action streams",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
