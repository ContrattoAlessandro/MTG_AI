"""Score checkpoints against XMage's AI outside a selfplay run (same games as its evaluation).

usage: python scripts/evaluate.py OUTPUT_DIR CHECKPOINT [CHECKPOINT ...] [--games 256] [--ai-skill 3]
Each checkpoint plays its deck (player_a of the game config) against XMage's AI on
the config's own player_b deck ("xmage-ai") and on each --gauntlet deck. Results
are appended to OUTPUT_DIR/evaluations.jsonl, one line per checkpoint.
"""

import argparse
import json
from pathlib import Path

import yaml

from magezero.policy.checkpoint import load_checkpoint
from magezero.policy.runner import _Collector, _evaluate, _primary_score, _ServicePool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--game", default="configs/commander-mirror.yml")
    parser.add_argument("--gauntlet", nargs="*", default=["xmage/decks/kinnan.dck"])
    parser.add_argument("--engine-script", default="xmage/mz-policy.bat")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--ai-skill", type=int, default=3)
    parser.add_argument("--engine-threads", type=int, default=16)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.game).read_text(encoding="utf-8"))
    for player in ("player_a", "player_b"):
        base[player]["deckPath"] = str(Path(base[player]["deckPath"]).resolve())
    base["training"]["threads"] = args.engine_threads
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    benchmarks = [("xmage-ai", None, None)] + [
        (f"xmage-ai:{Path(deck).stem}", None, Path(deck).resolve()) for deck in args.gauntlet]
    actor_ports = [50052]
    collector = _Collector(Path(args.engine_script).resolve(), root, _ServicePool(root, "cpu", 1))
    try:
        for index, checkpoint in enumerate(args.checkpoints):
            checkpoint = Path(checkpoint).resolve()
            _, _, catalog, data = load_checkpoint(checkpoint)
            folder = root / f"{index}-{checkpoint.parent.name}-{checkpoint.stem}"
            folder.mkdir(exist_ok=True)
            report = _evaluate(collector, base, folder, checkpoint, data["policy_version"], benchmarks,
                               args.games, args.ai_skill, actor_ports, catalog.digest, args.greedy)
            entry = {"checkpoint": str(checkpoint), "score": _primary_score(report), "games": args.games,
                     "ai_skill": args.ai_skill, "greedy": args.greedy, "report": report}
            with (root / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
