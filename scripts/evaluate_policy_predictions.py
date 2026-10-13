"""Opponent-hand prediction quality of a checkpoint on recorded trajectories.

Compares the model's hand belief with the statistical prior (see
magezero.diagnostics.evaluate_predictions) and optionally writes a full trace
of one decision. Use held-out trajectories, never the ones trained on.

    python scripts/evaluate_policy_predictions.py --checkpoint policy.pt --episodes heldout/ \
        --output artifacts/predictions.json --trace artifacts/trace.json
"""

import argparse
import json
from pathlib import Path

from magezero.diagnostics import evaluate_predictions, trace_decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", nargs="+", required=True, help="trajectory files or directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", help="also trace one decision of the first trajectory to this JSON file")
    args = parser.parse_args()
    paths = [file for item in map(Path, args.episodes)
             for file in (sorted(item.glob("*.jsonl")) if item.is_dir() else [item])]  # fmt: skip
    report = evaluate_predictions(args.checkpoint, paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.trace:
        trace_decision(args.checkpoint, paths[0], args.trace)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
