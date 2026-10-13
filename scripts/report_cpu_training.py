"""Aggregate the CPU pilot and its predeclared fixed-opponent evaluation.

Inputs: the run folder, arena results from evaluate_policy_games.py, and a
behaviour comparison from compare_policy_behavior.py. The report reproduced in
notes/history/cpu-training-2026-10-12.json came from this script.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize_arena(paths):
    all_games = []
    groups = defaultdict(list)
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for game in report["games"]:
            item = {**game, "block": str(Path(path).parent)}
            all_games.append(item)
            if game.get("terminal") and "finish_error" not in game:
                groups[(item["block"], game["seed"])].append(item)
    terminal = [g for g in all_games if g.get("terminal") and "finish_error" not in g]
    wins = sum(g["result"] == 1 for g in terminal)
    losses = sum(g["result"] == -1 for g in terminal)
    draws = sum(g["result"] == 0 for g in terminal)
    n = len(terminal)
    if n == 0:
        raise ValueError("no terminal evaluation games")
    rate = wins / n
    z = 1.96
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    pairs = [
        games for games in groups.values() if len(games) == 2 and {g["learner_starts"] for g in games} == {True, False}
    ]
    scores = np.array([np.mean([(g["result"] + 1) / 2 for g in games]) for games in pairs])
    if not len(scores):
        raise ValueError("no complete swapped-seat seed pairs")
    rng = np.random.default_rng(20261012)
    means = scores[rng.integers(0, len(scores), size=(20000, len(scores)))].mean(1)
    interval = np.quantile(means, [0.025, 0.975]).tolist()
    unbiased = n == len(all_games) and len(pairs) * 2 == n
    conclusion = (
        "supports improvement on this matchup"
        if interval[0] > 0.5
        else "supports regression on this matchup"
        if interval[1] < 0.5
        else "inconclusive: no demonstrated improvement"
    )
    if not unbiased:
        conclusion = "inconclusive because incomplete/error games can bias the comparison"
    return {
        "requested_games": len(all_games),
        "terminal_games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "errors_or_truncations": len(all_games) - n,
        "win_rate": rate,
        "win_rate_wilson_95": [max(0, center - half), min(1, center + half)],
        "complete_seed_pairs": len(pairs),
        "pair_mean_score": float(scores.mean()),
        "pair_score_bootstrap_95": interval,
        "conclusion": conclusion,
        "by_seat": {
            name: {
                "games": sum(g["learner_starts"] == first for g in terminal),
                "wins": sum(g["learner_starts"] == first and g["result"] == 1 for g in terminal),
                "losses": sum(g["learner_starts"] == first and g["result"] == -1 for g in terminal),
                "draws": sum(g["learner_starts"] == first and g["result"] == 0 for g in terminal),
            }
            for name, first in (("starts", True), ("second", False))
        },
        "scope": "one fixed mirror matchup; sampled policies; bootstrap resamples seed-pairs, "
        "but independence across pairs remains an assumption",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--arenas", nargs="+", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    run = json.loads((root / "run-summary.json").read_text())
    generations = [json.loads(p.read_text()) for p in sorted(root.glob("generation_*/metrics.json"))]
    evaluation = summarize_arena(args.arenas)
    behavior = json.loads(Path(args.behavior).read_text())
    training_keys = set()
    for path in root.glob("generation_*/episodes/*.jsonl"):
        episode = json.loads(path.read_text())
        obs = episode["steps"][0]["observation"]
        training_keys.add((obs["game_id"], obs["player_id"]))
    if training_keys.intersection(tuple(key) for key in behavior["episode_keys"]):
        raise ValueError("behavior evaluation leaked training trajectories")
    report = {
        "run": run,
        "generations": generations,
        "training_totals": {
            "games": sum(g["completed_games"] for g in generations),
            "recorded_decisions": sum(g["collected_decisions"] for g in generations),
            "ppo_decision_passes": sum(g["decisions"] for g in generations),
            "optimizer_steps": sum(g["optimizer_steps"] for g in generations),
            "collection_seconds": sum(g["collection_seconds"] for g in generations),
            "learning_seconds": sum(g["training_seconds"] for g in generations),
            "generation_seconds": sum(g["generation_seconds"] for g in generations),
        },
        "evaluation": evaluation,
        "behavior": behavior,
        "heldout_trajectory_ids_disjoint_from_training": True,
        "configuration": "configs/experiments/policy-cpu-30min.yml",
        "plan": "notes/cpu-30min-evaluation-plan.md",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "training": report["training_totals"],
                "evaluation": evaluation,
                "preferences": behavior["action_preferences"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
