"""Static measured training and fixed-history behavior figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    generations = report["generations"]
    minutes = np.cumsum([g["generation_seconds"] for g in generations]) / 60
    games = np.cumsum([g["completed_games"] for g in generations])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].plot(minutes, games, marker=".")
    axes[0].set(
        xlabel="Collection + learning minutes", ylabel="Completed training games", title="CPU training progress"
    )
    evaluation = report["evaluation"]
    rate = evaluation["win_rate"]
    lo, hi = evaluation["win_rate_wilson_95"]
    axes[1].errorbar([0], [rate * 100], yerr=[[100 * (rate - lo)], [100 * (hi - rate)]], fmt="o", capsize=6)
    axes[1].axhline(50, color="grey", linestyle="--", linewidth=1)
    axes[1].set(
        xticks=[0],
        xticklabels=["Final vs frozen start"],
        ylabel="Wins / terminal games (%)",
        ylim=(0, 100),
        title=f"Held-out games: {evaluation['wins']}–{evaluation['losses']}",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "cpu-training-results.png", dpi=160)
    plt.close(fig)
    preferences = report["behavior"]["action_preferences"]
    names = {
        "play_land_when_legal": "Play legal land",
        "cast_when_spell_available": "Cast available spell",
        "pass_at_priority_choice": "Pass priority",
        "attack_when_legal": "Attack",
        "block_when_legal": "Block",
        "keep_at_mulligan_choice": "Keep at mulligan",
    }
    keys = [key for key in names if key in preferences]
    positions = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(
        positions - 0.18,
        [preferences[k]["before_probability"] * 100 for k in keys],
        width=0.36,
        label="Starting policy",
    )
    ax.bar(
        positions + 0.18, [preferences[k]["after_probability"] * 100 for k in keys], width=0.36, label="Final policy"
    )
    ax.set(
        xticks=positions,
        xticklabels=[names[k] + f"\nN={preferences[k]['choice_states']}" for k in keys],
        ylabel="Mean action probability (%)",
        ylim=(0, 100),
        title="Same held-out histories · preferences, not optimality labels",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "cpu-policy-behavior.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
