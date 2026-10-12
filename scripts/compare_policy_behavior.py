"""Replay identical evaluation histories at before/after weights.

Describes action preferences, not optimality labels. All memory is reconstructed
from step zero and private hand labels are never passed to either actor.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from magezero.policy.checkpoint import load_checkpoint
from magezero.policy.schema import collate_observations


def queries(obs, catalog):
    actions = obs["actions"]
    if len(actions) < 2:
        return {}  # forced decisions have no strategic choice
    entities = {e["id"]: e for e in obs["entities"]}
    cards = {c["id"]: c for c in catalog.cards}
    result = {}
    if all(a["id"].startswith("priority-") for a in actions):
        land = [i for i, a in enumerate(actions) if a["type"] == "activate"
                and "land_access" in cards.get(a["card_id"], {}).get("roles", [])
                and entities.get(a["source"], {}).get("zone") == 2]
        casts = [i for i, a in enumerate(actions) if a["type"] == "cast"]
        if land:
            result["play_land_when_legal"] = land
        if casts:
            result["cast_when_spell_available"] = casts
        result["pass_at_priority_choice"] = [i for i, a in enumerate(actions) if a["type"] == "pass"]
    if all(a["type"] == "attack" for a in actions):
        attack = [i for i, a in enumerate(actions) if a["id"] != "stay"]
        result["attack_when_legal"] = attack
        opponents = [e for e in obs["entities"] if e["zone"] == 1 and e["controller"] == 1
                     and cards.get(e["card_id"], {}).get("features", [0, 0])[1] > 0]
        source = entities.get(actions[0]["source"], {})
        if not opponents and source.get("features", [0])[0] > 0:
            result["attack_positive_power_into_no_opposing_creatures"] = attack
    if all(a["type"] == "block" for a in actions):
        result["block_when_legal"] = [i for i, a in enumerate(actions) if a["id"] != "none"]
    if all(a["type"] == "mulligan" for a in actions):
        keep = [i for i, a in enumerate(actions) if a["id"] == "keep"]
        result["keep_at_mulligan_choice"] = keep
        lands = sum("land_access" in cards.get(e["card_id"], {}).get("roles", [])
                    for e in obs["entities"] if e["zone"] == 2 and e["owner"] == 0)
        if obs["globals"][5] * 60 >= 6:
            bucket = "zero_land" if lands == 0 else "one_land" if lands == 1 else "two_to_five_lands" if lands <= 5 else "six_plus_lands"
            result["keep_large_hand_" + bucket] = keep
    return result


def compare(before_path, after_path, paths, output, dataset_scope="unspecified"):
    torch.set_num_threads(1)
    before, _, catalog, _ = load_checkpoint(before_path)
    after, _, other_catalog, _ = load_checkpoint(after_path)
    if catalog.digest != other_catalog.digest:
        raise ValueError("fixed-history comparison requires the same observation catalog")
    totals = defaultdict(lambda: {"count": 0, "before": 0., "after": 0.})
    states = choices = 0
    kl_sum = 0.
    episode_keys = []
    with torch.inference_mode():
        for path in paths:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                episode = json.loads(line)
                memories = [None, None]
                first = episode["steps"][0]["observation"]
                episode_keys.append([first["game_id"], first["player_id"]])
                for step in episode["steps"]:
                    obs = step["observation"]
                    batch = collate_observations([obs], before.config)
                    outputs = [model(batch, memory) for model, memory in zip((before, after), memories)]
                    memories = [out.memory for out in outputs]
                    probabilities = [out.distribution().probs[0] for out in outputs]
                    if len(obs["actions"]) > 1:
                        choices += 1
                        kl_sum += float((probabilities[0] * (probabilities[0].clamp(min=1e-12).log()
                            - probabilities[1].clamp(min=1e-12).log())).sum())
                    for name, indices in queries(obs, catalog).items():
                        totals[name]["count"] += 1
                        for label, probs in zip(("before", "after"), probabilities):
                            totals[name][label] += float(probs[indices].sum())
                    states += 1
            print(f"Behavior replay: {len(episode_keys)} trajectories, {states} states", flush=True)
    results = {name: {"choice_states": values["count"],
        "before_probability": values["before"] / values["count"],
        "after_probability": values["after"] / values["count"]} for name, values in totals.items()}
    report = {"before_checkpoint": str(Path(before_path).resolve()), "after_checkpoint": str(Path(after_path).resolve()),
        "states": states, "choice_states": choices, "episodes": len(episode_keys),
        "episode_keys": episode_keys, "mean_policy_kl_at_choice_states": kl_sum / max(choices, 1),
        "action_preferences": results,
        "dataset_scope": dataset_scope,
        "scope": "same recorded histories replayed from step zero; correlated observations; preferences are descriptive, not proof of correct strategy"}
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--max-trajectories", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-scope", choices=["heldout", "training", "unspecified"], default="unspecified")
    args = parser.parse_args()
    groups = [sorted(Path(item).glob("*.jsonl")) for item in args.episodes]
    # Balance the two observer streams rather than selecting only the first
    # directory alphabetically. Ordering within each stream remains fixed.
    paths = [group[i] for i in range(max((len(group) for group in groups), default=0))
             for group in groups if i < len(group)][:args.max_trajectories]
    if not paths:
        raise ValueError("no recorded evaluation trajectories")
    print(json.dumps(compare(args.before, args.after, paths, args.output, args.dataset_scope), indent=2))
