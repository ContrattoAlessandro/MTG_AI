"""Label mix and size of a `mz teacher-collect` output (docs/warm-start.md).

usage: python scripts/teacher_stats.py artifacts/teacher
"""

import json
import sys
from collections import Counter
from pathlib import Path


def main(root):
    root = Path(root)
    files = sorted((root / "episodes").glob("*.jsonl"))
    decisions, choices, by_type, chosen_type = 0, 0, Counter(), Counter()
    priority_acts = attacks = 0
    results, sizes = Counter(), 0
    for path in files:
        sizes += path.stat().st_size
        episode = json.loads(path.read_text(encoding="utf-8"))
        results["truncated" if episode.get("truncated") else "terminal" if episode.get("terminal") else "other"] += 1
        for step in episode["steps"]:
            actions = step["observation"]["actions"]
            chosen = next(a for a in actions if a["id"] == step["action_id"])
            decision = (
                "priority"
                if chosen["type"] in ("pass", "cast", "activate")
                and len(actions) > 1
                and all(a["id"].startswith("priority-") for a in actions)
                else chosen["type"]
            )
            decisions += 1
            if len(actions) < 2:
                continue
            choices += 1
            by_type[decision] += 1
            chosen_type[(decision, chosen["type"])] += 1
            if decision == "priority" and chosen["type"] != "pass":
                priority_acts += 1
            if decision == "attack" and chosen["id"] != "stay":
                attacks += 1
    summary = json.loads((root / "teacher-summary.json").read_text(encoding="utf-8"))
    seconds = sum(b["seconds"] for b in summary["batches"])
    games = summary["games_played"]
    print(
        f"games {games} in {seconds / 60:.1f} min ({games / max(seconds, 1) * 3600:.0f}/h); "
        f"trajectories {len(files)} {dict(results)}"
    )
    print(f"decisions {decisions} ({decisions / max(games, 1):.0f}/game), with alternatives {choices}")
    print(f"disk {sizes / 2**30:.2f} GiB ({sizes / max(decisions, 1) / 1024:.1f} KiB/decision)")
    print(
        f"priority: {priority_acts}/{by_type['priority']} cast or activate; "
        f"attack: {attacks}/{by_type['attack']} attack"
    )
    for decision, count in by_type.most_common():
        mix = {kind: n for (d, kind), n in chosen_type.items() if d == decision}
        print(f"  {decision:10s} {count:7d}  {mix}")


if __name__ == "__main__":
    main(sys.argv[1])
