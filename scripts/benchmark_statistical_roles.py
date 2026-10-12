"""Matched learned-count versus statistical-role forward timings (no games)."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from magezero.policy.catalog import CardCatalog
from magezero.policy.model import EntityBeliefPolicy, ModelConfig
from magezero.policy.roles import ROLE_NAMES
from magezero.policy.schema import collate_observations
from policy_smoke import observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    catalog = CardCatalog.from_dict({"format_version": 2, "role_names": list(ROLE_NAMES), "cards": [
        {"id": i, "key": f"bench-{i}", "name": f"Bench {i}", "features": [0.] * 32,
         "roles": [ROLE_NAMES[(i - 1) % len(ROLE_NAMES)]]} for i in range(1, 25)]})
    results = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for mode in ("learned", "statistical_roles"):
            torch.manual_seed(35)
            config = ModelConfig(num_cards=catalog.num_cards, catalog_digest=catalog.digest, belief_mode=mode)
            model = EntityBeliefPolicy(config, catalog.tensor(), catalog.role_tensor() if mode == "statistical_roles" else None).to(device).eval()
            obs = observation(config, "benchmark", 0)
            obs["entities"] = [{**obs["entities"][0], "id": f"entity-{i}", "card_id": i % 24 + 1} for i in range(64)]
            obs["actions"] = [{**obs["actions"][1], "id": f"action-{i}", "source": f"entity-{i}"} for i in range(32)]
            obs["opponent_deck"] = [{"card_id": i, "lower": 0, "upper": 4} for i in range(1, 25)]
            obs["opponent_hand_size"] = 7
            batch = collate_observations([obs], config).to(device)
            def sync():
                if device == "cuda":
                    torch.cuda.synchronize()
            times = []
            with torch.inference_mode():
                for _ in range(10):
                    model(batch)
                for _ in range(50):
                    sync(); started = time.perf_counter(); model(batch); sync()
                    times.append((time.perf_counter() - started) * 1000)
            results.append({"device": device, "belief_mode": mode, "median_ms": statistics.median(times),
                "p95_ms": sorted(times)[47], "actor_parameters": sum(p.numel() for p in model.parameters())})
    report = {"shape": {"entities": 64, "actions": 32, "deck_kinds": 24, "hand_size": 7},
        "torch": torch.__version__, "results": results,
        "scope": "matched synthetic forward shapes/config, one CPU thread, batch 1, 50 warmed samples; excludes HTTP, Java, training and playing strength"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
