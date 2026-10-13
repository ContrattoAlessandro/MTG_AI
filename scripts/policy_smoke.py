"""Reproducible synthetic collect/train/reload check and inference benchmark.

python scripts/policy_smoke.py --output artifacts/smoke --device auto
Produces a small checkpoint/catalog also usable by integration/xmage/test/PolicySmoke.java.
These results measure implementation, not Magic playing strength.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from magezero.catalog import CardCatalog
from magezero.checkpoint import load_checkpoint, resolve_device, save_checkpoint
from magezero.learner import PPOConfig, train_from_files
from magezero.model import EntityBeliefPolicy, ModelConfig, PrivilegedCritic
from magezero.schema import ENTITY_FEATURES, collate_observations
from magezero.service import PolicyService
from magezero.roles import ROLE_NAMES


def observation(config, game, step):
    return {"schema_version": 2, "catalog_digest": config.catalog_digest, "game_id": game,
        "player_id": "smoke", "step": step,
        "entities": [{"id": "card", "card_id": 1, "kind": 1, "zone": 1, "owner": 0,
            "controller": 0, "visibility": "public", "features": [0.1] + [0.] * (ENTITY_FEATURES - 1)}],
        "relations": [], "globals": [step / 100.] + [0.] * 23,
        "events": [{"seq": step, "type": "pass", "actor": 1, "card_id": 0, "features": [0.] * 8}],
        "actions": [{"id": name, "type": name, "source": "card", "target": None,
            "card_id": 1, "features": [float(i)] + [0.] * 15} for i, name in enumerate(["pass", "attack"])],
        "opponent_deck": [{"card_id": i, "lower": 0, "upper": 4} for i in range(1, 4)],
        "opponent_hand_size": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/smoke")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "initial.pt").exists():
        raise ValueError("choose a fresh output directory to preserve prior smoke artifacts")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    device = resolve_device(args.device)
    catalog = CardCatalog.from_dict({"format_version": 2, "role_names": list(ROLE_NAMES), "cards": [
        {"id": i, "key": f"smoke-{i}", "name": f"Card {i}", "features": [0.] * 32,
         "roles": ["board_presence"] if i < 3 else ["land_access"]} for i in range(1, 4)]})
    config = ModelConfig(num_cards=catalog.num_cards, catalog_digest=catalog.digest)
    model = EntityBeliefPolicy(config, catalog.tensor(), catalog.role_tensor()).to(device).eval()
    critic = PrivilegedCritic(config).to(device)
    version = save_checkpoint(root / "initial.pt", model, critic, catalog)
    (root / "catalog.json").write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
    service = PolicyService(model, catalog, version, root / "episodes")
    for game in range(2):
        for step in range(8):
            service.act({"observation": observation(config, f"smoke-{game}", step), "sample": True})
        service.finish({"game_id": f"smoke-{game}", "player_id": "smoke", "terminal": True,
            "result": 1 if game == 0 else -1, "labels": [{"opponent_hand_counts": [1, 1, 1]}] * 8})
    metrics = train_from_files(root / "initial.pt", sorted((root / "episodes").glob("*.jsonl")),
        root / "trained.pt", PPOConfig(epochs=2, sequence_length=4), str(device))
    load_checkpoint(root / "trained.pt", str(device))
    # Benchmark a more representative 64-entity, 32-action, 24-card-kind snapshot.
    larger_catalog = CardCatalog.from_dict({"format_version": 2, "role_names": list(ROLE_NAMES), "cards": [
        {"id": i, "key": f"bench-{i}", "name": f"Bench {i}", "features": [0.] * 32,
         "roles": [ROLE_NAMES[(i - 1) % len(ROLE_NAMES)]]} for i in range(1, 25)]})
    larger_config = ModelConfig(num_cards=larger_catalog.num_cards, catalog_digest=larger_catalog.digest)
    larger_model = EntityBeliefPolicy(larger_config, larger_catalog.tensor(), larger_catalog.role_tensor()).to(device).eval()
    obs = observation(larger_config, "benchmark", 0)
    obs["entities"] = [{**obs["entities"][0], "id": f"entity-{i}", "card_id": i % 24 + 1} for i in range(64)]
    obs["actions"] = [{**obs["actions"][1], "id": f"action-{i}", "source": f"entity-{i}"} for i in range(32)]
    obs["opponent_deck"] = [{"card_id": i, "lower": 0, "upper": 4} for i in range(1, 25)]
    obs["opponent_hand_size"] = 7
    batch = collate_observations([obs], larger_config).to(device)
    def sync():
        if device.type == "cuda": torch.cuda.synchronize()
    times = []
    with torch.inference_mode():
        for _ in range(5): larger_model(batch)
        for _ in range(30):
            sync(); before = time.perf_counter(); out = larger_model(batch); sync()
            times.append((time.perf_counter() - before) * 1000)
    report = {"device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": torch.__version__, "actor_parameters_24_cards": sum(p.numel() for p in larger_model.parameters()),
        "forward_median_ms": statistics.median(times), "forward_p95_ms": sorted(times)[28],
        "benchmark_shape": {"entities": 64, "actions": 32, "deck_kinds": 24, "hand_size": 7},
        "expected_hand_count": float(out.belief.expected_counts.sum()), "ppo": metrics,
        "scope": "synthetic implementation smoke test; excludes HTTP/Java and measures no playing strength"}
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
