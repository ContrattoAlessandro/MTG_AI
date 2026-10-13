"""CPU versus CUDA learner timing on the same complete recorded episode.

Runs one PPO epoch from the same checkpoint on each available device, including
checkpoint loading and saving, the target replay and the backward pass. The
trained checkpoints are written to the output folder and are not used further.

    python scripts/benchmark_policy_learner.py --checkpoint initial.pt \
        --episodes run/generation_0000/episodes/selfplay --output artifacts/learner-benchmark
"""

import argparse
import json
import time
from pathlib import Path

import torch

from magezero.learner import PPOConfig, train_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", required=True, help="directory of on-policy trajectories of the checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    lengths = {path: len(json.loads(path.read_text(encoding="utf-8"))["steps"])
               for path in Path(args.episodes).glob("*.jsonl")}  # fmt: skip
    episode = min(lengths, key=lengths.get)
    states = lengths[episode]
    torch.set_num_threads(1)
    results = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        torch.manual_seed(17)
        if device == "cuda":
            torch.cuda.init()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics = train_from_files(
            args.checkpoint,
            [episode],
            root / f"profile-{device}.pt",
            PPOConfig(epochs=1, episodes_per_batch=1, sequence_length=32),
            device,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        results.append({
            "device": device,
            "seconds": seconds,
            "decision_passes_per_second": states / seconds,
            "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None,
            "metrics": metrics,
        })  # fmt: skip
    report = {
        "episode": str(episode),
        "states": states,
        "torch_cpu_threads": 1,
        "results": results,
        "scope": "same complete episode, same initial checkpoint, one PPO epoch; includes checkpoint "
        "loading/saving, target replay, backward; may include concurrent load",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
