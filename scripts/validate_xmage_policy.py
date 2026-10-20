"""Play one synthetic XMage game over HTTP with the built adapter, and train on it.

Requires the XMage distribution built by integration/xmage/build.py (xmage/ by
default). This exercises the actual rules engine but does not claim coverage of
the real card database.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from magezero.checkpoint import read_policy_version
from magezero.learner import PPOConfig, train_from_files
from magezero.service_client import start_service

ROOT = Path(__file__).resolve().parents[1]
PORT = 50054


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmage", default="xmage", help="distribution built by integration/xmage/build.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/java-smoke")
    args = parser.parse_args()
    xmage, root = Path(args.xmage).resolve(), Path(args.output).resolve()
    if not (xmage / "mz" / "build.json").is_file():
        raise SystemExit(f"no XMage distribution in {xmage}: run python integration/xmage/build.py")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "episodes").exists():
        raise ValueError("choose a fresh output directory")
    classpath = os.pathsep.join(str(xmage / folder / "*") for folder in ("mz", "lib", "plugins"))
    subprocess.run(["javac", "-nowarn", "-encoding", "UTF-8", "-cp", classpath, "-d", str(root / "classes"),
                    str(ROOT / "integration/xmage/test/PolicySmoke.java")], check=True)  # fmt: skip
    service = start_service(
        Path(args.checkpoint).resolve(),
        PORT,
        root / "episodes",
        "cpu",
        root / "service.log",
        read_policy_version(args.checkpoint),
    )
    try:
        with (root / "game.log").open("w", encoding="utf-8") as game_log:
            subprocess.run(["java", "-Xmx2G", "-cp", str(root / "classes") + os.pathsep + classpath, "PolicySmoke",
                            f"http://127.0.0.1:{PORT}"],
                           check=True, stdout=game_log, stderr=game_log, timeout=120)  # fmt: skip
    finally:
        service.stop()
    shards = sorted((root / "episodes").glob("*.jsonl"))
    if len(shards) != 2:
        raise AssertionError("both policy players must produce a terminal trajectory")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = train_from_files(
        args.checkpoint, shards, root / "trained.pt", PPOConfig(epochs=1, sequence_length=8), device
    )
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"{error.cmd[0]} failed with exit code {error.returncode}; inspect compiler output or game.log"
        ) from error
