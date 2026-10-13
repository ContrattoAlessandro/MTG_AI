"""Compile the adapter, play one synthetic XMage game over HTTP, and train it.

Requires the pinned XMage checkout with Mage/target/classes and dependencies
already built in artifacts/maven; see docs/xmage-integration.md. This exercises
the actual rules engine but does not claim coverage of the real card database.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from magezero.checkpoint import load_checkpoint
from magezero.learner import PPOConfig, train_from_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=".mz_engine")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/java-smoke")
    args = parser.parse_args()
    engine, root = Path(args.engine).resolve(), Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "episodes").exists(): raise ValueError("choose a fresh output directory")
    jars = subprocess.check_output(["rg", "--files", str(ROOT / "artifacts/maven"), "-g", "*.jar"], text=True).splitlines()
    classpath = os.pathsep.join([str(engine / "Mage/target/classes")] + jars)
    sources = [ROOT / "integration/xmage/src", engine / "Mage.Server.Plugins/Mage.Player.AI/src/main/java",
        engine / "Mage.Common/src/main/java", engine / "Mage.MageZero/src/main/java",
        engine / "Mage.Server.Plugins/Mage.Game.TwoPlayerDuel/src"]
    java_files = subprocess.check_output(["rg", "--files", str(sources[0]), "-g", "*.java"], text=True).splitlines()
    subprocess.run(["javac", "--release", "8", "-proc:none", "-Xlint:-options", "-encoding", "UTF-8", "-cp", classpath,
        "-sourcepath", os.pathsep.join(map(str, sources)), "-d", str(root / "classes")]
        + java_files + [str(ROOT / "integration/xmage/test/PolicySmoke.java")], check=True)
    _, _, _, checkpoint = load_checkpoint(args.checkpoint)
    log = (root / "service.log").open("w", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    server = subprocess.Popen([sys.executable, "-m", "magezero.cli", "serve", "--checkpoint", args.checkpoint,
        "--port", "50054", "--device", "cpu", "--episodes", str(root / "episodes")], env=env, stdout=log, stderr=log)
    try:
        ready = False
        for _ in range(100):
            if server.poll() is not None: raise RuntimeError("service exited; inspect service.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:50054/healthz", timeout=1) as reply: health = json.load(reply)
                if health["policy_version"] != checkpoint["policy_version"]: raise RuntimeError("port 50054 is occupied")
                ready = True; break
            except OSError: time.sleep(0.1)
        if not ready: raise TimeoutError("policy service not ready")
        with (root / "game.log").open("w", encoding="utf-8") as game_log:
            subprocess.run(["java", "-Xmx2G", "-cp", str(root / "classes") + os.pathsep + classpath,
                "PolicySmoke", "http://127.0.0.1:50054", str(ROOT / "configs/policy-game.yml")],
                check=True, stdout=game_log, stderr=game_log, timeout=120)
    finally:
        server.terminate(); server.wait(timeout=10); log.close()
    shards = sorted((root / "episodes").glob("*.jsonl"))
    if len(shards) != 2: raise AssertionError("both policy players must produce a terminal trajectory")
    metrics = train_from_files(args.checkpoint, shards, root / "trained.pt",
        PPOConfig(epochs=1, sequence_length=8), "cuda" if __import__("torch").cuda.is_available() else "cpu")
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try: main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{error.cmd[0]} failed with exit code {error.returncode}; inspect compiler output or game.log")
