"""Synchronous collect/update generations; fixed weights during each collection.

Built for unattended multi-day runs: resumable state, engine restarts, disk
retention, and periodic fixed-benchmark evaluation with best-checkpoint selection.
"""

import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import torch
import yaml

from .checkpoint import load_checkpoint
from .engine import PersistentEngine
from .league import League
from .learner import PPOConfig, train_from_files


def _launch_server(checkpoint, port, output_dir, device, log_path, torch_threads=1, allow_reload=False):
    # An existing server must never silently collect data for the wrong process.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"policy port {port} is unavailable: {error}") from error
    command = [sys.executable, "-m", "magezero.cli", "serve", "--checkpoint", str(checkpoint),
               "--port", str(port), "--device", device,
               # The engine reads only action_id; skip diagnostic belief payloads.
               "--no-belief", "--torch-threads", str(torch_threads)]
    if output_dir:
        command += ["--episodes", str(output_dir)]
    if allow_reload:
        command.append("--allow-reload")
    stream = Path(log_path).open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    process._policy_log = stream
    process._policy_port = port
    return process


def _await_server(process, policy_version, timeout=60):
    port, log_path = process._policy_port, process._policy_log.name
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy service exited; inspect {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as reply:
                health = json.load(reply)
            if health.get("schema_version") != 2:
                raise RuntimeError("engine port belongs to a legacy service")
            if health.get("policy_version") != policy_version:
                raise RuntimeError("policy service checkpoint version does not match this generation")
            return process
        except (OSError, ValueError):
            time.sleep(0.2)
    raise TimeoutError(f"policy service was not ready after {timeout} seconds; inspect {log_path}")


def _start_server(checkpoint, port, output_dir, device, log_path, policy_version, torch_threads=1):
    process = _launch_server(checkpoint, port, output_dir, device, log_path, torch_threads)
    try:
        return _await_server(process, policy_version)
    except Exception:
        _stop_server(process)
        raise


def _service_stats(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=5) as reply:
            return json.load(reply)
    except (OSError, ValueError):
        return None


def _reload_server(process, checkpoint, output_dir, policy_version):
    body = json.dumps({"checkpoint": str(checkpoint),
                       "episodes": str(output_dir) if output_dir else None}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{process._policy_port}/reload", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as reply:
        if json.load(reply).get("policy_version") != policy_version:
            raise RuntimeError("reloaded policy service reports the wrong checkpoint version")


class _ServicePool:
    """Policy services kept across generations and reloaded in place.

    Restarting a service costs ~2.5 s (Python, torch, checkpoint). A service that
    cannot reload (e.g. a session left open by a failed game) is restarted.
    """
    def __init__(self, root, device, torch_threads):
        self.root, self.device, self.torch_threads = root, device, torch_threads
        self.processes = {}

    def prepare(self, targets):
        """targets: {port: (checkpoint, episodes or None, policy_version, name)}."""
        fresh = []
        for port, (checkpoint, episodes, version, name) in targets.items():
            process = self.processes.get(port)
            if process is not None and process.poll() is None:
                try:
                    _reload_server(process, checkpoint, episodes, version)
                    continue
                except (OSError, ValueError, RuntimeError) as error:
                    print(f"Restarting policy service on port {port}: {error}")
                    _stop_server(process)
            process = _launch_server(checkpoint, port, episodes, self.device, self.root / f"{name}.log",
                                     self.torch_threads, allow_reload=True)
            self.processes[port] = process
            fresh.append((process, version))
        for process, version in fresh:  # launched together, awaited together
            _await_server(process, version)

    def close(self):
        for process in self.processes.values():
            _stop_server(process)
        self.processes.clear()


def _stop_server(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    process._policy_log.close()


def wilson_interval(successes, trials, z=1.96):
    if trials <= 0:
        return None
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _write_json(path, value):
    """Atomic replace, so a crash never leaves a half-written state file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _strip_optimizer(source, destination):
    """Copy a checkpoint without optimizer state (~1/3 the size) for opponents/play."""
    data = torch.load(source, map_location="cpu", weights_only=True)
    data["optimizer_state_dict"] = None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(data, temporary)
    os.replace(temporary, destination)
    return destination


def _batch_timeout(config):
    """Seconds the engine may take for one batch before it is presumed hung."""
    training = config["training"]
    rounds = math.ceil(training["games"] / training.get("threads", 2))
    return 60 * (rounds * (training.get("max_minutes", 20) + 1) + 5)


def _results(ports):
    """Per-player outcomes /finish recorded on these services since their last reload."""
    totals = dict.fromkeys(("win", "loss", "draw", "truncated", "aborted"), 0)
    for port in ports:
        stats = _service_stats(port) or {}
        for key, value in stats.get("results", {}).items():
            totals[key] += value
    return totals


class _Collector:
    """The persistent engine plus restart-on-failure; services are restarted with it."""
    def __init__(self, script, root, services):
        self.script, self.root, self.services = script, root, services
        self.engine = None
        self.restarts = 0

    def run(self, config_file, timeout):
        if self.engine is None:
            self.engine = PersistentEngine(self.script, self.root / "engine.log")
        return self.engine.run(config_file, timeout)

    def reset(self, reason):
        """Kill a failed/hung JVM; services may hold open sessions, so restart them too."""
        print(f"Restarting engine and policy services: {reason}")
        self.restarts += 1
        if self.engine is not None:
            self.engine.kill()
            self.engine = None
        self.services.close()

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self.services.close()


def _game_config(base, games, ports, opponent_ports):
    config = json.loads(json.dumps(base))
    config["training"]["games"] = games
    # Odd games swap seats: against a fixed opponent the learner plays both decks
    # of an asymmetric matchup, and each controller starts with each deck.
    config["goes_first"] = "random"
    config["server"] = {"host": "127.0.0.1", "port": ports[0], "opponent_port": opponent_ports[0],
                        "ports": ports, "opponent_ports": opponent_ports, "alternate_seats": True}
    for player in ("player_a", "player_b"):
        config[player]["hiddenInfo"] = {"see_opponent_hand": False}
    return config


def _evaluate(collector, base, folder, current, version, benchmarks, games, ai_skill, actor_ports, digest):
    """Fixed benchmark matches; never recorded as training data."""
    report = {}
    for name, anchor in benchmarks:
        config = _game_config(base, games, actor_ports,
                              actor_ports if anchor is None else [port + 1 for port in actor_ports])
        targets = {port: (current, None, version, f"actor-{i}") for i, port in enumerate(actor_ports)}
        if anchor is None:
            config["player_b"]["type"] = "xmage-ai"
            config["opponent_ai"] = {"skill": ai_skill}
        else:
            _, _, catalog, data = load_checkpoint(anchor)
            if catalog.digest != digest:
                raise ValueError(f"benchmark {anchor} uses a different card catalog")
            targets.update({port + 1: (anchor, None, data["policy_version"], f"opponent-{i}")
                            for i, port in enumerate(actor_ports)})
        config_file = folder / f"eval-{name.replace(':', '-').replace(' ', '_')}.yml"
        config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        started = time.perf_counter()
        try:
            collector.services.prepare(targets)
            counts = collector.run(config_file, _batch_timeout(config))
        except (RuntimeError, TimeoutError, OSError, ValueError) as error:
            collector.reset(f"evaluation against {name} failed: {error}")
            report[name] = {"error": str(error)}
            continue
        outcome = _results(actor_ports)
        scored = outcome["win"] + outcome["loss"] + outcome["draw"] + outcome["truncated"]
        # Unfinished games count as half; errors are reported, never scored.
        score = (outcome["win"] + 0.5 * (outcome["draw"] + outcome["truncated"])) / scored if scored else None
        report[name] = {**outcome, "engine": counts, "score": score,
                        "win_rate_wilson95": wilson_interval(outcome["win"], scored),
                        "seconds": time.perf_counter() - started}
        print(f"Evaluation vs {name}: {json.dumps(report[name])}")
    return report


def _primary_score(report):
    """Selection metric: score against XMage's AI when evaluated, else mean anchor score."""
    if "xmage-ai" in report:
        return report["xmage-ai"].get("score")
    scores = [entry["score"] for entry in report.values() if entry.get("score") is not None]
    return sum(scores) / len(scores) if scores else None


def run_selfplay(checkpoint, game_config, policy_config, engine_script, output,
                 iterations=10, games=32, device="auto", league_path=None, actor_device="cpu",
                 max_seconds=None, engine_threads=None, snapshot_every=5, actor_threads=1,
                 actor_shards=1, resume=False, eval_every=0, eval_games=64, eval_ai_skill=6,
                 eval_anchors=(), keep_episodes=False, keep_every=50, min_free_gb=20.0,
                 max_retries=3, max_failure_fraction=0.25):
    """Run collect/learn generations until `iterations` generations exist in `output`.

    Long-run safety: a failed or hung engine batch restarts the engine and services
    and replays the generation (up to max_retries); individual game failures are
    counted, not fatal. `resume=True` continues an interrupted run from its state
    file. Episodes are deleted after training unless kept, checkpoints are pruned
    to league snapshots, the best evaluated policy, and every keep_every-th one.
    """
    script = Path(engine_script).resolve()
    if not script.is_file():
        raise ValueError("v2 XMage engine launcher is missing; see docs/xmage-integration.md")
    if iterations < 1 or games < 1 or snapshot_every < 1 or keep_every < 1 or max_retries < 0:
        raise ValueError("iterations, games, snapshot and retention intervals must be positive")
    if not 1 <= actor_shards <= 8:
        raise ValueError("policy service shards must be between 1 and 8")
    if engine_threads is not None and not 1 <= engine_threads <= 16:
        raise ValueError("engine threads must be between 1 and 16")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("time budget must be positive and finite")
    if eval_every < 0 or eval_games < 1 or not 0 <= eval_ai_skill <= 10:
        raise ValueError("evaluation interval must be >= 0, games >= 1, AI skill 0 (off) to 10")
    if not 0 < max_failure_fraction <= 1:
        raise ValueError("max_failure_fraction must be in (0, 1]")
    raw = yaml.safe_load(Path(policy_config).read_text(encoding="utf-8"))
    training = PPOConfig(**raw.get("training", {}))
    base = yaml.safe_load(Path(game_config).read_text(encoding="utf-8"))
    for player in ("player_a", "player_b"):
        if base[player].get("type") != "policy":
            raise ValueError("v2 self-play requires policy players, never MCTS fallback")
        for key in ("mcts", "priors", "noise"):
            if key in base[player]:
                raise ValueError(f"legacy {key} settings are incompatible with direct policy players")
        base[player]["deckPath"] = str(Path(base[player]["deckPath"]).resolve())
        if not Path(base[player]["deckPath"]).is_file():
            raise ValueError(f"deck is missing: {base[player]['deckPath']}")
    if engine_threads is not None:
        base["training"]["threads"] = engine_threads
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / "run-state.json"
    if state_file.exists() and not resume:
        raise ValueError(f"{root} holds an existing run; pass --resume to continue it or choose a fresh output")
    if resume and state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("format_version") != 1:
            raise ValueError("incompatible run state")
        print(f"Resuming at generation {state['next_generation']} from {state['current']}")
    else:
        if any(root.glob("generation_*")):
            raise ValueError(f"{root} has generations but no run state; choose a fresh output directory")
        state = {"format_version": 1, "initial_checkpoint": str(Path(checkpoint).resolve()),
                 "current": str(Path(checkpoint).resolve()), "next_generation": 0, "durations": [],
                 "best": None, "league": str(Path(league_path).resolve() if league_path else root / "league.json")}
    league_file = Path(state["league"])
    league = League.load(league_file) if league_file.exists() else League()
    _, _, initial_catalog, _ = load_checkpoint(state["initial_checkpoint"])
    benchmarks = ([("xmage-ai", None)] if eval_ai_skill else []) + [
        (f"anchor:{Path(anchor).stem}", Path(anchor).resolve()) for anchor in eval_anchors]
    if eval_every and not benchmarks:
        raise ValueError("evaluation needs the XMage AI (skill > 0) or at least one anchor checkpoint")
    current = Path(state["current"])
    session_started = time.perf_counter()
    session_durations = []
    actor_ports = [50052 + 2 * i for i in range(actor_shards)]
    collector = _Collector(script, root, _ServicePool(root, actor_device, actor_threads))

    def save_state():
        state["current"] = str(current)
        _write_json(state_file, state)
        _write_json(root / "run-summary.json", {
            "initial_checkpoint": state["initial_checkpoint"], "latest_checkpoint": str(current),
            "completed_generations": state["next_generation"], "best": state["best"],
            "engine_restarts_this_session": collector.restarts,
            "session_wall_seconds": time.perf_counter() - session_started,
            "requested_budget_seconds": max_seconds, "learner_device": device, "actor_device": actor_device,
            "scope": "budget estimated between complete generations; no partial games relabelled as draws"})

    try:
        while state["next_generation"] < iterations:
            generation = state["next_generation"]
            elapsed = time.perf_counter() - session_started
            # Finish whole generations and keep checkpoints valid. Avoid starting
            # another generation when its recent measured duration would exceed
            # the budget; individual game lengths still make this an estimate.
            recent = session_durations[-3:]
            estimate = sum(recent) / len(recent) if recent else 0
            if max_seconds is not None and (elapsed >= max_seconds or
                    (recent and elapsed + estimate > max_seconds)):
                print(f"Time budget: stopped between generations after {elapsed:.1f}s; next estimate={estimate:.1f}s")
                break
            free_gb = shutil.disk_usage(root).free / 2**30
            if free_gb < min_free_gb:
                print(f"Disk guard: stopped with {free_gb:.1f} GB free (< {min_free_gb} GB); free space and --resume")
                break
            generation_started = time.perf_counter()
            folder = root / f"generation_{generation:04d}"
            if folder.exists():
                # Left by an interrupted attempt: its data never reached a checkpoint.
                failed = root / f"{folder.name}.incomplete-{int(time.time())}"
                shutil.rmtree(folder / "episodes", ignore_errors=True)
                folder.rename(failed)
                print(f"Moved incomplete {folder.name} to {failed.name}")
            folder.mkdir()
            _, _, catalog, initial = load_checkpoint(current)
            # Alternate latest-policy self-play and historical league games.
            opponent = Path(league.sample()) if league.opponents and generation % 2 else current
            _, _, opponent_catalog, opponent_data = load_checkpoint(opponent)
            if catalog.digest != opponent_catalog.digest or catalog.digest != initial_catalog.digest:
                raise ValueError("all league policies must use the same immutable card catalog")
            episodes = folder / "episodes"
            # Learner shard i listens on 50052+2i, historical-opponent shard i on 50053+2i.
            # The engine routes game g to shard g mod n for both of its players.
            opponent_ports = actor_ports if opponent == current else [port + 1 for port in actor_ports]
            config = _game_config(base, games, actor_ports, opponent_ports)
            config_file = folder / "game.yml"
            config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            targets = {port: (current, episodes, initial["policy_version"], f"actor-{i}")
                       for i, port in enumerate(actor_ports)}
            if opponent != current:
                targets.update({port: (opponent, None, opponent_data["policy_version"], f"opponent-{i}")
                                for i, port in enumerate(opponent_ports)})
            print(f"Generation {generation}: {games} games, learner={initial['policy_version']}, opponent={opponent}")
            collection_started = time.perf_counter()
            for attempt in range(max_retries + 1):
                try:
                    collector.services.prepare(targets)
                    counts = collector.run(config_file, _batch_timeout(config))
                    shards = sorted(episodes.glob("*.jsonl"))
                    if not shards:
                        raise RuntimeError("engine emitted no trajectories (every game failed)")
                    break
                except (RuntimeError, TimeoutError, OSError) as error:
                    collector.reset(f"generation {generation} attempt {attempt + 1} failed: {error}")
                    shutil.rmtree(episodes, ignore_errors=True)
                    if attempt == max_retries:
                        raise RuntimeError(f"generation {generation} failed {max_retries + 1} times; "
                                           f"inspect {root / 'engine.log'} and rerun with --resume") from error
            service_stats = {"actor": [_service_stats(port) for port in actor_ports]}
            if opponent != current:
                service_stats["opponent"] = [_service_stats(port) for port in opponent_ports]
            outcome = dict(zip(("terminal", "requested", "truncated", "errors", "timeouts"), counts))
            failures = outcome["errors"] + outcome["timeouts"]
            next_checkpoint = folder / "policy.pt"
            collection_seconds = time.perf_counter() - collection_started
            training_started = time.perf_counter()
            metrics = train_from_files(current, shards, next_checkpoint, training, device)
            training_seconds = time.perf_counter() - training_started
            records = [json.loads(shard.read_text(encoding="utf-8")) for shard in shards]
            decisions = sum(len(record["steps"]) for record in records)
            metrics.update(collection_seconds=collection_seconds, training_seconds=training_seconds,
                collected_decisions=decisions, games=outcome,
                failure_fraction=failures / outcome["requested"],
                truncation_fraction=outcome["truncated"] / outcome["requested"],
                completed_games=outcome["terminal"],
                collection_games_per_hour=3600 * outcome["terminal"] / collection_seconds,
                collected_decisions_per_second=decisions / collection_seconds,
                training_decision_passes_per_second=metrics["decisions"] / training_seconds,
                requested_games=games, actor_device=actor_device, learner_device=device,
                engine_restarts=collector.restarts, policy_services=service_stats)
            # Result statistics are terminal game outcomes, from the learner's side.
            if opponent != current:
                for record in records:
                    if record["terminal"]:
                        league.record(opponent, record["steps"][-1]["reward"])
            # Sparse snapshots: one per generation would fill the pool with nearly
            # identical policies whose win-rate estimates never leave their prior.
            if generation % snapshot_every == 0:
                league.add(_strip_optimizer(current, root / "snapshots" / f"generation_{generation:04d}.pt"))
            league.save(league_file)
            previous, current = current, next_checkpoint
            if eval_every and ((generation + 1) % eval_every == 0 or generation + 1 == iterations):
                report = _evaluate(collector, base, folder, current, _policy_version(current), benchmarks,
                                   eval_games, eval_ai_skill, actor_ports, catalog.digest)
                score = _primary_score(report)
                entry = {"generation": generation, "checkpoint": str(current), "score": score, "report": report}
                with (root / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry) + "\n")
                metrics["evaluation"] = report
                if score is not None and (state["best"] is None or score > state["best"]["score"]):
                    best = _strip_optimizer(current, root / "best.pt")
                    state["best"] = {"generation": generation, "score": score, "checkpoint": str(best),
                                     "source": str(current)}
                    print(f"New best policy: generation {generation}, score {score:.3f}")
            metrics["generation_seconds"] = time.perf_counter() - generation_started
            _write_json(folder / "metrics.json", metrics)
            if not keep_episodes:
                shutil.rmtree(episodes, ignore_errors=True)
            # The policy just trained from is no longer needed unless it is a
            # milestone; league snapshots and best.pt are separate stripped copies.
            if (previous.parent.parent == root and previous.name == "policy.pt"
                    and int(previous.parent.name.split("_")[1]) % keep_every != 0
                    and str(previous.resolve()) not in {x.checkpoint for x in league.opponents}):
                previous.unlink(missing_ok=True)
            state["next_generation"] = generation + 1
            state["durations"].append(metrics["generation_seconds"])
            session_durations.append(metrics["generation_seconds"])
            save_state()
            print(f"Saved {current}; {json.dumps({k: v for k, v in metrics.items() if k != 'policy_services'})}")
            if metrics["failure_fraction"] > max_failure_fraction:
                raise RuntimeError(f"{failures}/{outcome['requested']} games failed in generation {generation}; "
                                   f"training on the survivors would bias the data. Inspect {root / 'engine.log'}, "
                                   "fix the cause, then --resume")
    finally:
        collector.close()
    return current


def _policy_version(path):
    return load_checkpoint(path)[3]["policy_version"]
