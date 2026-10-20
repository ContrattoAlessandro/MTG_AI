"""Self-play training runs: synchronous collect/learn generations with fixed weights during collection.

Built for unattended multi-day runs: resumable state, engine restarts, disk
retention, and periodic fixed-benchmark evaluation with best-checkpoint selection.
See docs/long-run.md.
"""

import json
import math
import os
import shutil
import time
from pathlib import Path

import yaml

from .checkpoint import load_checkpoint, read_policy_version, strip_optimizer
from .engine import RESULT_FIELDS, PersistentEngine, batch_timeout, game_config, gauntlet_config, write_game_file
from .league import League
from .learner import PPOConfig, train_from_files
from .service_client import ServicePool, collect_results, service_stats

# Learner shard i listens on BASE_PORT + 2i, its historical opponent on BASE_PORT + 2i + 1.
BASE_PORT = 50052
RUN_STATE_VERSION = 1


def wilson_interval(successes: float, trials: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for a binomial proportion; None without trials."""
    if trials <= 0:
        return None
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _write_json(path, value) -> None:
    """Atomic replace, so a crash never leaves a half-written state file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class Collector:
    """The persistent engine plus restart-on-failure; services are restarted with it."""

    def __init__(self, script: Path, root: Path, services: ServicePool):
        self.script, self.root, self.services = script, root, services
        self.engine: PersistentEngine | None = None
        self.restarts = 0

    def run(self, config_file, timeout: float) -> tuple[int, ...]:
        if self.engine is None:
            self.engine = PersistentEngine(self.script, self.root / "engine.log")
        return self.engine.run(config_file, timeout)

    def reset(self, reason: str) -> None:
        """Kill a failed or hung JVM; services may hold open sessions, so restart them too."""
        print(f"Restarting engine and policy services: {reason}")
        self.restarts += 1
        if self.engine is not None:
            self.engine.kill()
            self.engine = None
        self.services.close()

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self.services.close()


def evaluate(
    collector: Collector,
    base: dict,
    folder: Path,
    current,
    version: str,
    benchmarks,
    games: int,
    ai_skill: int,
    actor_ports: list[int],
    digest: str,
    greedy: bool = False,
) -> dict:
    """Play fixed benchmark matches; they are never recorded as training data.

    benchmarks: (name, anchor checkpoint or None for XMage's AI, opponent deck or
    None for player_b's deck). greedy plays the learner's most likely actions.
    Score = (wins + 0.5 * (draws + truncated)) / scored games; errors are reported,
    never scored. score_wilson95 is the interval of the score itself (half points
    as half successes); win_rate_wilson95 counts only wins.
    """
    report = {}
    for name, anchor, deck in benchmarks:
        sample = not greedy
        if anchor is None:
            config = gauntlet_config(
                base, games, actor_ports, deck or base["player_b"]["deckPath"], ai_skill, sample=sample
            )
        else:
            config = game_config(base, games, actor_ports, [port + 1 for port in actor_ports], sample=sample)
        targets = {port: (current, None, version, f"actor-{i}") for i, port in enumerate(actor_ports)}
        if anchor is not None:
            _, _, catalog, data = load_checkpoint(anchor)
            if catalog.digest != digest:
                raise ValueError(f"benchmark {anchor} uses a different card catalog")
            targets.update(
                {
                    port + 1: (anchor, None, data["policy_version"], f"opponent-{i}")
                    for i, port in enumerate(actor_ports)
                }
            )
        config_file = write_game_file(folder / f"eval-{name.replace(':', '-').replace(' ', '_')}.json", config)
        started = time.perf_counter()
        try:
            collector.services.prepare(targets)
            counts = collector.run(config_file, batch_timeout(config))
        except (RuntimeError, TimeoutError, OSError, ValueError) as error:
            collector.reset(f"evaluation against {name} failed: {error}")
            report[name] = {"error": str(error)}
            continue
        outcome = collect_results(actor_ports)
        scored = outcome["win"] + outcome["loss"] + outcome["draw"] + outcome["truncated"]
        # Unfinished games count as half; errors are reported, never scored.
        points = outcome["win"] + 0.5 * (outcome["draw"] + outcome["truncated"])
        score = points / scored if scored else None
        report[name] = {
            **outcome,
            "engine": counts,
            "score": score,
            "score_wilson95": wilson_interval(points, scored),
            "win_rate_wilson95": wilson_interval(outcome["win"], scored),
            "seconds": time.perf_counter() - started,
        }
        print(f"Evaluation vs {name}: {json.dumps(report[name])}")
    return report


def primary_score(report: dict) -> float | None:
    """Selection metric: the mean score against XMage's AI (over every deck it pilots)
    when evaluated, else the mean anchor score."""
    ai = [entry.get("score") for name, entry in report.items() if name.startswith("xmage-ai")]
    scores = ai if ai else [entry.get("score") for entry in report.values()]
    scores = [score for score in scores if score is not None]
    return sum(scores) / len(scores) if scores else None


def load_game_config(game_yaml, engine_threads: int | None = None) -> dict:
    """The base game YAML with both seats piloted by the policy and absolute deck paths."""
    base = yaml.safe_load(Path(game_yaml).read_text(encoding="utf-8"))
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
    return base


def evaluate_checkpoints(
    checkpoints,
    game_yaml,
    engine_script,
    output,
    *,
    gauntlet=(),
    games: int = 256,
    ai_skill: int = 3,
    engine_threads: int | None = None,
    greedy: bool = False,
    actor_device: str = "cpu",
    actor_threads: int = 1,
) -> list[dict]:
    """Score checkpoints against XMage's AI without training them.

    Each checkpoint plays its deck (player_a of the game YAML) against XMage's AI
    with player_b's deck ("xmage-ai") and with each gauntlet deck
    ("xmage-ai:<deck>"): the same games as a self-play run's evaluation. One line
    per checkpoint is appended to output/evaluations.jsonl, and the batch
    configurations and logs go to one folder per checkpoint.
    """
    script = Path(engine_script).resolve()
    if not script.is_file():
        raise ValueError("XMage engine launcher is missing; see docs/xmage-integration.md")
    if games < 1 or not 1 <= ai_skill <= 10:
        raise ValueError("games must be positive and the AI skill in 1..10")
    if engine_threads is not None and not 1 <= engine_threads <= 16:
        raise ValueError("engine threads must be between 1 and 16")
    decks = [Path(deck).resolve() for deck in gauntlet]
    if any(not deck.is_file() for deck in decks):
        raise ValueError(f"gauntlet deck is missing: {[str(d) for d in decks if not d.is_file()]}")
    base = load_game_config(game_yaml, engine_threads)
    benchmarks = [("xmage-ai", None, None)] + [(f"xmage-ai:{deck.stem}", None, deck) for deck in decks]
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    actor_ports = [BASE_PORT]
    collector = Collector(script, root, ServicePool(root, actor_device, actor_threads))
    entries = []
    try:
        for index, checkpoint in enumerate(map(Path, checkpoints)):
            checkpoint = checkpoint.resolve()
            _, _, catalog, data = load_checkpoint(checkpoint)
            folder = root / f"{index}-{checkpoint.parent.name}-{checkpoint.stem}"
            folder.mkdir(exist_ok=True)
            report = evaluate(
                collector,
                base,
                folder,
                checkpoint,
                data["policy_version"],
                benchmarks,
                games,
                ai_skill,
                actor_ports,
                catalog.digest,
                greedy,
            )
            entry = {
                "checkpoint": str(checkpoint),
                "score": primary_score(report),
                "games": games,
                "ai_skill": ai_skill,
                "greedy": greedy,
                "report": report,
            }
            with (root / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            entries.append(entry)
    finally:
        collector.close()
    return entries


class SelfPlayRun:
    """One resumable self-play run in an output directory (see run_selfplay)."""

    def __init__(
        self,
        checkpoint,
        game_yaml,
        policy_yaml,
        engine_script,
        output,
        *,
        iterations,
        games,
        device,
        league_path,
        actor_device,
        max_seconds,
        engine_threads,
        snapshot_every,
        actor_threads,
        actor_shards,
        resume,
        eval_every,
        eval_games,
        eval_ai_skill,
        eval_anchors,
        keep_episodes,
        keep_every,
        min_free_gb,
        max_retries,
        max_failure_fraction,
        gauntlet,
        gauntlet_fraction,
        gauntlet_ai_skill,
        gauntlet_ai_think,
        eval_greedy,
        reference,
    ):
        self.script = Path(engine_script).resolve()
        if not self.script.is_file():
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
        self.gauntlet = [Path(deck).resolve() for deck in gauntlet]
        if any(not deck.is_file() for deck in self.gauntlet):
            raise ValueError(f"gauntlet deck is missing: {[str(d) for d in self.gauntlet if not d.is_file()]}")
        if self.gauntlet and not (
            0 < gauntlet_fraction <= 1 and 1 <= gauntlet_ai_skill <= 10 and gauntlet_ai_think >= 0
        ):
            raise ValueError("gauntlet fraction must be in (0, 1], its AI skill in 1..10 and think time >= 0")
        raw = yaml.safe_load(Path(policy_yaml).read_text(encoding="utf-8"))
        self.training = PPOConfig(**raw.get("training", {}))
        self.base = load_game_config(game_yaml, engine_threads)
        self.iterations, self.games, self.device, self.actor_device = iterations, games, device, actor_device
        self.max_seconds, self.snapshot_every, self.keep_episodes = max_seconds, snapshot_every, keep_episodes
        self.eval_every, self.eval_games, self.eval_ai_skill = eval_every, eval_games, eval_ai_skill
        self.eval_greedy, self.keep_every, self.min_free_gb = eval_greedy, keep_every, min_free_gb
        self.max_retries, self.max_failure_fraction = max_retries, max_failure_fraction
        self.gauntlet_fraction, self.gauntlet_ai_skill = gauntlet_fraction, gauntlet_ai_skill
        self.gauntlet_ai_think = gauntlet_ai_think

        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "run-state.json"
        self.state = self._load_state(checkpoint, resume, league_path, reference)
        self.league_file = Path(self.state["league"])
        self.league = League.load(self.league_file) if self.league_file.exists() else League()
        _, _, self.initial_catalog, _ = load_checkpoint(self.state["initial_checkpoint"])
        self.benchmarks = (
            ([("xmage-ai", None, None)] if eval_ai_skill else [])
            + [(f"xmage-ai:{deck.stem}", None, deck) for deck in (self.gauntlet if eval_ai_skill else [])]
            + [(f"anchor:{Path(anchor).stem}", Path(anchor).resolve(), None) for anchor in eval_anchors]
        )
        if eval_every and not self.benchmarks:
            raise ValueError("evaluation needs the XMage AI (skill > 0) or at least one anchor checkpoint")
        self.current = Path(self.state["current"])
        self.session_started = time.perf_counter()
        self.session_durations: list[float] = []
        self.actor_ports = [BASE_PORT + 2 * i for i in range(actor_shards)]
        self.collector = Collector(self.script, self.root, ServicePool(self.root, actor_device, actor_threads))

    def _load_state(self, checkpoint, resume: bool, league_path, reference) -> dict:
        if self.state_file.exists() and not resume:
            raise ValueError(
                f"{self.root} holds an existing run; pass --resume to continue it or choose a fresh output"
            )
        if resume and self.state_file.exists():
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            if state.get("format_version") != RUN_STATE_VERSION:
                raise ValueError("incompatible run state")
            print(f"Resuming at generation {state['next_generation']} from {state['current']}")
        else:
            if any(self.root.glob("generation_*")):
                raise ValueError(f"{self.root} has generations but no run state; choose a fresh output directory")
            state = {
                "format_version": RUN_STATE_VERSION,
                "initial_checkpoint": str(Path(checkpoint).resolve()),
                "current": str(Path(checkpoint).resolve()),
                "next_generation": 0,
                "durations": [],
                "best": None,
                "league": str(Path(league_path).resolve() if league_path else self.root / "league.json"),
                "reference": str(Path(reference).resolve()) if reference else None,
            }
        if reference and state.get("reference") != str(Path(reference).resolve()):
            raise ValueError("--reference differs from the reference this run started with")
        if self.training.reference_kl_coefficient > 0 and not state.get("reference"):
            raise ValueError(
                "training.reference_kl_coefficient > 0 needs --reference (the behaviour-cloned checkpoint)"
            )
        if state.get("reference") and not Path(state["reference"]).is_file():
            raise ValueError(f"reference checkpoint is missing: {state['reference']}")
        return state

    def _save_state(self) -> None:
        self.state["current"] = str(self.current)
        _write_json(self.state_file, self.state)
        _write_json(self.root / "run-summary.json", {
            "initial_checkpoint": self.state["initial_checkpoint"],
            "latest_checkpoint": str(self.current),
            "completed_generations": self.state["next_generation"],
            "best": self.state["best"],
            "engine_restarts_this_session": self.collector.restarts,
            "session_wall_seconds": time.perf_counter() - self.session_started,
            "requested_budget_seconds": self.max_seconds,
            "learner_device": self.device,
            "actor_device": self.actor_device,
            "scope": "budget estimated between complete generations; no partial games relabelled as draws",
        })  # fmt: skip

    def _should_stop(self) -> bool:
        """Stop between whole generations at the time budget or the disk guard.

        A generation is not started when its recent measured duration would exceed
        the budget; individual game lengths still make this an estimate.
        """
        elapsed = time.perf_counter() - self.session_started
        recent = self.session_durations[-3:]
        estimate = sum(recent) / len(recent) if recent else 0
        if self.max_seconds is not None and (
            elapsed >= self.max_seconds or (recent and elapsed + estimate > self.max_seconds)
        ):
            print(f"Time budget: stopped between generations after {elapsed:.1f}s; next estimate={estimate:.1f}s")
            return True
        free_gb = shutil.disk_usage(self.root).free / 2**30
        if free_gb < self.min_free_gb:
            print(f"Disk guard: stopped with {free_gb:.1f} GB free (< {self.min_free_gb} GB); free space and --resume")
            return True
        return False

    def _fresh_folder(self, generation: int) -> Path:
        folder = self.root / f"generation_{generation:04d}"
        if folder.exists():
            # Left by an interrupted attempt: its data never reached a checkpoint.
            failed = self.root / f"{folder.name}.incomplete-{int(time.time())}"
            shutil.rmtree(folder / "episodes", ignore_errors=True)
            folder.rename(failed)
            print(f"Moved incomplete {folder.name} to {failed.name}")
        folder.mkdir()
        return folder

    def _plan_batches(
        self, generation: int, folder: Path, learner_version: str, opponent: Path, opponent_version: str
    ) -> tuple[list, list[int], int]:
        """The generation's batches: (kind, config, config file, service targets) each.

        Mirror/league games come first, then a gauntlet_fraction share of games
        against the next gauntlet deck. The engine routes game g to shard
        g mod n for both of its players.
        """
        episodes = folder / "episodes"
        opponent_ports = self.actor_ports if opponent == self.current else [port + 1 for port in self.actor_ports]
        games = self.games
        gauntlet_games = min(games, round(games * self.gauntlet_fraction)) if self.gauntlet else 0
        batches = []
        if games - gauntlet_games:
            config = game_config(self.base, games - gauntlet_games, self.actor_ports, opponent_ports)
            config_file = write_game_file(folder / "game.json", config)
            targets = {
                port: (self.current, episodes / "selfplay", learner_version, f"actor-{i}")
                for i, port in enumerate(self.actor_ports)
            }
            if opponent != self.current:
                targets.update(
                    {port: (opponent, None, opponent_version, f"opponent-{i}") for i, port in enumerate(opponent_ports)}
                )
            batches.append(("selfplay", config, config_file, targets))
        if gauntlet_games:
            deck = self.gauntlet[generation % len(self.gauntlet)]
            config = gauntlet_config(
                self.base,
                gauntlet_games,
                self.actor_ports,
                deck,
                self.gauntlet_ai_skill,
                think_seconds=self.gauntlet_ai_think,
            )
            config_file = write_game_file(folder / "gauntlet.json", config)
            targets = {
                port: (self.current, episodes / "gauntlet", learner_version, f"actor-{i}")
                for i, port in enumerate(self.actor_ports)
            }
            batches.append((f"gauntlet:{deck.stem}", config, config_file, targets))
        return batches, opponent_ports, gauntlet_games

    def _collect(
        self, generation: int, batches: list, episodes: Path, opponent_ports: list[int], with_opponent: bool
    ) -> tuple[list[int], dict, dict, list[Path]]:
        """Play every batch; a failed or hung batch restarts everything and replays the generation."""
        for attempt in range(self.max_retries + 1):
            try:
                counts, by_kind, stats = [0] * len(RESULT_FIELDS), {}, {}
                for kind, config, config_file, targets in batches:
                    self.collector.services.prepare(targets)
                    result = self.collector.run(config_file, batch_timeout(config))
                    counts = [x + y for x, y in zip(counts, result)]
                    # Service counters restart at each reload, so these are this batch's.
                    by_kind[kind] = {
                        **dict(zip(RESULT_FIELDS, result)),
                        "learner_results": collect_results(self.actor_ports),
                    }
                    stats[kind] = {"actor": [service_stats(port) for port in self.actor_ports]}
                    if kind == "selfplay" and with_opponent:
                        stats[kind]["opponent"] = [service_stats(port) for port in opponent_ports]
                shards = sorted(episodes.glob("*/*.jsonl"))
                if not shards:
                    raise RuntimeError("engine emitted no trajectories (every game failed)")
                return counts, by_kind, stats, shards
            except (RuntimeError, TimeoutError, OSError) as error:
                self.collector.reset(f"generation {generation} attempt {attempt + 1} failed: {error}")
                shutil.rmtree(episodes, ignore_errors=True)
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"generation {generation} failed {self.max_retries + 1} times; "
                        f"inspect {self.root / 'engine.log'} and rerun with --resume"
                    ) from error
        raise AssertionError("unreachable")

    def _evaluate(self, generation: int, folder: Path, digest: str) -> dict:
        """Benchmark the new checkpoint; keep it as best.pt if it beats the best so far."""
        report = evaluate(
            self.collector,
            self.base,
            folder,
            self.current,
            read_policy_version(self.current),
            self.benchmarks,
            self.eval_games,
            self.eval_ai_skill,
            self.actor_ports,
            digest,
            self.eval_greedy,
        )
        score = primary_score(report)
        entry = {"generation": generation, "checkpoint": str(self.current), "score": score, "report": report}
        with (self.root / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        if score is not None and (self.state["best"] is None or score > self.state["best"]["score"]):
            best = strip_optimizer(self.current, self.root / "best.pt")
            self.state["best"] = {
                "generation": generation,
                "score": score,
                "checkpoint": str(best),
                "source": str(self.current),
            }
            print(f"New best policy: generation {generation}, score {score:.3f}")
        return report

    def _prune(self, previous: Path) -> None:
        """Delete the checkpoint just trained from unless it is a milestone.

        League snapshots and best.pt are separate optimizer-free copies.
        """
        if (
            previous.parent.parent == self.root
            and previous.name == "policy.pt"
            and int(previous.parent.name.split("_")[1]) % self.keep_every != 0
            and str(previous.resolve()) not in {x.checkpoint for x in self.league.opponents}
        ):
            previous.unlink(missing_ok=True)

    def _generation(self, generation: int) -> dict:
        """Collect one generation with fixed weights, train on it, and maybe evaluate."""
        generation_started = time.perf_counter()
        folder = self._fresh_folder(generation)
        _, _, catalog, learner = load_checkpoint(self.current)
        # Alternate latest-policy self-play and historical league games.
        opponent = Path(self.league.sample()) if self.league.opponents and generation % 2 else self.current
        _, _, opponent_catalog, opponent_data = load_checkpoint(opponent)
        if catalog.digest != opponent_catalog.digest or catalog.digest != self.initial_catalog.digest:
            raise ValueError("all league policies must use the same immutable card catalog")
        episodes = folder / "episodes"
        batches, opponent_ports, gauntlet_games = self._plan_batches(
            generation, folder, learner["policy_version"], opponent, opponent_data["policy_version"]
        )
        print(
            f"Generation {generation}: {self.games} games ({gauntlet_games} gauntlet), "
            f"learner={learner['policy_version']}, opponent={opponent}"
        )
        collection_started = time.perf_counter()
        counts, by_kind, stats, shards = self._collect(
            generation, batches, episodes, opponent_ports, opponent != self.current
        )
        outcome = dict(zip(RESULT_FIELDS, counts))
        failures = outcome["errors"] + outcome["timeouts"]
        next_checkpoint = folder / "policy.pt"
        collection_seconds = time.perf_counter() - collection_started
        training_started = time.perf_counter()
        metrics = train_from_files(
            self.current, shards, next_checkpoint, self.training, self.device, reference=self.state.get("reference")
        )
        training_seconds = time.perf_counter() - training_started
        records = [json.loads(shard.read_text(encoding="utf-8")) for shard in shards]
        decisions = sum(len(record["steps"]) for record in records)
        metrics.update(
            collection_seconds=collection_seconds,
            training_seconds=training_seconds,
            collected_decisions=decisions,
            games=outcome,
            failure_fraction=failures / outcome["requested"],
            truncation_fraction=outcome["truncated"] / outcome["requested"],
            completed_games=outcome["terminal"],
            games_by_kind=by_kind,
            collection_games_per_hour=3600 * outcome["terminal"] / collection_seconds,
            collected_decisions_per_second=decisions / collection_seconds,
            training_decision_passes_per_second=metrics["decisions"] / training_seconds,
            requested_games=self.games,
            actor_device=self.actor_device,
            learner_device=self.device,
            engine_restarts=self.collector.restarts,
            policy_services=stats,
        )
        # League results are terminal self-play outcomes, from the learner's side.
        if opponent != self.current:
            for shard, record in zip(shards, records):
                if record["terminal"] and shard.parent.name == "selfplay":
                    self.league.record(opponent, record["steps"][-1]["reward"])
        # Sparse snapshots: one per generation would fill the pool with nearly
        # identical policies whose win-rate estimates never leave their prior.
        if generation % self.snapshot_every == 0:
            snapshot = self.root / "snapshots" / f"generation_{generation:04d}.pt"
            self.league.add(strip_optimizer(self.current, snapshot))
        self.league.save(self.league_file)
        previous, self.current = self.current, next_checkpoint
        if self.eval_every and ((generation + 1) % self.eval_every == 0 or generation + 1 == self.iterations):
            metrics["evaluation"] = self._evaluate(generation, folder, catalog.digest)
        metrics["generation_seconds"] = time.perf_counter() - generation_started
        _write_json(folder / "metrics.json", metrics)
        if not self.keep_episodes:
            shutil.rmtree(episodes, ignore_errors=True)
        self._prune(previous)
        return metrics

    def run(self) -> Path:
        """Run generations until `iterations` exist; return the latest checkpoint."""
        try:
            while self.state["next_generation"] < self.iterations:
                if self._should_stop():
                    break
                generation = self.state["next_generation"]
                metrics = self._generation(generation)
                self.state["next_generation"] = generation + 1
                self.state["durations"].append(metrics["generation_seconds"])
                self.session_durations.append(metrics["generation_seconds"])
                self._save_state()
                print(
                    f"Saved {self.current}; {json.dumps({k: v for k, v in metrics.items() if k != 'policy_services'})}"
                )
                if metrics["failure_fraction"] > self.max_failure_fraction:
                    games = metrics["games"]
                    raise RuntimeError(
                        f"{games['errors'] + games['timeouts']}/{games['requested']} games failed in generation "
                        f"{generation}; training on the survivors would bias the data. Inspect "
                        f"{self.root / 'engine.log'}, fix the cause, then --resume"
                    )
        finally:
            self.collector.close()
        return self.current


def run_selfplay(
    checkpoint,
    game_config,
    policy_config,
    engine_script,
    output,
    iterations=10,
    games=32,
    device="auto",
    league_path=None,
    actor_device="cpu",
    max_seconds=None,
    engine_threads=None,
    snapshot_every=5,
    actor_threads=1,
    actor_shards=1,
    resume=False,
    eval_every=0,
    eval_games=64,
    eval_ai_skill=6,
    eval_anchors=(),
    keep_episodes=False,
    keep_every=50,
    min_free_gb=20.0,
    max_retries=3,
    max_failure_fraction=0.25,
    gauntlet=(),
    gauntlet_fraction=0.5,
    gauntlet_ai_skill=3,
    gauntlet_ai_think=0,
    eval_greedy=False,
    reference=None,
) -> Path:
    """Run collect/learn generations until `iterations` generations exist in `output`.

    Long-run safety: a failed or hung engine batch restarts the engine and services
    and replays the generation (up to max_retries); individual game failures are
    counted, not fatal. `resume=True` continues an interrupted run from its state
    file. Episodes are deleted after training unless kept, checkpoints are pruned
    to league snapshots, the best evaluated policy, and every keep_every-th one.

    Gauntlet: a `gauntlet_fraction` share of each generation's games puts the
    learner's deck (player_a) against one of the `gauntlet` decks piloted by
    XMage's AI, rotating through them by generation. The rest are mirror/league
    games. Only the learner's side is recorded in gauntlet games.

    reference: the frozen behaviour-cloned checkpoint for the PPO KL term
    (training.reference_kl_coefficient); kept in the run state across resumes.
    """
    return SelfPlayRun(
        checkpoint, game_config, policy_config, engine_script, output, iterations=iterations, games=games,
        device=device, league_path=league_path, actor_device=actor_device, max_seconds=max_seconds,
        engine_threads=engine_threads, snapshot_every=snapshot_every, actor_threads=actor_threads,
        actor_shards=actor_shards, resume=resume, eval_every=eval_every, eval_games=eval_games,
        eval_ai_skill=eval_ai_skill, eval_anchors=eval_anchors, keep_episodes=keep_episodes, keep_every=keep_every,
        min_free_gb=min_free_gb, max_retries=max_retries, max_failure_fraction=max_failure_fraction,
        gauntlet=gauntlet, gauntlet_fraction=gauntlet_fraction, gauntlet_ai_skill=gauntlet_ai_skill,
        gauntlet_ai_think=gauntlet_ai_think, eval_greedy=eval_greedy, reference=reference,
    ).run()  # fmt: skip
