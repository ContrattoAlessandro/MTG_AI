"""Behaviour-cloning warm start from XMage's own AI (AlphaStar's supervised stage).

collect_teacher_games plays games in which XMage's Mad AI (ComputerPlayer7) decides
while TeacherPlayer records each decision as a label over the policy's own
candidate list, with the observation the policy would have seen (POST /record).
behavior_cloning then fits the policy to those labels by cross-entropy over the
legal candidates, rebuilding recurrent memory from step zero exactly as PPO does,
and pre-trains both value heads on the games' (shaped) outcomes. The resulting
checkpoint starts PPO and serves as the frozen reference of its KL term.

The teacher plays deterministically and its search fixes some choices in advance
(e.g. targets chosen inside a simulated cast); such decisions are played but not
recorded, so teacher trajectories can skip decisions a policy game would contain.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import torch
import yaml

from .belief import constrained_hand_belief
from .checkpoint import load_checkpoint, resolve_device, save_checkpoint
from .engine import PersistentEngine
from .learner import _carry, _join, _prepare_chunks, _windows, clip_gradients, validate_episode
from .schema import ACTION_TYPES
from .service import TEACHER_VERSION
from .shaping import shaped_rewards


@dataclass(frozen=True)
class CloneConfig:
    learning_rate: float = 3e-4
    epochs: int = 10
    episodes_per_batch: int = 16
    sequence_length: int = 32
    # Value pre-training on the teacher games' outcomes, so the critic is not
    # random when PPO starts bootstrapping (gae_lambda < 1).
    value_coefficient: float = 0.5
    public_value_coefficient: float = 0.25
    belief_coefficient: float = 0.0
    max_grad_norm: float = 1.0
    # Whole games (both seats together) held out to select the best epoch.
    validation_fraction: float = 0.1
    max_episode_steps: int = 10000
    # Must match the PPO run's training.shaping_coefficient and gamma, so the
    # pre-trained values estimate the same (shaped) returns PPO will regress on.
    shaping_coefficient: float = 0.0
    gamma: float = 1.0
    seed: int = 0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"clone.{key} must be a finite number")
        for key in ("epochs", "episodes_per_batch", "sequence_length", "max_episode_steps", "seed"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"clone.{key} must be an integer")
        if self.learning_rate <= 0 or self.epochs < 1 or self.episodes_per_batch < 1 or self.sequence_length < 1:
            raise ValueError("clone learning rate, epochs, batch and sequence length must be positive")
        if not 0 <= self.validation_fraction < 1 or not 0 < self.gamma <= 1 or self.max_grad_norm <= 0:
            raise ValueError("invalid clone validation fraction, gamma, or gradient norm")
        if min(self.value_coefficient, self.public_value_coefficient, self.belief_coefficient,
               self.shaping_coefficient) < 0:
            raise ValueError("clone loss coefficients cannot be negative")


def _returns(episode, config: CloneConfig) -> tuple[torch.Tensor, bool]:
    """Discounted shaped returns; truncated games have none (no result to regress on)."""
    rewards = shaped_rewards(episode, config.shaping_coefficient, config.gamma)
    returns, running = torch.zeros_like(rewards), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + config.gamma * running
        returns[t] = running
    return returns, not episode.get("truncated", False)


@dataclass
class _Entry:
    """Where one teacher trajectory lives on disk, and its small training targets."""
    path: Path
    offset: int
    game_id: str
    returns: torch.Tensor
    valued: bool


class _TeacherGames:
    """Teacher trajectories indexed on disk; a minibatch is parsed and collated when used.

    2,000 Commander games are about 450,000 decisions (~8 GB of JSON). Parsed and
    collated all at once they exceed this machine's RAM and VRAM, so the index keeps
    only file offsets and return targets, and one minibatch (plus the next, being
    prefetched) is resident at a time.
    """
    def __init__(self, files, model_config, config: CloneConfig):
        self.entries, keys = [], set()
        for path in map(Path, files):
            with path.open("rb") as stream:
                offset = 0
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            episode = validate_episode(json.loads(line), model_config, TEACHER_VERSION,
                                                       config.max_episode_steps)
                            first = episode["steps"][0]["observation"]
                            key = (first["game_id"], first["player_id"])
                            if key in keys:
                                raise ValueError("duplicate game/player trajectory")
                        except (ValueError, TypeError, KeyError) as error:
                            raise ValueError(f"{path}:{line_number}: {error}") from error
                        keys.add(key)
                        returns, valued = _returns(episode, config)
                        self.entries.append(_Entry(path, offset, first["game_id"], returns, valued))
                    offset += len(line)
        if not self.entries:
            raise ValueError("no completed teacher episodes were found")

    def __len__(self):
        return len(self.entries)

    def load(self, index: int) -> dict:
        entry = self.entries[index]
        with entry.path.open("rb") as stream:
            stream.seek(entry.offset)
            return json.loads(stream.readline())


def _split(games: _TeacherGames, fraction, seed) -> tuple[list[int], list[int]]:
    """Indices of training and held-out trajectories; whole games (both seats) are held out."""
    names = sorted({entry.game_id for entry in games.entries})
    random.Random(seed).shuffle(names)
    held = set(names[:round(len(names) * fraction)]) if len(names) > 1 else set()
    train = [i for i, entry in enumerate(games.entries) if entry.game_id not in held]
    validation = [i for i, entry in enumerate(games.entries) if entry.game_id in held]
    return train, validation


def _pass(model, critic, games: _TeacherGames, indices, config: CloneConfig, optimizer=None):
    """One sweep over `indices`; trains when an optimizer is given. Returns summed statistics."""
    length = config.sequence_length
    trains_belief = model.config.belief_mode == "learned" or model.config.auxiliary_belief
    device = next(model.parameters()).device
    totals = {"nll": 0.0, "correct": 0.0, "choices": 0, "value_error": 0.0, "valued": 0, "belief_nll": 0.0,
              "decisions": 0}
    by_type = {name: [0, 0] for name in ACTION_TYPES}
    order = list(indices)
    if optimizer is not None:
        random.shuffle(order)
    groups = [order[start:start + config.episodes_per_batch]
              for start in range(0, len(order), config.episodes_per_batch)]

    def prepare(group):
        return {i: _prepare_chunks(games.load(i), model, length) for i in group}
    with ThreadPoolExecutor(1) as loader:  # parse and collate the next minibatch meanwhile
        pending = loader.submit(prepare, groups[0]) if groups else None
        for position, group in enumerate(groups):
            prepared = pending.result()
            pending = loader.submit(prepare, groups[position + 1]) if position + 1 < len(groups) else None
            choice_denominator = max(sum(int(c.choice.sum()) for i in group for c in prepared[i]), 1)
            step_denominator = max(sum(len(games.entries[i].returns) for i in group), 1)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            memory, previous = None, group
            with torch.set_grad_enabled(optimizer is not None):
                for window, members in _windows(group, prepared):
                    memory = _carry(memory, previous, members)
                    joined = _join([prepared[i][window] for i in members])
                    sizes = [len(prepared[i][window].actions) for i in members]
                    rows = slice(window * length, window * length + length)
                    returns = torch.cat([games.entries[i].returns[rows] for i in members]).to(device)
                    valued = torch.cat([torch.full((sizes[k],), float(games.entries[i].valued))
                                        for k, i in enumerate(members)]).to(device)
                    out = model.forward_episodes(joined.batch, sizes, memory, joined.event_counts)
                    memory, previous = out.memory.detach(), members
                    choice = joined.choice.float()
                    log_probability = out.distribution().log_prob(joined.actions)
                    nll = -log_probability * choice
                    critic_error = (critic(out.context, joined.batch.deck_cards, joined.labels.float())
                                    - returns).square()
                    public_error = (out.value - returns).square()
                    # Opponent-hand NLL per deck kind: trained in the learned and auxiliary
                    # modes, a diagnostic of the fixed prior otherwise.
                    belief = out.belief
                    if out.auxiliary_count_logits is not None:
                        belief = constrained_hand_belief(out.auxiliary_count_logits, joined.batch.deck_lower,
                                                         joined.batch.deck_upper, joined.batch.hand_size)
                    belief_loss = belief.nll(joined.labels) / joined.deck_kinds
                    if not trains_belief:
                        belief_loss = belief_loss.detach()
                    if optimizer is not None:
                        loss = (nll.sum() / choice_denominator
                                + (config.value_coefficient * (critic_error * valued).sum()
                                   + config.public_value_coefficient * (public_error * valued).sum()
                                   + config.belief_coefficient * belief_loss.sum()) / step_denominator)
                        loss.backward()
                    with torch.no_grad():
                        correct = (out.logits.argmax(-1) == joined.actions).float() * choice
                        totals["nll"] += float(nll.sum())
                        totals["correct"] += float(correct.sum())
                        totals["choices"] += int(choice.sum())
                        totals["value_error"] += float((public_error * valued).sum())
                        totals["valued"] += int(valued.sum())
                        totals["belief_nll"] += float(belief_loss.sum())
                        totals["decisions"] += len(joined.actions)
                        kinds = joined.batch.action_types.gather(1, joined.actions[:, None]).squeeze(1)
                        for kind, hit, counted in zip(kinds.tolist(), correct.tolist(), choice.tolist()):
                            if counted:
                                by_type[ACTION_TYPES[kind]][0] += hit
                                by_type[ACTION_TYPES[kind]][1] += 1
            if optimizer is not None:
                clip_gradients(model, critic, config.max_grad_norm)
                optimizer.step()
    return totals, by_type


def _summary(totals, by_type):
    choices = max(totals["choices"], 1)
    return {"nll": totals["nll"] / choices, "accuracy": totals["correct"] / choices,
            "choice_decisions": totals["choices"],
            "public_value_mse": totals["value_error"] / max(totals["valued"], 1),
            "hand_nll_per_card": totals["belief_nll"] / max(totals["decisions"], 1),
            "accuracy_by_type": {name: hits / count for name, (hits, count) in by_type.items() if count}}


def behavior_cloning(checkpoint_path, episode_paths, output_path, config=CloneConfig(), device="auto"):
    """Fit the checkpoint's policy (and value heads) to teacher trajectories; save the best epoch."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model, critic, catalog, checkpoint = load_checkpoint(checkpoint_path, resolve_device(device))
    files = []
    for path in map(Path, episode_paths):
        files += sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    games = _TeacherGames(files, model.config, config)
    train_ids, validation_ids = _split(games, config.validation_fraction, config.seed)
    if not train_ids:
        raise ValueError("no teacher games left for training after the validation split")
    # As in PPO: deterministic attention, training-mode GRUs for cuDNN backward.
    model.eval()
    model.event_memory.train()
    model.decision_memory.train()
    critic.train()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()),
                                  lr=config.learning_rate, weight_decay=0.0)
    history, best = [], None
    started = time.perf_counter()
    for epoch in range(config.epochs):
        trained = _summary(*_pass(model, critic, games, train_ids, config, optimizer))
        held = _summary(*_pass(model, critic, games, validation_ids, config)) if validation_ids else None
        history.append({"epoch": epoch, "train": trained, "validation": held})
        print(f"clone epoch {epoch}: train nll {trained['nll']:.4f} acc {trained['accuracy']:.3f}"
              + (f"; validation nll {held['nll']:.4f} acc {held['accuracy']:.3f}" if held else ""))
        score = (held or trained)["nll"]
        if best is None or score < best[0]:
            best = (score, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()},
                    {k: v.detach().clone() for k, v in critic.state_dict().items()})
    _, best_epoch, model_state, critic_state = best
    model.load_state_dict(model_state)
    critic.load_state_dict(critic_state)
    metrics = {"behavior_cloning": {"config": asdict(config), "best_epoch": best_epoch, "history": history,
               "train_episodes": len(train_ids), "validation_episodes": len(validation_ids),
               "seconds": time.perf_counter() - started, "source_checkpoint": str(checkpoint_path)}}
    # No optimizer state: PPO starts its own AdamW moments from the warm start.
    version = save_checkpoint(output_path, model, critic, catalog, updates=checkpoint["updates"], metrics=metrics)
    metrics["policy_version"] = version
    return metrics


def teacher_game_config(base: dict, games: int, port: int, ai_skill: int, ai_think: int, seats: str) -> dict:
    """Both seats (or only player_a's deck) played by recording XMage AI teachers."""
    if seats not in ("both", "a"):
        raise ValueError("teacher seats must be 'both' or 'a'")
    from .runner import _game_config
    config = _game_config(base, games, [port], [port], keep_deck=True)
    config["player_a"]["type"] = "xmage-teacher"
    config["player_b"]["type"] = "xmage-teacher" if seats == "both" else "xmage-ai"
    config["opponent_ai"] = {"skill": ai_skill, **({"think_seconds": ai_think} if ai_think else {})}
    return config


def collect_teacher_games(checkpoint, game_config, engine_script, output, games, *, batch_games=32,
                          engine_threads=None, ai_skill=1, ai_think=0, seats="both", port=50060,
                          max_retries=3):
    """Play `games` teacher games into output/episodes; resumable by counting what is there.

    The checkpoint only supplies the card catalog and validation limits for the
    service; it never chooses an action.
    """
    from .runner import _await_server, _batch_timeout, _launch_server, _results, _stop_server
    script = Path(engine_script).resolve()
    if not script.is_file():
        raise ValueError("XMage engine launcher is missing; see docs/xmage-integration.md")
    if games < 1 or batch_games < 1 or not 1 <= ai_skill <= 10 or ai_think < 0:
        raise ValueError("games and batch size must be positive, AI skill 1..10, think time >= 0")
    base = yaml.safe_load(Path(game_config).read_text(encoding="utf-8"))
    for player in ("player_a", "player_b"):
        base[player]["deckPath"] = str(Path(base[player]["deckPath"]).resolve())
        if not Path(base[player]["deckPath"]).is_file():
            raise ValueError(f"deck is missing: {base[player]['deckPath']}")
    if engine_threads is not None:
        base["training"]["threads"] = engine_threads
    root = Path(output).resolve()
    episodes = root / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    summary_file = root / "teacher-summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {
        "format_version": 1, "games_played": 0, "batches": [], "checkpoint": str(Path(checkpoint).resolve()),
        "ai_skill": ai_skill, "ai_think": ai_think, "seats": seats}
    version = load_checkpoint(checkpoint)[3]["policy_version"]
    engine = server = None
    counted = dict.fromkeys(("win", "loss", "draw", "truncated", "aborted"), 0)  # service totals so far

    def start():
        nonlocal engine, server, counted
        server = _launch_server(checkpoint, port, episodes, "cpu", root / "teacher-service.log")
        _await_server(server, version)
        counted = dict.fromkeys(counted, 0)
        engine = PersistentEngine(script, root / "teacher-engine.log")

    def stop():
        nonlocal engine, server
        if engine is not None:
            engine.kill()
            engine = None
        if server is not None:
            _stop_server(server)
            server = None

    try:
        while summary["games_played"] < games:
            count = min(batch_games, games - summary["games_played"])
            config = teacher_game_config(base, count, port, ai_skill, ai_think, seats)
            config_file = root / "teacher-game.yml"
            config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            for attempt in range(max_retries + 1):
                try:
                    if engine is None:
                        start()
                    started = time.perf_counter()
                    result = engine.run(config_file, _batch_timeout(config))
                    break
                except (RuntimeError, TimeoutError, OSError) as error:
                    print(f"Teacher batch failed (attempt {attempt + 1}): {error}; restarting engine and service")
                    stop()
                    if attempt == max_retries:
                        raise
            outcome = dict(zip(("terminal", "requested", "truncated", "errors", "timeouts"), result))
            # Per-recorder outcomes of this batch: the service counts since its start.
            totals = _results([port])
            outcome["recorders"] = {key: totals[key] - counted[key] for key in counted}
            counted = totals
            outcome["seconds"] = time.perf_counter() - started
            summary["games_played"] += count
            summary["batches"].append(outcome)
            summary["episodes"] = len(list(episodes.glob("*.jsonl")))
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Teacher games {summary['games_played']}/{games}: {json.dumps(outcome)}; "
                  f"{summary['episodes']} trajectories")
    finally:
        stop()
    return summary
