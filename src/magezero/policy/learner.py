"""On-policy recurrent PPO with a training-only critic and hand-belief labels.

Episodes are trajectories for ONE player's successive decisions, not alternating
global game turns. The engine queues that player's rewards until its next decision.
No MCTS labels, old replay buffer, heuristic search, or opponent-hand actor inputs.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .belief import constrained_hand_belief
from .checkpoint import load_checkpoint, resolve_device, save_checkpoint
from .model import PolicyMemory
from .schema import (SCHEMA_VERSION, ObservationBatch, collate_observations, concat_batches, pad_cat,
                     validate_labels, validate_observation)


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    epochs: int = 4
    episodes_per_batch: int = 4
    sequence_length: int = 32
    gamma: float = 1.0
    gae_lambda: float = 0.97
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    public_value_coefficient: float = 0.25
    belief_coefficient: float = 0.1
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    max_episode_steps: int = 10000
    log_probability_tolerance: float = 0.005
    # Trajectories per forward pass when computing PPO targets (no gradients).
    target_batch_episodes: int = 64

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        for key in ("epochs", "episodes_per_batch", "sequence_length", "max_episode_steps",
                    "target_batch_episodes"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid return/GAE discounts")
        if any(getattr(self, key) <= 0 for key in ("learning_rate", "epochs", "episodes_per_batch",
                "sequence_length", "clip_ratio", "max_grad_norm", "max_episode_steps",
                "target_batch_episodes")):
            raise ValueError("PPO sizes, optimizer parameters, and clip ratio must be positive")
        if any(getattr(self, key) < 0 for key in ("value_coefficient", "public_value_coefficient",
                "belief_coefficient", "entropy_coefficient", "target_kl", "log_probability_tolerance")):
            raise ValueError("loss coefficients and tolerances cannot be negative")


def generalized_advantages(rewards: Tensor, values: Tensor, terminated: Tensor,
                           truncated: Tensor | None = None, *, bootstrap_value=0.0,
                           gamma=1.0, gae_lambda=0.97) -> tuple[Tensor, Tensor]:
    """Bootstrap time limits, but stop recursion at both kinds of episode boundary."""
    if rewards.ndim != 1 or values.shape != rewards.shape or terminated.shape != rewards.shape:
        raise ValueError("GAE expects equally sized vectors for one player trajectory")
    truncated = torch.zeros_like(terminated) if truncated is None else truncated
    advantage = torch.zeros_like(values)
    following_value = torch.as_tensor(bootstrap_value, device=values.device, dtype=values.dtype)
    running = torch.zeros_like(following_value)
    for t in range(len(rewards) - 1, -1, -1):
        nonterminal = (~terminated[t].bool()).to(values.dtype)
        continuation = (~(terminated[t].bool() | truncated[t].bool())).to(values.dtype)
        delta = rewards[t] + gamma * following_value * nonterminal - values[t]
        running = delta + gamma * gae_lambda * continuation * running
        advantage[t] = running
        following_value = values[t]
    return advantage, advantage + values


EPISODE_FIELDS = {"schema_version", "policy_version", "steps", "terminal"}


def validate_episode(episode: dict, config, policy_version: str, max_steps=10000) -> dict:
    """A completed game, or an error-free game stopped by the turn limit.

    Truncated trajectories carry ``truncated: true``, zero rewards, and full labels;
    the learner bootstraps them from the critic instead of scoring them as draws.
    """
    if not isinstance(episode, dict) or set(episode) - {"truncated"} != EPISODE_FIELDS:
        raise ValueError("invalid episode fields")
    truncated = episode.get("truncated", False)
    if episode["schema_version"] != SCHEMA_VERSION or not isinstance(truncated, bool):
        raise ValueError("learner requires v2 episodes")
    if not isinstance(episode["terminal"], bool) or episode["terminal"] is truncated:
        raise ValueError("an episode must be exactly one of terminal or truncated")
    if episode["policy_version"] != policy_version:
        raise ValueError("stale/off-policy trajectories: collect new games from the input checkpoint")
    steps = episode["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= max_steps:
        raise ValueError("invalid trajectory length")
    key, cursor = None, -1
    for t, step in enumerate(steps):
        if set(step) != {"observation", "action_id", "log_probability", "reward", "labels"}:
            raise ValueError("invalid transition fields")
        obs = validate_observation(step["observation"], config)
        new_key = (obs["game_id"], obs["player_id"])
        if key is not None and key != new_key:
            raise ValueError("a recurrent trajectory cannot mix players or games")
        key = new_key
        if obs["step"] != t:
            raise ValueError("trajectories must contain every player decision from step zero")
        if obs["events"]:
            if obs["events"][0]["seq"] <= cursor:
                raise ValueError("a history event was replayed more than once")
            cursor = obs["events"][-1]["seq"]
        validate_labels(step["labels"], obs)
        if step["action_id"] not in {a["id"] for a in obs["actions"]}:
            raise ValueError("recorded action is not legal in its recorded observation")
        for field in ("log_probability", "reward"):
            value = step[field]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"invalid {field}")
        if step["log_probability"] > 1e-6:
            raise ValueError("behavior log probabilities must be <= 0")
        if step["reward"] != 0 and t != len(steps) - 1:
            raise ValueError("the initial learner uses terminal game rewards only")
    if steps[-1]["reward"] not in (-1, 0, 1):
        raise ValueError("terminal reward must be win (+1), loss (-1), or a real draw (0)")
    if truncated and steps[-1]["reward"] != 0:
        raise ValueError("a truncated game has no result; it must not carry a reward")
    return episode


def load_episodes(paths, model_config, policy_version, max_steps=10000):
    episodes, keys = [], set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    episode = validate_episode(json.loads(line), model_config, policy_version, max_steps)
                    obs = episode["steps"][0]["observation"]
                    key = (obs["game_id"], obs["player_id"])
                    if key in keys:
                        raise ValueError("duplicate game/player trajectory")
                    keys.add(key)
                    episodes.append(episode)
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not episodes:
        raise ValueError("no completed on-policy episodes were found")
    return episodes


@dataclass
class _Chunk:
    """Up to sequence_length consecutive decisions, collated once per update."""
    batch: ObservationBatch
    labels: Tensor
    actions: Tensor
    behavior_log_probability: Tensor
    choice: Tensor
    deck_kinds: Tensor
    event_counts: Tensor  # per row, on the CPU: GRU packing needs host lengths


def _prepare_chunks(episode, model, length) -> list[_Chunk]:
    """Collation depends only on recorded data, so every epoch reuses it."""
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(episode["steps"]), length):
        steps = episode["steps"][start:start + length]
        observations = [step["observation"] for step in steps]
        # Already checked by validate_episode before chunking.
        batch = collate_observations(observations, model.config, validate=False)
        labels = torch.zeros(batch.deck_cards.shape, dtype=torch.long)
        for row, step in enumerate(steps):
            counts = validate_labels(step["labels"], step["observation"])
            labels[row, :len(counts)] = torch.tensor(counts)
        actions = [next(i for i, a in enumerate(step["observation"]["actions"]) if a["id"] == step["action_id"])
                   for step in steps]
        chunks.append(_Chunk(batch.to(device), labels.to(device),
            torch.tensor(actions, device=device),
            torch.tensor([float(step["log_probability"]) for step in steps], device=device),
            # A single legal action carries no policy gradient, entropy, or KL.
            torch.tensor([len(o["actions"]) > 1 for o in observations], device=device),
            batch.deck_mask.sum(1).clamp(min=1).to(device),
            batch.event_mask.sum(-1)))
    return chunks


def _join(chunks: list[_Chunk]) -> _Chunk:
    """Rows of several trajectories' chunks, episode-major, as one model batch."""
    if len(chunks) == 1:
        return chunks[0]
    return _Chunk(concat_batches([c.batch for c in chunks]), pad_cat([c.labels for c in chunks]),
                  torch.cat([c.actions for c in chunks]),
                  torch.cat([c.behavior_log_probability for c in chunks]),
                  torch.cat([c.choice for c in chunks]), torch.cat([c.deck_kinds for c in chunks]),
                  torch.cat([c.event_counts for c in chunks]))


def _windows(indices, prepared):
    """Window w holds chunk w of every listed trajectory that is that long.

    Trajectories never re-enter after ending, so each window's members are an
    order-preserving subset of the previous window's.
    """
    window = 0
    while True:
        members = [i for i in indices if window < len(prepared[i])]
        if not members:
            return
        yield window, members
        window += 1


def _carry(memory: PolicyMemory | None, previous: list[int], members: list[int]) -> PolicyMemory | None:
    """Keep the memory rows of trajectories that continue into this window."""
    if memory is None or previous == members:
        return memory
    keep = torch.tensor([previous.index(i) for i in members], device=memory.events.device)
    return PolicyMemory(memory.events[keep], memory.decisions[keep])


def ppo_update(model, critic, optimizer, episodes, config: PPOConfig, policy_version: str,
               *, validated=False):
    """Rebuild memory from step zero at current weights on every PPO pass.

    Trajectories are processed together: window w stacks chunk w (sequence_length
    decisions) of every trajectory in the minibatch into one forward pass, with
    memory detached between windows (truncated BPTT). An optimizer step occurs
    after whole episode minibatches, so forward memories never straddle a weight
    change, and policy probabilities are computed without dropout noise.

    Policy loss, entropy and KL average over decisions with alternatives only;
    forced single-action decisions still train the value heads and carry memory.
    Truncated trajectories bootstrap from the critic's value of their last state.
    """
    model.eval()  # eval changes dropout/attention behavior, not gradient tracking
    # cuDNN requires training-mode GRU reserve buffers for backward on CUDA.
    # These single-layer GRUs have zero dropout, so behavior probabilities stay
    # deterministic while the Transformer retains its inference configuration.
    model.event_memory.train()
    model.decision_memory.train()
    critic.train()
    if not validated:  # load_episodes already validated files it read
        for episode in episodes:
            validate_episode(episode, model.config, policy_version, config.max_episode_steps)
    length = config.sequence_length
    prepared = [_prepare_chunks(episode, model, length) for episode in episodes]
    targets = []
    with torch.no_grad():
        values = [[] for _ in episodes]
        everyone = list(range(len(episodes)))
        for group_start in range(0, len(everyone), config.target_batch_episodes):
            group = everyone[group_start:group_start + config.target_batch_episodes]
            memory, previous = None, group
            for window, members in _windows(group, prepared):
                memory = _carry(memory, previous, members)
                joined = _join([prepared[i][window] for i in members])
                sizes = [len(prepared[i][window].actions) for i in members]
                out = model.forward_episodes(joined.batch, sizes, memory, joined.event_counts)
                memory, previous = out.memory, members
                actual = out.distribution().log_prob(joined.actions)
                if (actual - joined.behavior_log_probability).abs().max().item() > config.log_probability_tolerance:
                    raise ValueError("behavior policy mismatch: weights, history, candidate ordering, or sampling changed")
                critic_values = critic(out.context, joined.batch.deck_cards, joined.labels.float())
                for i, value in zip(members, critic_values.split(sizes)):
                    values[i].append(value)
        for episode, chunks in zip(episodes, values):
            values_t = torch.cat(chunks)
            rewards = torch.tensor([s["reward"] for s in episode["steps"]], device=values_t.device,
                                   dtype=values_t.dtype)
            end = torch.zeros(len(values_t), device=values_t.device, dtype=torch.bool)
            end[-1] = True
            if episode.get("truncated", False):
                # No observation follows the last recorded decision; its own value
                # is the bootstrap estimate, so that step's advantage is zero.
                advantages, returns = generalized_advantages(rewards, values_t, torch.zeros_like(end), end,
                    bootstrap_value=values_t[-1], gamma=config.gamma, gae_lambda=config.gae_lambda)
            else:
                advantages, returns = generalized_advantages(rewards, values_t, end,
                    gamma=config.gamma, gae_lambda=config.gae_lambda)
            targets.append((advantages, returns))
        choices = torch.cat([chunk.choice for chunks in prepared for chunk in chunks])
        # Forced steps receive no policy gradient, so they must not shift the
        # normalization of the advantages that do.
        chosen_advantages = torch.cat([x[0] for x in targets])[choices]
        if chosen_advantages.numel():
            mean = chosen_advantages.mean()
            std = chosen_advantages.std(unbiased=False).clamp(min=1e-8)
        else:
            mean, std = chosen_advantages.new_zeros(()), chosen_advantages.new_ones(())
        targets = [((a - mean) / std, r) for a, r in targets]

    trains_belief = model.config.belief_mode == "learned" or model.config.auxiliary_belief
    belief_coefficient = config.belief_coefficient if trains_belief else 0.0
    names = ("policy_loss", "critic_loss", "public_value_loss", "belief_nll", "entropy", "approx_kl")
    metrics = {"decisions": 0, "choice_decisions": 0, "epochs": 0, "optimizer_steps": 0,
               "belief_mode": model.config.belief_mode, "belief_loss_coefficient": belief_coefficient,
               "auxiliary_belief": model.config.auxiliary_belief,
               "truncated_episodes": sum(bool(e.get("truncated", False)) for e in episodes)}
    total_choices = max(int(choices.sum().item()), 1)
    step_counts = [len(e["steps"]) for e in episodes]
    choice_counts = [sum(int(c.choice.sum().item()) for c in chunks) for chunks in prepared]
    device = choices.device
    # Accumulate on the device; one host sync per epoch instead of per chunk.
    sums = {name: torch.zeros((), device=device) for name in names}
    for epoch in range(config.epochs):
        epoch_kl = torch.zeros((), device=device)
        ordering = torch.randperm(len(episodes)).tolist()
        for start in range(0, len(ordering), config.episodes_per_batch):
            group = ordering[start:start + config.episodes_per_batch]
            step_denominator = sum(step_counts[i] for i in group)
            choice_denominator = max(sum(choice_counts[i] for i in group), 1)
            optimizer.zero_grad(set_to_none=True)
            memory, previous = None, group
            for window, members in _windows(group, prepared):
                memory = _carry(memory, previous, members)
                joined = _join([prepared[i][window] for i in members])
                sizes = [len(prepared[i][window].actions) for i in members]
                rows = slice(window * length, window * length + length)
                advantage = torch.cat([targets[i][0][rows] for i in members])
                returns = torch.cat([targets[i][1][rows] for i in members])
                out = model.forward_episodes(joined.batch, sizes, memory, joined.event_counts)
                distribution = out.distribution()
                log_ratio = distribution.log_prob(joined.actions) - joined.behavior_log_probability
                ratio = log_ratio.exp()
                choice = joined.choice.to(ratio.dtype)
                policy = -torch.minimum(ratio * advantage,
                    ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantage) * choice
                entropy = distribution.entropy() * choice
                value = critic(out.context, joined.batch.deck_cards, joined.labels.float())
                critic_loss = (value - returns).square()
                public_loss = (out.value - returns).square()
                belief = out.belief
                if out.auxiliary_count_logits is not None:
                    belief = constrained_hand_belief(out.auxiliary_count_logits, joined.batch.deck_lower,
                                                     joined.batch.deck_upper, joined.batch.hand_size)
                belief_loss = belief.nll(joined.labels) / joined.deck_kinds
                loss = ((policy.sum() - config.entropy_coefficient * entropy.sum()) / choice_denominator
                        + (config.value_coefficient * critic_loss.sum()
                           + config.public_value_coefficient * public_loss.sum()
                           + belief_coefficient * belief_loss.sum()) / step_denominator)
                loss.backward()
                memory, previous = out.memory.detach(), members
                kl = ((ratio - 1 - log_ratio) * choice).sum().detach()
                epoch_kl += kl
                metrics["decisions"] += len(joined.actions)
                for name, tensor in (("policy_loss", policy), ("critic_loss", critic_loss),
                        ("public_value_loss", public_loss), ("belief_nll", belief_loss),
                        ("entropy", entropy)):
                    sums[name] += tensor.detach().sum()
                sums["approx_kl"] += kl
            metrics["choice_decisions"] += sum(choice_counts[i] for i in group)
            parameters = list(model.parameters()) + list(critic.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            metrics["optimizer_steps"] += 1
        metrics["epochs"] = epoch + 1
        if epoch_kl.item() / total_choices > config.target_kl:
            break
    sums = {name: value.item() for name, value in sums.items()}
    for name in ("policy_loss", "entropy", "approx_kl"):
        metrics[name] = sums[name] / max(metrics["choice_decisions"], 1)
    for name in ("critic_loss", "public_value_loss", "belief_nll"):
        metrics[name] = sums[name] / max(metrics["decisions"], 1)
    metrics["episodes"] = len(episodes)
    return metrics


def train_from_files(checkpoint_path, episode_paths, output_path, config=PPOConfig(), device="auto"):
    model, critic, catalog, checkpoint = load_checkpoint(checkpoint_path, resolve_device(device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()),
                                  lr=config.learning_rate, weight_decay=0.0)
    if checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
    episodes = load_episodes(episode_paths, model.config, checkpoint["policy_version"], config.max_episode_steps)
    metrics = ppo_update(model, critic, optimizer, episodes, config, checkpoint["policy_version"],
                         validated=True)
    save_checkpoint(output_path, model, critic, catalog, optimizer=optimizer,
                    updates=checkpoint["updates"] + 1, metrics=metrics)
    return metrics
