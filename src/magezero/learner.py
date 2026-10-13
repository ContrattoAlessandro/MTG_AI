"""On-policy recurrent PPO with a training-only privileged critic.

Episodes are trajectories of ONE player's successive decisions, not alternating
global game turns; the engine queues that player's rewards until its next
decision. Every update replays each trajectory from step zero at the current
weights, so recurrent memory is never stale.
"""

import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .belief import constrained_hand_belief
from .checkpoint import load_checkpoint, resolve_device, save_checkpoint
from .model import EntityBeliefPolicy, PolicyMemory, PrivilegedCritic
from .schema import (
    SCHEMA_VERSION,
    ObservationBatch,
    collate_observations,
    concat_batches,
    pad_cat,
    validate_labels,
    validate_observation,
)
from .shaping import shaped_rewards


@dataclass(frozen=True)
class PPOConfig:
    """The `training:` section of a policy YAML."""

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
    # Weight of the policy entropy divided by log(number of legal candidates), so
    # binary choices and 256-option amounts get comparable exploration pressure.
    entropy_coefficient: float = 0.01
    # Applied to the actor (policy + public value) and the privileged critic
    # separately, so a large critic gradient cannot shrink policy steps.
    max_grad_norm: float = 1.0
    # A minibatch whose approximate KL from the behaviour policy exceeds 1.5 x this
    # ends the update before its optimizer step (as Stable-Baselines3). 0 disables.
    target_kl: float = 0.02
    max_episode_steps: int = 10000
    log_probability_tolerance: float = 0.005
    # Trajectories per forward pass when computing PPO targets (no gradients).
    target_batch_episodes: int = 64
    # Potential-based shaping scale (policy-invariant; see shaping.py). 0 disables.
    shaping_coefficient: float = 0.0
    # KL(reference || policy) weight towards a frozen warm-start (behaviour-cloned)
    # policy, as AlphaStar regularizes towards its supervised policy. Multiplied by
    # reference_kl_decay once per PPO update since the reference. 0 disables.
    reference_kl_coefficient: float = 0.0
    reference_kl_decay: float = 1.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        for key in ("epochs", "episodes_per_batch", "sequence_length", "max_episode_steps", "target_batch_episodes"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid return/GAE discounts")
        if not 0 < self.reference_kl_decay <= 1:
            raise ValueError("reference_kl_decay must be in (0, 1]")
        positive = ("learning_rate", "epochs", "episodes_per_batch", "sequence_length", "clip_ratio",
                    "max_grad_norm", "max_episode_steps", "target_batch_episodes")  # fmt: skip
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError("PPO sizes, optimizer parameters, and clip ratio must be positive")
        non_negative = ("value_coefficient", "public_value_coefficient", "belief_coefficient", "entropy_coefficient",
                        "target_kl", "log_probability_tolerance", "shaping_coefficient",
                        "reference_kl_coefficient")  # fmt: skip
        if any(getattr(self, key) < 0 for key in non_negative):
            raise ValueError("loss coefficients and tolerances cannot be negative")


def generalized_advantages(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor | None = None,
    *,
    bootstrap_value=0.0,
    gamma: float = 1.0,
    gae_lambda: float = 0.97,
) -> tuple[Tensor, Tensor]:
    """GAE advantages and returns for one trajectory.

    Time limits bootstrap from `bootstrap_value`, but the recursion stops at both
    kinds of episode boundary.
    """
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


def validate_episode(episode: dict, config, policy_version: str, max_steps: int = 10000) -> dict:
    """Check a completed game, or an error-free game stopped by the turn limit; return it.

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


def load_episodes(paths, model_config, policy_version: str, max_steps: int = 10000) -> list[dict]:
    """Read and validate on-policy trajectories (one JSON object per line); reject duplicates."""
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


# ─── Trajectory chunks (shared with behaviour cloning) ───────────────────────


@dataclass
class Chunk:
    """Up to sequence_length consecutive decisions of one trajectory, collated once per update."""

    batch: ObservationBatch
    labels: Tensor
    actions: Tensor
    behavior_log_probability: Tensor
    choice: Tensor  # decisions with more than one legal candidate
    deck_kinds: Tensor
    event_counts: Tensor  # per row, on the CPU: GRU packing needs host lengths
    # The frozen reference policy's log-probabilities over the candidates (KL term).
    reference_log_probability: Tensor | None = None


def prepare_chunks(episode: dict, model: EntityBeliefPolicy, length: int) -> list[Chunk]:
    """Collate a validated trajectory into chunks of `length` decisions.

    Collation depends only on recorded data, so every epoch reuses it.
    """
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(episode["steps"]), length):
        steps = episode["steps"][start : start + length]
        observations = [step["observation"] for step in steps]
        # Already checked by validate_episode before chunking.
        batch = collate_observations(observations, model.config, validate=False)
        labels = torch.zeros(batch.deck_cards.shape, dtype=torch.long)
        for row, step in enumerate(steps):
            counts = validate_labels(step["labels"], step["observation"])
            labels[row, : len(counts)] = torch.tensor(counts)
        actions = [
            next(i for i, a in enumerate(step["observation"]["actions"]) if a["id"] == step["action_id"])
            for step in steps
        ]
        chunks.append(
            Chunk(
                batch.to(device),
                labels.to(device),
                torch.tensor(actions, device=device),
                torch.tensor([float(step["log_probability"]) for step in steps], device=device),
                # A single legal action carries no policy gradient, entropy, or KL.
                torch.tensor([len(o["actions"]) > 1 for o in observations], device=device),
                batch.deck_mask.sum(1).clamp(min=1).to(device),
                batch.event_mask.sum(-1),
            )
        )
    return chunks


def join_chunks(chunks: list[Chunk]) -> Chunk:
    """Rows of several trajectories' chunks, episode-major, as one model batch."""
    if len(chunks) == 1:
        return chunks[0]
    reference = None
    if chunks[0].reference_log_probability is not None:
        reference = pad_cat([c.reference_log_probability for c in chunks], float("-inf"))
    return Chunk(
        concat_batches([c.batch for c in chunks]),
        pad_cat([c.labels for c in chunks]),
        torch.cat([c.actions for c in chunks]),
        torch.cat([c.behavior_log_probability for c in chunks]),
        torch.cat([c.choice for c in chunks]),
        torch.cat([c.deck_kinds for c in chunks]),
        torch.cat([c.event_counts for c in chunks]),
        reference,
    )


def chunk_windows(indices: list[int], prepared: list[list[Chunk]]) -> Iterator[tuple[int, list[int]]]:
    """Yield (w, members): window w holds chunk w of every listed trajectory that is that long.

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


def carry_memory(memory: PolicyMemory | None, previous: list[int], members: list[int]) -> PolicyMemory | None:
    """Keep the memory rows of trajectories that continue into this window."""
    if memory is None or previous == members:
        return memory
    keep = torch.tensor([previous.index(i) for i in members], device=memory.events.device)
    return PolicyMemory(memory.events[keep], memory.decisions[keep])


def set_learning_modes(model: EntityBeliefPolicy, critic: PrivilegedCritic) -> None:
    """Deterministic attention, but training-mode GRUs and critic.

    eval() changes dropout/attention behaviour, not gradient tracking. cuDNN
    requires training-mode GRU reserve buffers for backward on CUDA; these
    single-layer GRUs have zero dropout, so behaviour probabilities stay
    deterministic while the Transformer keeps its inference configuration.
    """
    model.eval()
    model.event_memory.train()
    model.decision_memory.train()
    critic.train()


def clip_gradients(model: EntityBeliefPolicy, critic: PrivilegedCritic, max_norm: float) -> None:
    """Clip the actor (with its public value head) and the privileged critic separately."""
    for parameters in (model.parameters(), critic.parameters()):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)


# ─── PPO ──────────────────────────────────────────────────────────────────────


def reference_kl(reference_log_probability: Tensor, logits: Tensor, mask: Tensor) -> Tensor:
    """KL(reference || policy) per row, summed over the legal candidates only."""
    log_policy = torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)
    reference = reference_log_probability.masked_fill(~mask, 0.0)
    return (reference.exp() * (reference - log_policy)).masked_fill(~mask, 0.0).sum(-1)


def _ppo_targets(model, critic, episodes, prepared, config: PPOConfig, reference) -> list[tuple[Tensor, Tensor]]:
    """Normalized advantages and returns per trajectory, at the current weights.

    Also checks that the replayed policy reproduces the behaviour log-probabilities,
    and stores the reference policy's candidate log-probabilities in the chunks.
    """
    targets = []
    with torch.no_grad():
        values = [[] for _ in episodes]
        everyone = list(range(len(episodes)))
        for group_start in range(0, len(everyone), config.target_batch_episodes):
            group = everyone[group_start : group_start + config.target_batch_episodes]
            memory, reference_memory, previous = None, None, group
            for window, members in chunk_windows(group, prepared):
                memory = carry_memory(memory, previous, members)
                reference_memory = carry_memory(reference_memory, previous, members)
                joined = join_chunks([prepared[i][window] for i in members])
                sizes = [len(prepared[i][window].actions) for i in members]
                out = model.forward_episodes(joined.batch, sizes, memory, joined.event_counts)
                memory, previous = out.memory, members
                if reference is not None:
                    anchor = reference.forward_episodes(joined.batch, sizes, reference_memory, joined.event_counts)
                    reference_memory = anchor.memory
                    for i, rows in zip(members, torch.log_softmax(anchor.logits, -1).split(sizes)):
                        chunk = prepared[i][window]
                        chunk.reference_log_probability = rows[:, : chunk.batch.action_mask.shape[1]]
                actual = out.distribution().log_prob(joined.actions)
                if (actual - joined.behavior_log_probability).abs().max().item() > config.log_probability_tolerance:
                    raise ValueError(
                        "behavior policy mismatch: weights, history, candidate ordering, or sampling changed"
                    )
                critic_values = critic(out.context, joined.batch.deck_cards, joined.labels.float())
                for i, value in zip(members, critic_values.split(sizes)):
                    values[i].append(value)
        for episode, chunks in zip(episodes, values):
            values_t = torch.cat(chunks)
            rewards = shaped_rewards(episode, config.shaping_coefficient, config.gamma).to(
                device=values_t.device, dtype=values_t.dtype
            )
            end = torch.zeros(len(values_t), device=values_t.device, dtype=torch.bool)
            end[-1] = True
            if episode.get("truncated", False):
                # No observation follows the last recorded decision; its own value
                # is the bootstrap estimate, so that step's advantage is zero.
                advantages, returns = generalized_advantages(
                    rewards, values_t, torch.zeros_like(end), end,
                    bootstrap_value=values_t[-1], gamma=config.gamma, gae_lambda=config.gae_lambda,
                )  # fmt: skip
            else:
                advantages, returns = generalized_advantages(
                    rewards, values_t, end, gamma=config.gamma, gae_lambda=config.gae_lambda
                )
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
        return [((a - mean) / std, r) for a, r in targets]


def ppo_update(
    model: EntityBeliefPolicy,
    critic: PrivilegedCritic,
    optimizer: torch.optim.Optimizer,
    episodes: list[dict],
    config: PPOConfig,
    policy_version: str,
    *,
    validated: bool = False,
    reference: EntityBeliefPolicy | None = None,
    reference_coefficient: float | None = None,
) -> dict:
    """One PPO update on complete trajectories; returns averaged loss statistics.

    Memory is rebuilt from step zero at the current weights on every pass.
    Trajectories are processed together: window w stacks chunk w (sequence_length
    decisions) of every trajectory in the minibatch into one forward pass, with
    memory detached between windows (truncated BPTT). An optimizer step occurs
    after whole episode minibatches, so forward memories never straddle a weight
    change, and policy probabilities are computed without dropout noise.

    Policy loss, entropy and KL average over decisions with alternatives only;
    forced single-action decisions still train the value heads and carry memory.
    Truncated trajectories bootstrap from the critic's value of their last state.

    reference: a frozen policy (the behaviour-cloned warm start). Its candidate
    distribution is computed once, with its own memory, in the target pass, and the
    loss adds reference_coefficient (default config.reference_kl_coefficient) times
    KL(reference || policy), averaged over decisions with alternatives.
    """
    set_learning_modes(model, critic)
    if not validated:  # load_episodes already validated files it read
        for episode in episodes:
            validate_episode(episode, model.config, policy_version, config.max_episode_steps)
    length = config.sequence_length
    beta = config.reference_kl_coefficient if reference_coefficient is None else reference_coefficient
    if reference is None:
        beta = 0.0
    elif reference.config.catalog_digest != model.config.catalog_digest:
        raise ValueError("the reference policy uses a different card catalog")
    prepared = [prepare_chunks(episode, model, length) for episode in episodes]
    targets = _ppo_targets(model, critic, episodes, prepared, config, reference if beta > 0 else None)

    trains_belief = model.config.belief_mode == "learned" or model.config.auxiliary_belief
    belief_coefficient = config.belief_coefficient if trains_belief else 0.0
    names = ("policy_loss", "critic_loss", "public_value_loss", "belief_nll", "entropy", "normalized_entropy",
             "approx_kl", "reference_kl")  # fmt: skip
    metrics = {
        "decisions": 0,
        "choice_decisions": 0,
        "epochs": 0,
        "optimizer_steps": 0,
        "early_stopped": False,
        "belief_mode": model.config.belief_mode,
        "belief_loss_coefficient": belief_coefficient,
        "auxiliary_belief": model.config.auxiliary_belief,
        "truncated_episodes": sum(bool(e.get("truncated", False)) for e in episodes),
        "gae_lambda": config.gae_lambda,
        "shaping_coefficient": config.shaping_coefficient,
        "reference_kl_coefficient": beta,
    }
    step_counts = [len(e["steps"]) for e in episodes]
    choice_counts = [sum(int(c.choice.sum().item()) for c in chunks) for chunks in prepared]
    device = prepared[0][0].choice.device
    # Accumulate on the device; one host sync per minibatch (the KL check).
    sums = {name: torch.zeros((), device=device) for name in names}
    for epoch in range(config.epochs):
        ordering = torch.randperm(len(episodes)).tolist()
        for start in range(0, len(ordering), config.episodes_per_batch):
            group = ordering[start : start + config.episodes_per_batch]
            step_denominator = sum(step_counts[i] for i in group)
            choice_denominator = max(sum(choice_counts[i] for i in group), 1)
            optimizer.zero_grad(set_to_none=True)
            group_kl = torch.zeros((), device=device)
            memory, previous = None, group
            for window, members in chunk_windows(group, prepared):
                memory = carry_memory(memory, previous, members)
                joined = join_chunks([prepared[i][window] for i in members])
                sizes = [len(prepared[i][window].actions) for i in members]
                rows = slice(window * length, window * length + length)
                advantage = torch.cat([targets[i][0][rows] for i in members])
                returns = torch.cat([targets[i][1][rows] for i in members])
                out = model.forward_episodes(joined.batch, sizes, memory, joined.event_counts)
                distribution = out.distribution()
                log_ratio = distribution.log_prob(joined.actions) - joined.behavior_log_probability
                ratio = log_ratio.exp()
                choice = joined.choice.to(ratio.dtype)
                clipped = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
                policy = -torch.minimum(ratio * advantage, clipped * advantage) * choice
                entropy = distribution.entropy() * choice
                legal = joined.batch.action_mask.sum(-1).clamp(min=2).to(entropy.dtype)
                normalized_entropy = entropy / legal.log()
                value = critic(out.context, joined.batch.deck_cards, joined.labels.float())
                critic_loss = (value - returns).square()
                public_loss = (out.value - returns).square()
                belief = out.belief
                if out.auxiliary_count_logits is not None:
                    belief = constrained_hand_belief(
                        out.auxiliary_count_logits, joined.batch.deck_lower, joined.batch.deck_upper,
                        joined.batch.hand_size,
                    )  # fmt: skip
                belief_loss = belief.nll(joined.labels) / joined.deck_kinds
                if beta > 0:
                    anchor = reference_kl(joined.reference_log_probability, out.logits, joined.batch.action_mask)
                    anchor = anchor * choice
                else:
                    anchor = torch.zeros_like(choice)
                loss = (
                    policy.sum() - config.entropy_coefficient * normalized_entropy.sum() + beta * anchor.sum()
                ) / choice_denominator + (
                    config.value_coefficient * critic_loss.sum()
                    + config.public_value_coefficient * public_loss.sum()
                    + belief_coefficient * belief_loss.sum()
                ) / step_denominator
                loss.backward()
                memory, previous = out.memory.detach(), members
                kl = ((ratio - 1 - log_ratio) * choice).sum().detach()
                group_kl += kl
                metrics["decisions"] += len(joined.actions)
                for name, tensor in (
                    ("policy_loss", policy),
                    ("critic_loss", critic_loss),
                    ("public_value_loss", public_loss),
                    ("belief_nll", belief_loss),
                    ("entropy", entropy),
                    ("normalized_entropy", normalized_entropy),
                    ("reference_kl", anchor),
                ):
                    sums[name] += tensor.detach().sum()
                sums["approx_kl"] += kl
            metrics["choice_decisions"] += sum(choice_counts[i] for i in group)
            metrics["epochs"] = epoch + 1
            # These forward passes used the weights before this step: the KL is what
            # the previous steps produced. Stop before moving further away.
            if config.target_kl > 0 and group_kl.item() / choice_denominator > 1.5 * config.target_kl:
                metrics["early_stopped"] = True
                break
            clip_gradients(model, critic, config.max_grad_norm)
            optimizer.step()
            metrics["optimizer_steps"] += 1
        if metrics["early_stopped"]:
            break
    sums = {name: value.item() for name, value in sums.items()}
    for name in ("policy_loss", "entropy", "normalized_entropy", "approx_kl", "reference_kl"):
        metrics[name] = sums[name] / max(metrics["choice_decisions"], 1)
    for name in ("critic_loss", "public_value_loss", "belief_nll"):
        metrics[name] = sums[name] / max(metrics["decisions"], 1)
    metrics["episodes"] = len(episodes)
    return metrics


def check_warm_start(checkpoint: dict, config: PPOConfig) -> None:
    """Reject a cloned checkpoint whose value heads estimate different returns.

    A cloned checkpoint's value heads estimate returns under the clone's shaping
    and gamma; when PPO bootstraps from them (gae_lambda < 1) those must match.
    """
    clone = checkpoint.get("metrics", {}).get("behavior_cloning")
    if clone is None or config.gae_lambda == 1:
        return
    for key in ("shaping_coefficient", "gamma"):
        if clone["config"][key] != getattr(config, key):
            raise ValueError(
                f"the cloned critic was trained with {key} {clone['config'][key]}, but PPO "
                f"uses {getattr(config, key)} with gae_lambda < 1: re-clone or match the configs"
            )


def reference_coefficient(config: PPOConfig, updates: int, reference_updates: int) -> float:
    """The KL weight after `updates - reference_updates` PPO updates of decay."""
    return config.reference_kl_coefficient * config.reference_kl_decay ** max(0, updates - reference_updates)


def train_from_files(
    checkpoint_path, episode_paths, output_path, config: PPOConfig | None = None, device: str = "auto", reference=None
) -> dict:
    """Load a checkpoint, run one PPO update on its trajectories, and save the result.

    reference: optional frozen warm-start checkpoint for the KL term.
    """
    config = config or PPOConfig()
    model, critic, catalog, checkpoint = load_checkpoint(checkpoint_path, resolve_device(device))
    check_warm_start(checkpoint, config)
    anchor, beta = None, 0.0
    if reference is not None and config.reference_kl_coefficient > 0:
        anchor, _, reference_catalog, reference_data = load_checkpoint(reference, resolve_device(device))
        if reference_catalog.digest != catalog.digest:
            raise ValueError("the reference checkpoint uses a different card catalog")
        beta = reference_coefficient(config, checkpoint["updates"], reference_data["updates"])
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(critic.parameters()), lr=config.learning_rate, weight_decay=0.0
    )
    if checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
    episodes = load_episodes(episode_paths, model.config, checkpoint["policy_version"], config.max_episode_steps)
    metrics = ppo_update(
        model, critic, optimizer, episodes, config, checkpoint["policy_version"],
        validated=True, reference=anchor, reference_coefficient=beta,
    )  # fmt: skip
    save_checkpoint(
        output_path, model, critic, catalog, optimizer=optimizer, updates=checkpoint["updates"] + 1, metrics=metrics
    )
    return metrics
