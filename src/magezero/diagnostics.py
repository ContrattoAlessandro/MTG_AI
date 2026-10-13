"""Held-out hand-prediction evaluation and a reproducible single-decision trace.

Both replay each observer's complete history from step zero at the evaluated
checkpoint. They do not train on the evaluation games or treat their behaviour
policy as the evaluated one.
"""

import json
import math
import time
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .roles import ROLE_NAMES
from .schema import collate_observations, validate_labels
from .statistics import statistical_hand_belief

# Modules whose outputs trace_decision records (those absent in a belief mode are skipped).
TRACED_STAGES = ("entity_numeric", "entity_norm", "entity_attention", "global_encoder", "decision_memory",
                 "belief_head", "role_encoder", "belief_encoder", "context_encoder", "action_numeric",
                 "action_encoder", "action_scorer", "value_head")  # fmt: skip


def read_trajectories(paths) -> list[dict]:
    """Every trajectory (one JSON object per nonempty line) in the given files."""
    return [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_predictions(checkpoint, paths, device: str = "cpu") -> dict:
    """Opponent-hand prediction quality of a checkpoint against the statistical prior.

    Reports joint and per-card NLL, expected-count error, presence Brier score
    and, for statistical_roles models, role-presence Brier scores, averaged over
    every recorded state; plus the policy entropy over decisions with a choice.
    """
    torch.set_num_threads(1)
    model, _, _, data = load_checkpoint(checkpoint, device)
    started = time.perf_counter()
    sums = dict.fromkeys(("joint_nll", "nll_per_card", "expected_count_mae", "presence_brier", "prior_joint_nll",
                          "prior_expected_count_mae", "policy_entropy_choices"), 0.0)  # fmt: skip
    states, choice_states = 0, 0
    role_brier = torch.zeros(len(ROLE_NAMES), dtype=torch.float64)
    role_targets = torch.zeros_like(role_brier)
    episodes = read_trajectories(paths)
    with torch.inference_mode():
        for episode in episodes:
            memory = None
            for step in episode["steps"]:
                obs = step["observation"]
                batch = collate_observations([obs], model.config).to(device)
                labels = torch.tensor([validate_labels(step["labels"], obs)], device=device)
                out = model(batch, memory)
                memory = out.memory
                prior = statistical_hand_belief(
                    batch.deck_lower, batch.deck_upper, batch.hand_size, model.config.max_hand_size + 1
                )
                joint = out.belief.nll(labels).item()
                sums["joint_nll"] += joint
                sums["nll_per_card"] += joint / len(obs["opponent_deck"])
                sums["expected_count_mae"] += (out.belief.expected_counts - labels).abs().mean().item()
                presence = 1 - out.belief.probabilities[..., 0]
                sums["presence_brier"] += (presence - (labels > 0).float()).square().mean().item()
                sums["prior_joint_nll"] += prior.nll(labels).item()
                sums["prior_expected_count_mae"] += (prior.expected_counts - labels).abs().mean().item()
                if out.role_presence is not None:
                    true_presence = (labels.unsqueeze(-1) * model.card_roles[batch.deck_cards]).sum(1) > 0
                    role_brier += (out.role_presence - true_presence.float()).square()[0].cpu().double()
                    role_targets += true_presence[0].cpu().double()
                if len(obs["actions"]) > 1:
                    sums["policy_entropy_choices"] += out.distribution().entropy().item()
                    choice_states += 1
                states += 1
    result = {name: total / max(choice_states if name == "policy_entropy_choices" else states, 1)
              for name, total in sums.items()}  # fmt: skip
    result.update(
        states=states,
        choice_states=choice_states,
        episodes=len(episodes),
        seconds=time.perf_counter() - started,
        policy_version=data["policy_version"],
        checkpoint=str(Path(checkpoint).resolve()),
        scope="recorded observer trajectories; caller must ensure held-out separation; "
        "correlated states, not independent win-rate evidence",
        belief_mode=model.config.belief_mode,
    )
    if model.config.belief_mode == "statistical_roles":
        result["role_presence_brier_by_role"] = dict(zip(ROLE_NAMES, (role_brier / max(states, 1)).tolist()))
        result["role_observed_presence_by_role"] = dict(zip(ROLE_NAMES, (role_targets / max(states, 1)).tolist()))
    return result


def _vector_info(tensor) -> dict:
    if isinstance(tensor, tuple):  # recurrent modules return (outputs, final hidden state)
        tensor = tensor[0]
    value = tensor.detach().float().cpu()
    return {"shape": list(value.shape), "first_values": value.reshape(-1)[:8].tolist(), "norm": float(value.norm()),
            "minimum": float(value.min()), "maximum": float(value.max())}  # fmt: skip


def _pool_attention(model, entities) -> list[float]:
    """First-layer attention of the pooling token over the entities, averaged over heads."""
    layer = model.entity_attention.layers[0]
    heads, width = model.config.attention_heads, model.config.width
    normalized = layer.norm1(torch.cat([model.pool_token, entities], 1))
    q, k, _ = torch.nn.functional.linear(
        normalized, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias
    ).chunk(3, -1)

    def split(x):
        return x.reshape(1, -1, heads, width // heads).transpose(1, 2)

    weights = (split(q) @ split(k).transpose(-1, -2) / math.sqrt(width // heads)).softmax(-1)
    return weights[0, :, 0].mean(0).tolist()


def trace_decision(checkpoint, episode_path, output, step_index: int | None = None) -> dict:
    """Write a JSON trace of every model stage for one recorded decision.

    The default decision is the first with at least three candidates and eight
    entities. The trace shows the checkpoint's own outputs on the recorded
    history; it need not choose the originally recorded action.
    """
    torch.set_num_threads(1)
    model, critic, catalog, data = load_checkpoint(checkpoint, "cpu")
    episode = read_trajectories([episode_path])[0]
    if step_index is None:
        step_index = next(
            (i for i, s in enumerate(episode["steps"])
             if len(s["observation"]["actions"]) >= 3 and len(s["observation"]["entities"]) >= 8),
            0,
        )  # fmt: skip
    stages, raw, handles = {}, {}, []
    for name in TRACED_STAGES:
        if not hasattr(model, name):
            continue

        def hook(module, inputs, result, stage=name):
            stages[stage] = _vector_info(result)
            if stage == "entity_norm":
                raw[stage] = result.detach()

        handles.append(getattr(model, name).register_forward_hook(hook))
    try:
        with torch.inference_mode():
            memory = before = None
            for step in episode["steps"][: step_index + 1]:
                batch = collate_observations([step["observation"]], model.config)
                before = memory
                out = model(batch, memory)
                memory = out.memory
            obs = step["observation"]
            names = {card["id"]: card["name"] for card in catalog.cards}
            labels = torch.tensor([validate_labels(step["labels"], obs)])
            distribution = out.distribution()
            trace = {
                "checkpoint": str(Path(checkpoint).resolve()),
                "policy_version": data["policy_version"],
                "source_episode": str(Path(episode_path).resolve()),
                "step": step_index,
                "observation": obs,
                "card_names": names,
                "stages": stages,
                "first_layer_pool_attention": _pool_attention(model, raw["entity_norm"]),
                "memory_before": None if before is None else {
                    "events": _vector_info(before.events), "decisions": _vector_info(before.decisions)},
                "memory_after": {"events": _vector_info(memory.events), "decisions": _vector_info(memory.decisions)},
                "actions": [
                    {**action, "name": names.get(action["card_id"], action["type"]),
                     "source_index": int(batch.action_sources[0, j]), "target_index": int(batch.action_targets[0, j]),
                     "logit": float(out.logits[0, j]), "probability": float(distribution.probs[0, j])}
                    for j, action in enumerate(obs["actions"])
                ],
                "belief": [
                    {**card, "name": names[card["card_id"]],
                     "expected_count": float(out.belief.expected_counts[0, j]),
                     "presence_probability": float(1 - out.belief.probabilities[0, j, 0]),
                     "count_probabilities": out.belief.probabilities[0, j, : obs["opponent_hand_size"] + 1].tolist(),
                     "training_only_true_count": int(labels[0, j])}
                    for j, card in enumerate(obs["opponent_deck"])
                ],
                "public_value": float(out.value[0]),
                "training_only_privileged_value": float(critic(out.context, batch.deck_cards, labels.float())[0]),
                "joint_belief_nll": float(out.belief.nll(labels)[0]),
                "original_recorded_action": step["action_id"],
                "original_behavior_log_probability": step["log_probability"],
                "actual_terminal_reward": episode["steps"][-1]["reward"],
                "scope": "actual forward trace at this checkpoint on recorded history; "
                "it need not choose the original behavior action",
            }  # fmt: skip
            if out.role_presence is not None:
                trace["role_presence"] = dict(zip(ROLE_NAMES, out.role_presence[0].tolist()))
    finally:
        for handle in handles:
            handle.remove()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return trace
