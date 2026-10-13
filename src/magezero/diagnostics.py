"""Held-out prediction evaluation and a reproducible decision trace.

Evaluation replays each observer's complete history at the evaluated checkpoint.
It does not train on the evaluation games or treat their behavior policy as ours.
"""

import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .statistics import statistical_hand_belief
from .roles import ROLE_NAMES
from .checkpoint import load_checkpoint
from .schema import collate_observations, validate_labels
from .learner import generalized_advantages


def read_trajectories(paths):
    return [json.loads(line) for path in paths for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def deck_prior(batch, classes):
    """Without-replacement public count prior, conditioned on known hand cards."""
    return statistical_hand_belief(batch.deck_lower, batch.deck_upper, batch.hand_size, classes)


def evaluate_predictions(checkpoint, paths, device="cpu"):
    torch.set_num_threads(1)
    model, _, catalog, data = load_checkpoint(checkpoint, device)
    started = time.perf_counter()
    sums = dict(joint_nll=0., nll_per_card=0., expected_count_mae=0., presence_brier=0.,
                prior_joint_nll=0., prior_expected_count_mae=0., policy_entropy_choices=0.)
    states, choice_states = 0, 0
    role_brier = torch.zeros(len(ROLE_NAMES), dtype=torch.float64)
    role_targets = torch.zeros_like(role_brier)
    with torch.inference_mode():
        for episode in read_trajectories(paths):
            memory = None
            for step in episode["steps"]:
                obs = step["observation"]
                batch = collate_observations([obs], model.config).to(device)
                labels = torch.tensor([validate_labels(step["labels"], obs)], device=device)
                out = model(batch, memory); memory = out.memory
                prior = deck_prior(batch, model.config.max_hand_size + 1)
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
                    role_brier += ((out.role_presence - true_presence.float()).square()[0]).cpu().double()
                    role_targets += true_presence[0].cpu().double()
                if len(obs["actions"]) > 1:
                    sums["policy_entropy_choices"] += out.distribution().entropy().item()
                    choice_states += 1
                states += 1
    results = {name: total / max(choice_states if name == "policy_entropy_choices" else states, 1)
               for name, total in sums.items()}
    result = {**results, "states": states, "choice_states": choice_states,
        "episodes": len(read_trajectories(paths)), "seconds": time.perf_counter() - started,
        "policy_version": data["policy_version"], "checkpoint": str(Path(checkpoint).resolve()),
        "scope": "recorded observer trajectories; caller must ensure held-out separation; correlated states, not independent win-rate evidence",
        "belief_mode": model.config.belief_mode}
    if model.config.belief_mode == "statistical_roles":
        result["role_presence_brier_by_role"] = dict(zip(ROLE_NAMES, (role_brier / max(states, 1)).tolist()))
        result["role_observed_presence_by_role"] = dict(zip(ROLE_NAMES, (role_targets / max(states, 1)).tolist()))
    return result


def vector_info(tensor):
    value = tensor.detach().float().cpu()
    return {"shape": list(value.shape), "first_values": value.reshape(-1)[:8].tolist(),
        "norm": float(value.norm()), "minimum": float(value.min()), "maximum": float(value.max())}


def trace_decision(checkpoint, episode_path, output, step_index=None):
    torch.set_num_threads(1)
    model, critic, catalog, data = load_checkpoint(checkpoint, "cpu")
    episode = read_trajectories([episode_path])[0]
    if step_index is None:
        step_index = next((i for i, s in enumerate(episode["steps"])
            if len(s["observation"]["actions"]) >= 3 and len(s["observation"]["entities"]) >= 8), 0)
    stages, handles, raw = {}, [], {}
    for name in ("entity_numeric", "entity_norm", "entity_attention", "global_encoder", "decision_memory",
                 "belief_head", "role_encoder", "belief_encoder", "context_encoder", "action_numeric", "action_encoder",
                 "action_scorer", "value_head"):
        def hook(module, inputs, result, stage=name):
            stages[stage] = vector_info(result)
            if stage == "entity_norm": raw[stage] = result.detach()
        if hasattr(model, name):
            handles.append(getattr(model, name).register_forward_hook(hook))
    with torch.inference_mode():
        memory = None
        for i, step in enumerate(episode["steps"][:step_index + 1]):
            batch = collate_observations([step["observation"]], model.config)
            before_memory = memory
            out = model(batch, memory); memory = out.memory
        obs = step["observation"]
        names = {card["id"]: card["name"] for card in catalog.cards}
        labels = torch.tensor([validate_labels(step["labels"], obs)])
        distribution = out.distribution()
        # Inspect actual first-layer pooling-query attention, averaged over heads.
        layer = model.entity_attention.layers[0]
        tokens = torch.cat([model.pool_token, raw["entity_norm"]], 1)
        normalized = layer.norm1(tokens)
        q, k, v = F.linear(normalized, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).chunk(3, -1)
        def heads(x): return x.reshape(1, -1, model.config.attention_heads,
                                      model.config.width // model.config.attention_heads).transpose(1, 2)
        weights = (heads(q) @ heads(k).transpose(-1, -2) / math.sqrt(model.config.width // model.config.attention_heads)).softmax(-1)
        pooling_attention = weights[0, :, 0].mean(0).tolist()
        trace = {"checkpoint": str(Path(checkpoint).resolve()), "policy_version": data["policy_version"],
            "source_episode": str(Path(episode_path).resolve()), "step": step_index,
            "observation": obs, "card_names": names, "stages": stages,
            "first_layer_pool_attention": pooling_attention,
            "memory_before": {"events": vector_info(before_memory.events), "decisions": vector_info(before_memory.decisions)}
                if before_memory else None,
            "memory_after": {"events": vector_info(memory.events), "decisions": vector_info(memory.decisions)},
            "actions": [{**a, "name": names.get(a["card_id"], a["type"]),
                "source_index": int(batch.action_sources[0, j]), "target_index": int(batch.action_targets[0, j]),
                "logit": float(out.logits[0, j]), "probability": float(distribution.probs[0, j])}
                for j, a in enumerate(obs["actions"])],
            "belief": [{**c, "name": names[c["card_id"]], "expected_count": float(out.belief.expected_counts[0, j]),
                "presence_probability": float(1 - out.belief.probabilities[0, j, 0]),
                "count_probabilities": out.belief.probabilities[0, j, :obs["opponent_hand_size"] + 1].tolist(),
                "training_only_true_count": int(labels[0, j])} for j, c in enumerate(obs["opponent_deck"])],
            "public_value": float(out.value[0]), "training_only_privileged_value": float(critic(out.context, batch.deck_cards, labels.float())[0]),
            "joint_belief_nll": float(out.belief.nll(labels)[0]),
            "original_recorded_action": step["action_id"], "original_behavior_log_probability": step["log_probability"],
            "actual_terminal_reward": episode["steps"][-1]["reward"],
            "scope": "actual forward trace at this checkpoint on recorded history; it need not choose the original behavior action"}
    if out.role_presence is not None:
        trace["role_presence"] = dict(zip(ROLE_NAMES, out.role_presence[0].tolist()))
    for handle in handles: handle.remove()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return trace


def explain_training_sample(before_checkpoint, after_checkpoint, paths, episode_path, step_index, output):
    """Actual on-policy data and endpoint losses, not a reconstructed optimizer log."""
    torch.set_num_threads(1)
    model, critic, _, data = load_checkpoint(before_checkpoint, "cpu")
    advantages, returns, keys = [], [], []
    with torch.no_grad():
        for path in paths:
            episode = read_trajectories([path])[0]
            if episode["policy_version"] != data["policy_version"]: raise ValueError("sample is not on-policy")
            memory, values = None, []
            for step in episode["steps"]:
                batch = collate_observations([step["observation"]], model.config)
                labels = torch.tensor([validate_labels(step["labels"], step["observation"])])
                out = model(batch, memory); memory = out.memory
                values.append(critic(out.context, batch.deck_cards, labels.float())[0])
            values = torch.stack(values)
            rewards = torch.tensor([s["reward"] for s in episode["steps"]])
            terminal = torch.zeros(len(values), dtype=torch.bool); terminal[-1] = True
            a, r = generalized_advantages(rewards, values, terminal, gamma=1., gae_lambda=.97)
            advantages.append(a); returns.append(r); keys.append(str(Path(path).resolve()))
    all_a = torch.cat(advantages)
    index = keys.index(str(Path(episode_path).resolve()))
    advantage = float((advantages[index][step_index] - all_a.mean()) / all_a.std(unbiased=False).clamp(min=1e-8))
    target = float(returns[index][step_index])
    before = trace_decision(before_checkpoint, episode_path, str(Path(output).with_name("flow-onpolicy-before.json")), step_index)
    after = trace_decision(after_checkpoint, episode_path, str(Path(output).with_name("flow-onpolicy-after.json")), step_index)
    action = next(a for a in after["actions"] if a["id"] == after["original_recorded_action"])
    old_logp = before["original_behavior_log_probability"]
    ratio = action["probability"] / math.exp(old_logp)
    clipped = min(1.2, max(.8, ratio))
    entropy = -sum(a["probability"] * math.log(a["probability"]) for a in after["actions"])
    losses = {"ppo": -min(ratio * advantage, clipped * advantage),
        "privileged_value_mse": (after["training_only_privileged_value"] - target)**2,
        "public_value_mse": (after["public_value"] - target)**2,
        "belief_nll_per_card": after["joint_belief_nll"] / len(after["belief"]), "entropy": entropy}
    result = {"step": step_index, "source_episode": str(Path(episode_path).resolve()),
        "normalized_advantage": advantage, "unnormalized_advantage": float(advantages[index][step_index]),
        "gae_return_target": target, "actual_terminal_reward": before["actual_terminal_reward"],
        "old_selected_probability": math.exp(old_logp), "new_selected_probability": action["probability"],
        "probability_ratio": ratio, "clipped_ratio": clipped, "losses": losses,
        "weighted_loss": losses["ppo"]+.5*losses["privileged_value_mse"]+.25*losses["public_value_mse"]
            +(.1 if model.config.belief_mode == "learned" else 0.)*losses["belief_nll_per_card"]-.01*entropy,
        "scope": "single recorded training sample evaluated at before/after checkpoints; losses are diagnostic endpoint values, not the actual final minibatch loss"}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
