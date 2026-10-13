"""Format-4 architecture and learner changes from the 13 October architecture review."""

import copy

import pytest
import torch

from magezero.learner import (PPOConfig, carry_memory, join_chunks, prepare_chunks, chunk_windows, check_warm_start,
                                     clip_gradients, ppo_update)
from magezero.schema import EVENT_TYPES, collate_observations
from magezero.service import PolicyService
from test_entity_policy import components, observation  # noqa: F401  (fixture)
from test_long_run import _episode, _varied_trajectory

torch.set_num_threads(1)
LABELS = {"opponent_hand_counts": [1, 1, 1]}


def _events(step, count):
    kinds = [EVENT_TYPES[i % len(EVENT_TYPES)] for i in range(count)]
    return [{"seq": step * 1000 + i, "type": kind, "actor": i % 3,
             "card_id": 0 if kind == "draw" else i % 4, "features": [i / 50] + [0.0] * 7}
            for i, kind in enumerate(kinds)]


def _bursty_trajectory(config, counts, game_id):
    """Decisions with 0, 1 or many new events, as Kinnan mana loops produce."""
    steps = [observation(config, t, game_id=game_id) for t in range(len(counts))]
    for t, (obs, count) in enumerate(zip(steps, counts)):
        obs["events"] = _events(t, count)
    return steps


def test_event_pooling_is_identical_stepwise_sequenced_and_batched(components):
    model, _, _ = components
    trajectories = [_bursty_trajectory(model.config, [3, 0, 40, 1, 0, 7, 200], "a"),
                    _bursty_trajectory(model.config, [0, 0, 5], "b"),
                    _bursty_trajectory(model.config, [1, 300, 2, 0], "c")]
    with torch.no_grad():
        stepwise = []
        for steps in trajectories:
            memory, rows = None, []
            for obs in steps:
                out = model(collate_observations([obs], model.config), memory)
                memory = out.memory
                rows.append(out.logits[0])
            stepwise.append((torch.stack(rows), memory))
        for (logits, memory), steps in zip(stepwise, trajectories):
            out = model.forward_sequence(collate_observations(steps, model.config))
            torch.testing.assert_close(out.logits[:, :logits.shape[1]], logits, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(out.memory.events, memory.events, atol=1e-5, rtol=1e-5)
        episodes = [{"steps": [{"observation": o, "action_id": o["actions"][0]["id"], "log_probability": 0.0,
                                "labels": LABELS} for o in t]} for t in trajectories]
        prepared = [prepare_chunks(e, model, 3) for e in episodes]
        memory, previous, rows = None, [0, 1, 2], {i: [] for i in range(3)}
        for window, members in chunk_windows([0, 1, 2], prepared):
            memory = carry_memory(memory, previous, members)
            joined = join_chunks([prepared[i][window] for i in members])
            sizes = [len(prepared[i][window].actions) for i in members]
            out = model.forward_episodes(joined.batch, sizes, memory, joined.event_counts)
            for i, logits in zip(members, out.logits.split(sizes)):
                rows[i].append(logits)
            memory, previous = out.memory, members
    for i, (logits, _) in enumerate(stepwise):
        batched = torch.cat([l[:, :logits.shape[1]] for l in rows[i]])
        torch.testing.assert_close(batched, logits, atol=1e-5, rtol=1e-5)


def test_event_pooling_ignores_padding_and_sees_the_events(components):
    model, _, _ = components
    alone = observation(model.config)
    alone["events"] = _events(0, 4)
    crowded = copy.deepcopy(alone)
    crowded["events"] = _events(0, 500)
    changed = copy.deepcopy(alone)
    changed["events"][-1]["type"] = "damage"
    with torch.no_grad():
        single = model._encode(collate_observations([alone], model.config)).events
        padded = model._encode(collate_observations([alone, crowded], model.config)).events
        other = model._encode(collate_observations([changed], model.config)).events
    torch.testing.assert_close(padded[0], single[0], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(other, single)
    assert not torch.allclose(padded[1], single[0])


def test_initial_policy_is_near_uniform(components):
    model, _, _ = components
    with torch.no_grad():
        probs = model(collate_observations([observation(model.config)], model.config)).distribution().probs
    torch.testing.assert_close(probs[0], torch.full((2,), 0.5), atol=0.02, rtol=0)


def test_actor_and_critic_gradients_are_clipped_separately(components):
    model, critic, _ = components
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1e-4)
    for parameter in critic.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)
    actor_before = [p.grad.clone() for p in model.parameters()]
    clip_gradients(model, critic, 1.0)
    for before, parameter in zip(actor_before, model.parameters()):
        torch.testing.assert_close(parameter.grad, before)  # a large critic gradient no longer shrinks it
    critic_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in critic.parameters()]))
    assert critic_norm.item() == pytest.approx(1.0, rel=1e-4)


def test_kl_early_stop_skips_the_step_and_entropy_is_normalized(components, tmp_path):
    model, critic, catalog = components
    service = PolicyService(model, catalog, "v")
    episodes = [_episode(service, _varied_trajectory(model.config, n, f"g{n}"), "v", reward=r)
                for n, r in ((6, 1), (3, -1), (8, 1), (5, -1))]
    def update(config):
        m, c = copy.deepcopy(model), copy.deepcopy(critic)
        optimizer = torch.optim.AdamW(list(m.parameters()) + list(c.parameters()), lr=0.05)
        return ppo_update(m, c, optimizer, episodes, config, "v")
    metrics = update(PPOConfig(epochs=8, episodes_per_batch=1, sequence_length=3, target_kl=1e-6))
    assert metrics["early_stopped"] and 1 <= metrics["optimizer_steps"] < 8 * len(episodes)
    assert 0 < metrics["normalized_entropy"] <= 1 + 1e-6
    disabled = update(PPOConfig(epochs=2, episodes_per_batch=2, sequence_length=3, target_kl=0.0))
    assert not disabled["early_stopped"] and disabled["optimizer_steps"] == 4


def test_warm_start_shaping_must_match_when_bootstrapping():
    cloned = {"metrics": {"behavior_cloning": {"config": {"shaping_coefficient": 0.5, "gamma": 1.0}}}}
    check_warm_start(cloned, PPOConfig(gae_lambda=0.95, shaping_coefficient=0.5))
    check_warm_start(cloned, PPOConfig(gae_lambda=1.0, shaping_coefficient=0.0))  # baseline only: allowed
    check_warm_start({"metrics": {}}, PPOConfig(gae_lambda=0.95))  # not a clone
    with pytest.raises(ValueError, match="shaping_coefficient"):
        check_warm_start(cloned, PPOConfig(gae_lambda=0.95, shaping_coefficient=0.0))
    with pytest.raises(ValueError, match="gamma"):
        check_warm_start(cloned, PPOConfig(gae_lambda=0.95, shaping_coefficient=0.5, gamma=0.99))
