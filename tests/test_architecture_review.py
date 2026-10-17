"""Format-4 architecture and learner changes from the 13 October architecture review."""

import copy

import pytest
import torch

from magezero.learner import (
    PPOConfig,
    _ppo_targets,
    carry_memory,
    check_warm_start,
    chunk_windows,
    clip_gradients,
    join_chunks,
    ppo_update,
    prepare_chunks,
)
from magezero.schema import EVENT_TYPES, collate_observations
from magezero.service import PolicyService

from helpers import observation, recorded_episode, varied_trajectory

torch.set_num_threads(1)
LABELS = {"opponent_hand_counts": [1, 1, 1]}


def _events(step, count):
    kinds = [EVENT_TYPES[i % len(EVENT_TYPES)] for i in range(count)]
    return [
        {
            "seq": step * 1000 + i,
            "type": kind,
            "actor": i % 3,
            "card_id": 0 if kind == "draw" else i % 4,
            "features": [i / 50] + [0.0] * 7,
        }
        for i, kind in enumerate(kinds)
    ]


def _bursty_trajectory(config, counts, game_id):
    """Decisions with 0, 1 or many new events, as Kinnan mana loops produce."""
    steps = [observation(config, t, game_id=game_id) for t in range(len(counts))]
    for t, (obs, count) in enumerate(zip(steps, counts)):
        obs["events"] = _events(t, count)
    return steps


def test_event_pooling_is_identical_stepwise_sequenced_and_batched(components):
    model, _, _ = components
    trajectories = [
        _bursty_trajectory(model.config, [3, 0, 40, 1, 0, 7, 200], "a"),
        _bursty_trajectory(model.config, [0, 0, 5], "b"),
        _bursty_trajectory(model.config, [1, 300, 2, 0], "c"),
    ]
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
            torch.testing.assert_close(out.logits[:, : logits.shape[1]], logits, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(out.memory.events, memory.events, atol=1e-5, rtol=1e-5)
        episodes = [
            {
                "steps": [
                    {"observation": o, "action_id": o["actions"][0]["id"], "log_probability": 0.0, "labels": LABELS}
                    for o in t
                ]
            }
            for t in trajectories
        ]
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
        batched = torch.cat([row[:, : logits.shape[1]] for row in rows[i]])
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
    assert clip_gradients(model, critic, 1.0)
    for before, parameter in zip(actor_before, model.parameters()):
        torch.testing.assert_close(parameter.grad, before)  # a large critic gradient no longer shrinks it
    critic_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in critic.parameters()]))
    assert critic_norm.item() == pytest.approx(1.0, rel=1e-4)
    next(critic.parameters()).grad[0] = float("nan")
    assert not clip_gradients(model, critic, 1.0)  # reported, not raised


def _recorded_episodes(model, catalog):
    service = PolicyService(model, catalog, "v")
    return [
        recorded_episode(service, varied_trajectory(model.config, n, f"g{n}"), "v", reward=r)
        for n, r in ((6, 1), (3, -1), (8, 1), (5, -1))
    ]


def test_non_finite_gradients_skip_the_step_and_only_a_fully_broken_update_fails(components, monkeypatch):
    model, critic, catalog = components
    episodes = _recorded_episodes(model, catalog)
    config = PPOConfig(epochs=1, episodes_per_batch=1, sequence_length=3, target_kl=0.0)
    verdicts = iter([False, True, False, True])
    monkeypatch.setattr("magezero.learner.clip_gradients", lambda *args: next(verdicts))
    m, c = copy.deepcopy(model), copy.deepcopy(critic)
    optimizer = torch.optim.AdamW(list(m.parameters()) + list(c.parameters()), lr=1e-3)
    metrics = ppo_update(m, c, optimizer, episodes, config, "v")
    assert metrics["nonfinite_steps"] == 2 and metrics["optimizer_steps"] == 2
    monkeypatch.undo()
    # A NaN critic poisons every advantage, so no step can be taken: that is an error.
    broken = copy.deepcopy(critic)
    with torch.no_grad():
        broken.value[-1].bias.fill_(float("nan"))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(broken.parameters()), lr=1e-3)
    with pytest.raises(RuntimeError, match="non-finite"):
        ppo_update(copy.deepcopy(model), broken, optimizer, episodes, config, "v")


def test_a_rare_log_probability_mismatch_is_masked_and_widespread_ones_are_refused(components):
    model, critic, catalog = components
    episodes = _recorded_episodes(model, catalog)
    choice_steps = [
        (e, s) for e, episode in enumerate(episodes) for s, step in enumerate(episode["steps"])
        if len(step["observation"]["actions"]) > 1
    ]  # fmt: skip
    e, s = choice_steps[0]
    episodes[e]["steps"][s]["log_probability"] -= 0.5
    prepared = [prepare_chunks(episode, model, 3) for episode in episodes]
    before = sum(int(c.choice.sum()) for chunks in prepared for c in chunks)
    _, mismatches = _ppo_targets(model, critic, episodes, prepared, PPOConfig(max_mismatch_fraction=0.1), None)
    assert mismatches == 1
    assert not prepared[e][s // 3].choice[s % 3]
    assert sum(int(c.choice.sum()) for chunks in prepared for c in chunks) == before - 1
    # The default bound (1% of decisions) is below one in these few decisions.
    with pytest.raises(ValueError, match="behavior policy mismatch on 1 of"):
        _ppo_targets(model, critic, episodes, [prepare_chunks(x, model, 3) for x in episodes], PPOConfig(), None)
    config = PPOConfig(epochs=1, episodes_per_batch=2, sequence_length=3, max_mismatch_fraction=0.1)
    m, c = copy.deepcopy(model), copy.deepcopy(critic)
    optimizer = torch.optim.AdamW(list(m.parameters()) + list(c.parameters()), lr=1e-3)
    assert ppo_update(m, c, optimizer, episodes, config, "v")["mismatched_decisions"] == 1


def test_reference_log_probabilities_are_compact_copies(components):
    model, critic, catalog = components
    episodes = _recorded_episodes(model, catalog)
    prepared = [prepare_chunks(episode, model, 3) for episode in episodes]
    _ppo_targets(model, critic, episodes, prepared, PPOConfig(), copy.deepcopy(model))
    for chunks in prepared:
        for chunk in chunks:
            reference = chunk.reference_log_probability
            assert reference.shape == chunk.batch.action_mask.shape and reference.is_contiguous()
            assert reference.untyped_storage().nbytes() == reference.numel() * reference.element_size()


def test_kl_early_stop_skips_the_step_and_entropy_is_normalized(components, tmp_path):
    model, critic, catalog = components
    service = PolicyService(model, catalog, "v")
    episodes = [
        recorded_episode(service, varied_trajectory(model.config, n, f"g{n}"), "v", reward=r)
        for n, r in ((6, 1), (3, -1), (8, 1), (5, -1))
    ]

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
