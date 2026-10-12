import copy
import itertools
import json
import math

import pytest
import torch

from magezero.policy.belief import constrained_hand_belief
from magezero.policy.catalog import CardCatalog
from magezero.policy.checkpoint import load_checkpoint, save_checkpoint
from magezero.policy.learner import PPOConfig, generalized_advantages, ppo_update, validate_episode
from magezero.policy.model import EntityBeliefPolicy, ModelConfig, PrivilegedCritic
from magezero.policy.schema import collate_observations, validate_observation
from magezero.policy.service import PolicyService, create_app

torch.set_num_threads(1)


@pytest.fixture
def components():
    torch.manual_seed(2)
    catalog = CardCatalog.from_dict({"format_version": 1, "cards": [
        {"id": i, "key": f"card-{i}", "name": f"Card {i}", "features": [0.0] * 32}
        for i in range(1, 4)]})
    config = ModelConfig(num_cards=catalog.num_cards, catalog_digest=catalog.digest,
                         width=32, entity_layers=1, max_hand_size=6, belief_mode="learned")
    return EntityBeliefPolicy(config, catalog.tensor()).eval(), PrivilegedCritic(config), catalog


def observation(config, step=0, game_id="game", player_id="player"):
    return {"schema_version": 2, "catalog_digest": config.catalog_digest, "game_id": game_id,
        "player_id": player_id, "step": step,
        "entities": [
            {"id": "me", "card_id": 0, "kind": 0, "zone": 0, "owner": 0, "controller": 0,
             "visibility": "public", "features": [0.0] * 32},
            {"id": "creature", "card_id": 1, "kind": 1, "zone": 1, "owner": 0, "controller": 0,
             "visibility": "public", "features": [0.1] + [0.0] * 31}],
        "relations": [{"source": "creature", "target": "me", "type": 0}],
        "globals": [step / 10] + [0.0] * 23,
        "events": [{"seq": step, "type": "pass", "actor": 1, "card_id": 0, "features": [0.0] * 8}],
        "actions": [
            {"id": "pass", "type": "pass", "source": None, "target": None, "card_id": 0,
             "features": [0.0] * 16},
            {"id": "attack", "type": "attack", "source": "creature", "target": "me", "card_id": 1,
             "features": [1.0] + [0.0] * 15}],
        "opponent_deck": [{"card_id": 1, "lower": 0, "upper": 2},
                          {"card_id": 2, "lower": 0, "upper": 2},
                          {"card_id": 3, "lower": 0, "upper": 2}],
        "opponent_hand_size": 3}


def test_joint_belief_matches_enumeration_and_has_finite_gradients():
    logits = torch.randn(1, 3, 4, requires_grad=True)
    lower, upper = torch.tensor([[1, 0, 0]]), torch.tensor([[2, 2, 2]])
    belief = constrained_hand_belief(logits, lower, upper, torch.tensor([3]))
    hands = [n for n in itertools.product(range(3), repeat=3) if sum(n) == 3 and n[0] >= 1]
    masses = torch.softmax(torch.stack([sum(logits[0, i, n[i]] for i in range(3)) for n in hands]), 0)
    expected = torch.zeros(3, 4)
    for mass, hand in zip(masses, hands):
        for i, count in enumerate(hand):
            expected[i, count] += mass
    torch.testing.assert_close(belief.probabilities[0], expected)
    torch.testing.assert_close(belief.expected_counts.sum(), torch.tensor(3.0))
    joint_index = hands.index((1, 1, 1))
    torch.testing.assert_close(belief.nll(torch.tensor([[1, 1, 1]]))[0], -masses[joint_index].log())
    (belief.nll(torch.tensor([[1, 1, 1]])).sum() + belief.expected_counts[:, 0].sum()).backward()
    assert torch.isfinite(logits.grad).all()


def test_empty_hand_padding_and_infeasible_bounds():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    belief = constrained_hand_belief(logits, torch.zeros(2, 3, dtype=torch.long),
        torch.tensor([[0, 0, 0], [2, 2, 0]]), torch.tensor([0, 3]))
    torch.testing.assert_close(belief.expected_counts.sum(-1), torch.tensor([0.0, 3.0]))
    torch.testing.assert_close(belief.probabilities[0, :, 0], torch.ones(3))
    belief.nll(torch.tensor([[0, 0, 0], [1, 2, 0]])).sum().backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="hand size"):
        constrained_hand_belief(logits[:1], torch.tensor([[2, 2, 0]]),
                                torch.tensor([[2, 2, 0]]), torch.tensor([3]))


def test_entity_and_action_permutations_and_padding(components):
    model, _, _ = components
    obs = observation(model.config)
    permuted = copy.deepcopy(obs)
    permuted["entities"].reverse()
    permuted["actions"].reverse()
    with torch.no_grad():
        normal = model(collate_observations([obs], model.config))
        perm = model(collate_observations([permuted], model.config))
    torch.testing.assert_close(normal.logits, perm.logits.flip(-1))
    torch.testing.assert_close(normal.value, perm.value)
    shorter = copy.deepcopy(obs)
    shorter["entities"] = shorter["entities"][:1]
    shorter["relations"] = []
    shorter["actions"] = shorter["actions"][:1]
    shorter["events"] = []
    with torch.no_grad():
        mixed = model(collate_observations([obs, shorter], model.config))
        alone = model(collate_observations([shorter], model.config))
    torch.testing.assert_close(mixed.logits[1, :1], alone.logits[0], atol=1e-6, rtol=1e-5)
    assert mixed.distribution().probs[1, 1].item() == 0
    assert torch.isfinite(mixed.value).all()


def test_memory_changes_decisions_and_reset_restores_them(components):
    model, _, _ = components
    batch = collate_observations([observation(model.config)], model.config)
    with torch.no_grad():
        first = model(batch)
        remembered = model(batch, first.memory)
        reset = model(batch, first.memory, torch.tensor([True]))
    assert not torch.allclose(first.logits, remembered.logits)
    torch.testing.assert_close(first.logits, reset.logits)


def test_private_actor_fields_and_catalog_mismatch_are_rejected(components):
    model, _, _ = components
    obs = observation(model.config)
    obs["opponent_hand_counts"] = [1, 1, 1]
    with pytest.raises(ValueError, match="training labels"):
        validate_observation(obs, model.config)
    obs.pop("opponent_hand_counts")
    obs["entities"][1]["visibility"] = "hidden"
    with pytest.raises(ValueError, match="hidden zones"):
        validate_observation(obs, model.config)
    obs = observation(model.config)
    obs["events"][0] = {"seq": 0, "type": "draw", "actor": 1, "card_id": 2, "features": [0.0] * 8}
    with pytest.raises(ValueError, match="private"):
        validate_observation(obs, model.config)
    obs = observation(model.config)
    obs["catalog_digest"] = "different"
    with pytest.raises(ValueError, match="catalog"):
        validate_observation(obs, model.config)


def test_privileged_critic_cannot_backpropagate_into_actor(components):
    model, critic, _ = components
    batch = collate_observations([observation(model.config)], model.config)
    output = model(batch)
    critic(output.context, batch.deck_cards, torch.tensor([[1.0, 1.0, 1.0]])).sum().backward()
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None for p in critic.parameters())


def test_returns_terminal_and_truncated_bootstrapping():
    rewards, values = torch.tensor([0.0, 1.0]), torch.tensor([0.2, 0.3])
    advantages, returns = generalized_advantages(rewards, values, torch.tensor([False, True]),
                                                  gamma=1, gae_lambda=1)
    torch.testing.assert_close(returns, torch.ones(2))
    torch.testing.assert_close(advantages, torch.tensor([0.8, 0.7]))
    _, returns = generalized_advantages(torch.tensor([0.0]), torch.tensor([0.3]),
        torch.tensor([False]), torch.tensor([True]), bootstrap_value=0.6)
    torch.testing.assert_close(returns, torch.tensor([0.6]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression check")
def test_cuda_recurrent_ppo_backward(components):
    model, critic, catalog = components
    model.cuda(); critic.cuda()
    obs = observation(model.config)
    service = PolicyService(model, catalog, "cuda-test")
    response = service.act({"observation": obs, "sample": True})
    episode = {"schema_version": 2, "policy_version": "cuda-test", "terminal": True,
        "steps": [{"observation": obs, "action_id": response["action_id"],
            "log_probability": response["log_probability"], "reward": 1,
            "labels": {"opponent_hand_counts": [1, 1, 1]}}]}
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()), lr=1e-4)
    metrics = ppo_update(model, critic, optimizer, [episode], PPOConfig(epochs=1), "cuda-test")
    assert metrics["decisions"] == 1
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_service_idempotence_history_recording_and_checkpoint_training(components, tmp_path):
    model, critic, catalog = components
    checkpoint = tmp_path / "initial.pt"
    save_checkpoint(checkpoint, model, critic, catalog, policy_version="initial")
    restored, restored_critic, _, data = load_checkpoint(checkpoint)
    service = PolicyService(restored, catalog, "initial", tmp_path / "episodes")
    client = create_app(service).test_client()
    for t in range(4):
        payload = {"observation": observation(model.config, t), "sample": True}
        first = client.post("/act", json=payload)
        assert first.status_code == 200, first.json
        retry = client.post("/act", json=payload)
        assert retry.json == first.json
    repeated = observation(model.config, 4)
    repeated["events"][0]["seq"] = 3
    assert client.post("/act", json={"observation": repeated, "sample": True}).status_code == 400
    finish = {"game_id": "game", "player_id": "player", "terminal": True, "result": 1,
              "labels": [{"opponent_hand_counts": [1, 1, 1]}] * 4}
    response = client.post("/finish", json=finish)
    assert response.status_code == 200, response.json
    assert client.post("/finish", json=finish).json == response.json
    assert len(list((tmp_path / "episodes").glob("*.jsonl"))) == 1
    episode = json.loads(open(response.json["episode_file"]).read())
    assert [s["reward"] for s in episode["steps"]] == [0, 0, 0, 1]
    optimizer = torch.optim.Adam(list(restored.parameters()) + list(restored_critic.parameters()), lr=1e-3)
    before = restored.belief_head[0].weight.detach().clone()
    metrics = ppo_update(restored, restored_critic, optimizer, [episode],
                         PPOConfig(epochs=2, sequence_length=2), "initial")
    assert metrics["decisions"] >= 4
    assert all(torch.isfinite(p).all() for p in restored.parameters())
    assert not torch.allclose(before, restored.belief_head[0].weight)
    with pytest.raises(ValueError, match="stale"):
        validate_episode(episode, model.config, "new-version")


def test_aborted_games_are_not_draw_training_data(components, tmp_path):
    model, _, catalog = components
    service = PolicyService(model, catalog, "version", tmp_path)
    service.act({"observation": observation(model.config), "sample": True})
    result = service.finish({"game_id": "game", "player_id": "player", "result": 0,
                             "terminal": False, "labels": []})
    assert not result["recorded"]
    assert not list(tmp_path.glob("*.jsonl"))


def _trajectory(config, length=5):
    steps = [observation(config, t) for t in range(length)]
    steps[2]["events"] = []  # a decision with no new history must carry memory
    steps[3]["actions"] = steps[3]["actions"][:1]  # a forced decision
    return steps


def test_sequence_forward_matches_stepwise_inference(components):
    model, _, _ = components
    steps = _trajectory(model.config)
    with torch.no_grad():
        memory, stepwise = None, []
        for obs in steps:
            out = model(collate_observations([obs], model.config), memory)
            memory = out.memory
            stepwise.append(out)
        # Split across two chunks to cover carried memory between sequence calls.
        first = model.forward_sequence(collate_observations(steps[:3], model.config))
        second = model.forward_sequence(collate_observations(steps[3:], model.config), first.memory)
    for row, out in enumerate(stepwise):
        chunk, index = (first, row) if row < 3 else (second, row - 3)
        width = out.logits.shape[-1]
        torch.testing.assert_close(chunk.logits[index, :width], out.logits[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(chunk.value[index], out.value[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(second.memory.decisions, memory.decisions, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(second.memory.events, memory.events, atol=1e-5, rtol=1e-5)


def test_grucell_checkpoints_load_into_sequence_memory(components):
    model, _, _ = components
    state = model.state_dict()
    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        state[f"decision_memory.{name}"] = state.pop(f"decision_memory.{name}_l0")
    fresh = EntityBeliefPolicy(model.config, model.card_features).eval()
    fresh.load_state_dict(state)
    batch = collate_observations([observation(model.config)], model.config)
    with torch.no_grad():
        torch.testing.assert_close(fresh(batch).logits, model(batch).logits)


def test_forced_decisions_are_excluded_from_policy_statistics(components):
    model, critic, catalog = components
    service = PolicyService(model, catalog, "forced")
    steps = []
    for obs in _trajectory(model.config):
        response = service.act({"observation": obs, "sample": True})
        steps.append({"observation": obs, "action_id": response["action_id"],
            "log_probability": response["log_probability"], "reward": 0,
            "labels": {"opponent_hand_counts": [1, 1, 1]}})
    steps[-1]["reward"] = 1
    episode = {"schema_version": 2, "policy_version": "forced", "terminal": True, "steps": steps}
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()), lr=1e-4)
    metrics = ppo_update(model, critic, optimizer, [episode], PPOConfig(epochs=1, sequence_length=2), "forced")
    assert metrics["decisions"] == 5 and metrics["choice_decisions"] == 4
    assert metrics["entropy"] > 0 and math.isfinite(metrics["critic_loss"])


def test_batched_inference_matches_individual_requests(components):
    from magezero.policy.service import Session, _Job
    model, _, catalog = components
    service = PolicyService(model, catalog, "batch")
    observations = [observation(model.config, 0, game_id=f"g{i}") for i in range(3)]
    observations[1]["entities"] = observations[1]["entities"][:1]
    observations[1]["relations"] = []
    observations[1]["actions"] = observations[1]["actions"][:1]
    observations[2]["events"] = []
    observations[2]["opponent_deck"] = observations[2]["opponent_deck"][:2]
    jobs = [_Job((o["game_id"], "player"), Session(sampled=False), o, "", True) for o in observations]
    batched, batched_memory = service._infer(jobs)
    for job, response, memory in zip(jobs, batched, batched_memory):
        (alone,), (alone_memory,) = service._infer([job])
        assert response["action_id"] == alone["action_id"]
        assert len(response["belief"]["expected_counts"]) == len(job.observation["opponent_deck"])
        for field in ("log_probability", "value"):
            assert response[field] == pytest.approx(alone[field], abs=1e-5)
        torch.testing.assert_close(torch.tensor(response["probabilities"]),
                                   torch.tensor(alone["probabilities"]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(memory.decisions, alone_memory.decisions, atol=1e-5, rtol=1e-5)


def test_concurrent_retries_share_one_inference(components):
    import threading
    model, _, catalog = components
    service = PolicyService(model, catalog, "retry")
    payload = {"observation": observation(model.config), "sample": True}
    results, barrier = [], threading.Barrier(6)

    def call():
        barrier.wait()
        results.append(service.act(copy.deepcopy(payload)))
    threads = [threading.Thread(target=call) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 6 and all(r == results[0] for r in results)
    assert service.stats()["inferences"] == 1


def test_inference_failure_releases_requests_and_new_sessions(components, monkeypatch):
    model, _, catalog = components
    service = PolicyService(model, catalog, "failure")

    def broken(jobs):
        raise RuntimeError("device lost")
    monkeypatch.setattr(service, "_infer", broken)
    with pytest.raises(RuntimeError, match="device lost"):
        service.act({"observation": observation(model.config), "sample": True})
    assert not service.sessions


def test_compact_responses_omit_only_belief_diagnostics(components):
    model, _, catalog = components
    payload = {"observation": observation(model.config), "sample": False}
    full = PolicyService(model, catalog, "v").act(copy.deepcopy(payload))
    compact = PolicyService(model, catalog, "v", include_belief=False).act(copy.deepcopy(payload))
    assert "belief" in full and "belief" not in compact
    assert compact == {key: value for key, value in full.items() if key != "belief"}


def test_reload_swaps_checkpoints_only_between_games(components, tmp_path):
    model, critic, catalog = components
    save_checkpoint(tmp_path / "next.pt", model, critic, catalog, policy_version="next")
    assert create_app(PolicyService(model, catalog, "v")).test_client().post(
        "/reload", json={"checkpoint": str(tmp_path / "next.pt"), "episodes": None}).status_code == 404
    service = PolicyService(model, catalog, "first", tmp_path / "first")
    client = create_app(service, allow_reload=True).test_client()
    reload = {"checkpoint": str(tmp_path / "next.pt"), "episodes": str(tmp_path / "second")}
    assert client.post("/act", json={"observation": observation(model.config), "sample": True}).status_code == 200
    refused = client.post("/reload", json=reload)
    assert refused.status_code == 400 and "active" in refused.json["error"]
    client.post("/finish", json={"game_id": "game", "player_id": "player", "result": 0,
                                 "terminal": False, "labels": []})
    assert client.post("/reload", json=reload).json == {"policy_version": "next"}
    assert client.get("/healthz").json["policy_version"] == "next"
    assert service.output_dir == tmp_path / "second" and service.stats()["act_requests"] == 0
    # The finished game ID was from the previous policy; it may be reused afterwards.
    assert client.post("/act", json={"observation": observation(model.config), "sample": True}).status_code == 200
    other = CardCatalog.from_dict({"format_version": 1, "cards": [
        {"id": 1, "key": "other", "name": "Other", "features": [0.0] * 32}]})
    with pytest.raises(ValueError, match="catalog"):
        service.reload(model, other, "other")
