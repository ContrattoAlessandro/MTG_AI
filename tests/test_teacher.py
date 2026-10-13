"""Behaviour-cloning warm start, reference KL, and potential-based shaping."""

import copy
import json

import pytest
import torch

from magezero import runner
from magezero.checkpoint import load_checkpoint, save_checkpoint
from magezero.learner import (PPOConfig, generalized_advantages, ppo_update, reference_coefficient,
                                     reference_kl, validate_episode)
from magezero.schema import ACTION_TYPES, collate_observations
from magezero.service import TEACHER_VERSION, PolicyService
from magezero.shaping import potential, shaped_rewards
from magezero.teacher import CloneConfig, behavior_cloning, teacher_game_config
from test_entity_policy import components, observation  # noqa: F401  (fixture)
from test_long_run import _episode

torch.set_num_threads(1)
LABELS = {"opponent_hand_counts": [1, 1, 1]}


def _players(config, my_life=0.5, their_life=0.5, step=0):
    obs = observation(config, step)
    obs["entities"][1]["features"][22] = 1  # the helper's creature becomes a land: no board term
    obs["entities"][0]["features"][11] = my_life
    rival = copy.deepcopy(obs["entities"][0])
    rival.update(id="rival", owner=1, controller=1)
    rival["features"][11] = their_life
    obs["entities"].append(rival)
    return obs


def test_potential_is_antisymmetric_bounded_and_counts_every_loss_condition(components):
    config = components[0].config
    assert potential(_players(config)) == pytest.approx(0.0, abs=1e-12)
    ahead, behind = potential(_players(config, 0.9, 0.2)), potential(_players(config, 0.2, 0.9))
    assert 0 < ahead < 1 and ahead == pytest.approx(-behind)
    poisoned = _players(config, 0.9, 0.2)
    poisoned["entities"][0]["features"][12] = 0.9  # nine poison counters dominate a high life total
    assert potential(poisoned) < 0


def test_shaping_telescopes_and_leaves_monte_carlo_advantages_unchanged(components):
    config = components[0].config
    steps = [{"observation": _players(config, 0.5 + 0.1 * t, 0.5, t), "reward": 0} for t in range(4)]
    steps[-1]["reward"] = 1
    episode = {"steps": steps, "terminal": True}
    phi = torch.tensor([0.5 * potential(s["observation"]) for s in steps])
    shaped = shaped_rewards(episode, 0.5)
    assert float(shaped.sum()) == pytest.approx(1 - float(phi[0]), abs=1e-6)  # phi(terminal) = 0
    # gamma = lambda = 1: the shaped return is R - phi(s_t), i.e. only a baseline shift.
    values = torch.zeros(4)
    end = torch.tensor([False, False, False, True])
    _, plain = generalized_advantages(shaped_rewards(episode, 0.0), values, end, gae_lambda=1.0)
    _, shaped_returns = generalized_advantages(shaped, values, end, gae_lambda=1.0)
    assert torch.allclose(shaped_returns, plain - phi, atol=1e-6)
    # A truncated game has no next observation: its last step gets no shaping term.
    truncated = {"steps": copy.deepcopy(steps), "terminal": False, "truncated": True}
    truncated["steps"][-1]["reward"] = 0
    assert float(shaped_rewards(truncated, 0.5)[-1]) == pytest.approx(0.0, abs=1e-7)


def test_reference_kl_matches_the_definition_on_legal_candidates():
    logits = torch.tensor([[0.3, -1.0, float("-inf")], [2.0, 0.0, 1.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    reference = torch.log_softmax(torch.tensor([[1.0, 0.0, float("-inf")], [0.0, 0.0, 0.0]]), -1)
    expected = (reference.exp() * (reference - torch.log_softmax(logits, -1))).nan_to_num(0.0).sum(-1)
    assert torch.allclose(reference_kl(reference, logits, mask), expected)
    assert torch.allclose(reference_kl(torch.log_softmax(logits, -1), logits, mask), torch.zeros(2), atol=1e-7)
    assert reference_coefficient(PPOConfig(reference_kl_coefficient=0.1, reference_kl_decay=0.5), 7, 5) == 0.025


def _games(model, catalog, tag, count=4):
    """Fresh on-policy episodes from the model's current weights."""
    service = PolicyService(model, catalog, "v")
    return [_episode(service, [observation(model.config, t, game_id=f"{tag}{g}") for t in range(6)], "v",
                     reward=1 if g % 2 else -1) for g in range(count)]


def test_ppo_reference_kl_is_zero_for_the_same_policy_and_pulls_towards_the_reference(components):
    model, critic, catalog = components
    config = PPOConfig(epochs=1, episodes_per_batch=2, sequence_length=4, reference_kl_coefficient=0.5,
                       shaping_coefficient=0.3, gae_lambda=0.95)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()), lr=0.0)
    same = ppo_update(model, critic, optimizer, _games(model, catalog, "same"), config, "v",
                      reference=copy.deepcopy(model))
    assert same["reference_kl"] == pytest.approx(0.0, abs=1e-6)
    assert same["reference_kl_coefficient"] == 0.5 and same["shaping_coefficient"] == 0.3
    without = ppo_update(model, critic, optimizer, _games(model, catalog, "none"), config, "v")
    assert without["reference_kl_coefficient"] == 0.0
    # A reference with different action preferences: a strong KL term moves the policy towards it.
    reference = copy.deepcopy(model)

    def prefers_attacks(batch, *args, head=reference._head):
        out = head(batch, *args)
        out.logits = out.logits + 2.0 * (batch.action_types == ACTION_TYPES.index("attack"))
        return out
    reference._head = prefers_attacks
    batch = collate_observations([observation(model.config)], model.config)
    attack = lambda m: m(batch).distribution().probs[0, 1].item()  # noqa: E731
    before = abs(attack(model) - attack(reference))
    assert before > 1e-3
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()), lr=3e-3)
    strong = PPOConfig(epochs=4, episodes_per_batch=2, sequence_length=4, reference_kl_coefficient=50.0,
                       entropy_coefficient=0.0, target_kl=100.0)
    for round_ in range(3):
        result = ppo_update(model, critic, optimizer, _games(model, catalog, f"r{round_}-"), strong, "v",
                            reference=reference)
        assert result["reference_kl"] > 0
    assert abs(attack(model) - attack(reference)) < before


def _teacher_episode(service, config, game, player, length=6, result=1):
    """XMage's AI stand-in: attacks whenever it can, records through /record."""
    for t in range(length):
        obs = observation(config, t, game_id=game, player_id=player)
        choice = "attack" if any(a["id"] == "attack" for a in obs["actions"]) else obs["actions"][0]["id"]
        service.record({"observation": obs, "action_id": choice})
    return service.finish({"game_id": game, "player_id": player, "result": result, "terminal": True,
                           "labels": [LABELS] * length})


def test_record_sessions_are_separate_idempotent_and_write_teacher_episodes(components, tmp_path):
    model, _, catalog = components
    service = PolicyService(model, catalog, "v", tmp_path)
    obs = observation(model.config, 0, game_id="t")
    first = service.record({"observation": obs, "action_id": "attack"})
    assert service.record({"observation": obs, "action_id": "attack"}) == first  # retry
    with pytest.raises(ValueError, match="retry changed"):
        service.record({"observation": obs, "action_id": "pass"})
    with pytest.raises(ValueError, match="cannot request policy actions"):
        service.act({"observation": observation(model.config, 1, game_id="t"), "sample": True})
    with pytest.raises(ValueError, match="contiguous"):
        service.record({"observation": observation(model.config, 3, game_id="t"), "action_id": "pass"})
    with pytest.raises(ValueError, match="not one of"):
        service.record({"observation": observation(model.config, 1, game_id="t"), "action_id": "cast"})
    service.act({"observation": observation(model.config, 0, game_id="p"), "sample": True})
    with pytest.raises(ValueError, match="plays the policy"):
        service.record({"observation": observation(model.config, 1, game_id="p"), "action_id": "pass"})
    service.record({"observation": observation(model.config, 1, game_id="t"), "action_id": "pass"})
    response = service.finish({"game_id": "t", "player_id": "player", "result": -1, "terminal": True,
                               "labels": [LABELS, LABELS]})
    episode = json.loads(open(response["episode_file"], encoding="utf-8").read())
    assert episode["policy_version"] == TEACHER_VERSION
    assert [s["action_id"] for s in episode["steps"]] == ["attack", "pass"]
    assert [s["reward"] for s in episode["steps"]] == [0, -1]
    validate_episode(episode, model.config, TEACHER_VERSION)
    with pytest.raises(ValueError, match="episode directory"):
        PolicyService(model, catalog, "v").record({"observation": obs, "action_id": "attack"})


def test_behavior_cloning_fits_the_teacher_and_pretrains_values(components, tmp_path):
    model, critic, catalog = components
    start = tmp_path / "start.pt"
    save_checkpoint(start, model, critic, catalog, policy_version="start")
    service = PolicyService(model, catalog, "start", tmp_path / "teacher")
    for game in range(8):
        _teacher_episode(service, model.config, f"g{game}", "a", result=1)
        _teacher_episode(service, model.config, f"g{game}", "b", result=-1)
    batch = collate_observations([observation(model.config)], model.config)
    before = model(batch).distribution().probs[0, 1].item()
    config = CloneConfig(epochs=6, episodes_per_batch=4, sequence_length=4, learning_rate=3e-3,
                         validation_fraction=0.25, shaping_coefficient=0.2)
    metrics = behavior_cloning(start, [tmp_path / "teacher"], tmp_path / "cloned.pt", config, "cpu")
    clone = metrics["behavior_cloning"]
    assert clone["validation_episodes"] == 4 and clone["train_episodes"] == 12  # 2 of 8 games, both seats
    best = clone["history"][clone["best_epoch"]]
    assert best["validation"]["accuracy"] == 1.0
    assert best["validation"]["nll"] < clone["history"][0]["validation"]["nll"] or clone["best_epoch"] == 0
    cloned, _, _, data = load_checkpoint(tmp_path / "cloned.pt")
    assert cloned(batch).distribution().probs[0, 1].item() > max(before, 0.9)
    assert data["policy_version"] == metrics["policy_version"] and data["optimizer_state_dict"] is None
    assert best["train"]["accuracy_by_type"]["attack"] == 1.0


def test_clone_and_runner_configuration_checks(components, tmp_path):
    with pytest.raises(ValueError):
        CloneConfig(validation_fraction=1.0)
    with pytest.raises(ValueError):
        PPOConfig(reference_kl_decay=0.0)
    base = {"player_a": {"deckPath": "a.dck", "type": "policy"}, "player_b": {"deckPath": "b.dck", "type": "policy"},
            "training": {"games": 1}}
    config = teacher_game_config(base, 4, 50060, 2, 5, "a")
    assert config["player_a"]["type"] == "xmage-teacher" and config["player_b"]["type"] == "xmage-ai"
    assert config["opponent_ai"] == {"skill": 2, "think_seconds": 5}
    assert config["server"]["learner_keeps_deck"] is True and config["training"]["games"] == 4
    model, critic, catalog = components
    save_checkpoint(tmp_path / "c.pt", model, critic, catalog)
    script = tmp_path / "engine.bat"
    script.write_text("")
    policy = tmp_path / "policy.yml"
    policy.write_text("training:\n  reference_kl_coefficient: 0.1\n", encoding="utf-8")
    deck = tmp_path / "deck.dck"
    deck.write_text("")
    game = tmp_path / "game.yml"
    game.write_text(json.dumps({"player_a": {"deckPath": str(deck), "type": "policy"},
                                "player_b": {"deckPath": str(deck), "type": "policy"},
                                "training": {"games": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="needs --reference"):
        runner.run_selfplay(tmp_path / "c.pt", game, policy, script, tmp_path / "run", iterations=1)


def test_teacher_games_are_indexed_on_disk_and_loaded_per_trajectory(components, tmp_path):
    """mz clone keeps only offsets and returns in memory; several trajectories may share a file."""
    from magezero.teacher import _TeacherGames, _split
    model, critic, catalog = components
    service = PolicyService(model, catalog, "start", tmp_path / "teacher")
    for game in range(3):
        for seat, result in (("a", 1), ("b", -1)):
            _teacher_episode(service, model.config, f"g{game}", seat, result=result)
    files = sorted((tmp_path / "teacher").rglob("*.jsonl"))
    combined = tmp_path / "combined.jsonl"
    combined.write_bytes(b"".join(f.read_bytes().rstrip(b"\r\n") + b"\n" for f in files) + b"\n")
    games = _TeacherGames([combined], model.config, CloneConfig(shaping_coefficient=0.2))
    assert len(games) == 6
    for i, path in enumerate(files):
        assert games.load(i) == json.loads(path.read_text(encoding="utf-8"))
    assert all(len(entry.returns) == 6 and entry.valued for entry in games.entries)
    train, held = _split(games, 1 / 3, seed=0)
    assert len(held) == 2 and len({games.entries[i].game_id for i in held}) == 1  # both seats of one game
    with pytest.raises(ValueError, match="duplicate"):
        _TeacherGames([combined, files[0]], model.config, CloneConfig())
