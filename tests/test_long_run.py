"""Long-run machinery: batched learner, truncation bootstrap, resumable runner."""

import copy
import json
import math

import pytest
import torch
import yaml

from magezero.policy import runner
from magezero.policy.checkpoint import load_checkpoint, save_checkpoint
from magezero.policy.learner import (PPOConfig, _carry, _join, _prepare_chunks, _windows,
                                     ppo_update, validate_episode)
from magezero.policy.schema import collate_observations
from magezero.policy.service import PolicyService
from test_entity_policy import components, observation  # noqa: F401  (fixture)

torch.set_num_threads(1)


def _varied_trajectory(config, length, game_id, events=True):
    steps = [observation(config, t, game_id=game_id) for t in range(length)]
    for t, obs in enumerate(steps):
        if not events or t % 3 == 2:
            obs["events"] = []  # decisions without new history must carry memory
        if t % 4 == 3:
            obs["actions"] = obs["actions"][:1]
        if t % 2:
            obs["entities"] = obs["entities"][:1]  # vary padding widths across rows
            obs["relations"] = []
            obs["actions"] = [a for a in obs["actions"] if a["source"] is None]
    return steps


def _episode(service, steps, version, *, terminal=True, reward=1):
    recorded = []
    for obs in steps:
        response = service.act({"observation": obs, "sample": True})
        recorded.append({"observation": obs, "action_id": response["action_id"],
                         "log_probability": response["log_probability"], "reward": 0,
                         "labels": {"opponent_hand_counts": [1, 1, 1]}})
    if terminal:
        recorded[-1]["reward"] = reward
    episode = {"schema_version": 2, "policy_version": version, "terminal": terminal, "steps": recorded}
    if not terminal:
        episode["truncated"] = True
    return episode


def test_multi_episode_forward_matches_separate_sequences(components):
    model, _, _ = components
    trajectories = [_varied_trajectory(model.config, 7, "a"),
                    _varied_trajectory(model.config, 2, "b"),
                    _varied_trajectory(model.config, 5, "c", events=False)]
    episodes = [{"steps": [{"observation": o, "action_id": o["actions"][0]["id"], "log_probability": 0.0,
                            "labels": {"opponent_hand_counts": [1, 1, 1]}} for o in t]} for t in trajectories]
    prepared = [_prepare_chunks(e, model, 3) for e in episodes]
    with torch.no_grad():
        expected = []
        for steps in trajectories:
            out = model.forward_sequence(collate_observations(steps, model.config))
            expected.append(out)
        memory, previous, rows = None, [0, 1, 2], {i: [] for i in range(3)}
        final = {}
        for window, members in _windows([0, 1, 2], prepared):
            memory = _carry(memory, previous, members)
            joined = _join([prepared[i][window] for i in members])
            sizes = [len(prepared[i][window].actions) for i in members]
            out = model.forward_episodes(joined.batch, sizes, memory)
            # The learner's sync-free path (CPU event counts) must agree exactly.
            hinted = model.forward_episodes(joined.batch, sizes, memory, joined.event_counts)
            torch.testing.assert_close(hinted.logits, out.logits)
            torch.testing.assert_close(hinted.memory.events, out.memory.events)
            for i, logits, value, events, decisions in zip(
                    members, out.logits.split(sizes), out.value.split(sizes),
                    out.memory.events, out.memory.decisions):
                rows[i].append((logits, value))
                final[i] = (events, decisions)
            memory, previous = out.memory, members
    for i, out in enumerate(expected):
        logits = torch.cat([l[:, :out.logits.shape[1]] for l, _ in rows[i]])
        width = out.logits.shape[1]
        torch.testing.assert_close(logits[:, :width], out.logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.cat([v for _, v in rows[i]]), out.value, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final[i][0], out.memory.events[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final[i][1], out.memory.decisions[0], atol=1e-5, rtol=1e-5)


def test_target_batching_does_not_change_the_update(components):
    model, critic, catalog = components
    service = PolicyService(model, catalog, "v")
    episodes = [_episode(service, _varied_trajectory(model.config, n, f"g{n}"), "v", reward=r)
                for n, r in ((6, 1), (3, -1), (8, 1))]
    results = []
    for target_batch in (1, 3):
        torch.manual_seed(0)
        m, c = copy.deepcopy(model), copy.deepcopy(critic)
        optimizer = torch.optim.AdamW(list(m.parameters()) + list(c.parameters()), lr=1e-3)
        metrics = ppo_update(m, c, optimizer, episodes,
                             PPOConfig(epochs=1, episodes_per_batch=3, sequence_length=3,
                                       target_batch_episodes=target_batch), "v")
        results.append((metrics, [p.detach().clone() for p in m.parameters()]))
    assert results[0][0]["decisions"] == results[1][0]["decisions"] == 17
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_truncated_games_are_recorded_bootstrapped_and_counted(components, tmp_path):
    model, critic, catalog = components
    service = PolicyService(model, catalog, "v", tmp_path)
    steps = _varied_trajectory(model.config, 4, "long")
    for obs in steps:
        service.act({"observation": obs, "sample": True})
    with pytest.raises(ValueError, match="truncated"):
        service.finish({"game_id": "long", "player_id": "player", "result": 1, "terminal": False,
                        "truncated": True, "labels": [{"opponent_hand_counts": [1, 1, 1]}] * 4})
    response = service.finish({"game_id": "long", "player_id": "player", "result": 0, "terminal": False,
                               "truncated": True, "labels": [{"opponent_hand_counts": [1, 1, 1]}] * 4})
    assert response["recorded"] and response["truncated"]
    episode = json.loads(open(response["episode_file"]).read())
    assert episode["truncated"] is True and episode["terminal"] is False
    assert service.stats()["results"]["truncated"] == 1
    finished = _episode(service, _varied_trajectory(model.config, 3, "short"), "v", reward=-1)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()), lr=1e-4)
    metrics = ppo_update(model, critic, optimizer, [episode, finished],
                         PPOConfig(epochs=1, episodes_per_batch=2, sequence_length=2), "v")
    assert metrics["truncated_episodes"] == 1 and math.isfinite(metrics["critic_loss"])
    rewarded = copy.deepcopy(episode)
    rewarded["steps"][-1]["reward"] = 1
    with pytest.raises(ValueError, match="no result"):
        validate_episode(rewarded, model.config, "v")
    both = copy.deepcopy(episode)
    both["terminal"] = True
    with pytest.raises(ValueError, match="exactly one"):
        validate_episode(both, model.config, "v")


def test_service_outcome_counters_reset_on_reload(components, tmp_path):
    model, _, catalog = components
    service = PolicyService(model, catalog, "v")
    for game, result, terminal in (("w", 1, True), ("l", -1, True), ("x", 0, False)):
        service.act({"observation": observation(model.config, game_id=game), "sample": True})
        service.finish({"game_id": game, "player_id": "player", "result": result, "terminal": terminal,
                        "labels": []})
    assert service.stats()["results"] == {"win": 1, "loss": 1, "draw": 0, "truncated": 0, "aborted": 1}
    service.reload(model, catalog, "v2")
    assert sum(service.stats()["results"].values()) == 0


def test_wilson_interval_and_selection_metric():
    low, high = runner.wilson_interval(149, 200)
    assert abs(low - 0.680) < 0.002 and abs(high - 0.800) < 0.002
    assert runner.wilson_interval(0, 0) is None
    assert runner._primary_score({"xmage-ai": {"score": 0.4}, "anchor:a": {"score": 0.9}}) == 0.4
    assert runner._primary_score({"anchor:a": {"score": 0.5}, "anchor:b": {"score": 0.7}}) == pytest.approx(0.6)
    assert runner._primary_score({"xmage-ai": {"score": 0.4}, "xmage-ai:kinnan": {"score": 0.2},
                                  "anchor:a": {"score": 0.9}}) == pytest.approx(0.3)


class _FakeServices:
    def __init__(self):
        self.targets = {}

    def prepare(self, targets):
        self.targets = dict(targets)

    def close(self):
        pass


class _FakeCollector:
    """Plays no games: writes trajectory stubs where the learner service would."""
    fail_next = 0
    resets = 0
    evaluations = 0
    last_config = None
    configs = []

    def __init__(self, script, root, services):
        self.services, self.restarts = services, 0

    def run(self, config_file, timeout):
        config = yaml.safe_load(open(config_file, encoding="utf-8"))
        assert config["server"]["alternate_seats"] is True and timeout > 0
        _FakeCollector.last_config = config_file.name
        _FakeCollector.configs.append((config_file.name, config))
        if _FakeCollector.fail_next:
            _FakeCollector.fail_next -= 1
            raise RuntimeError("engine batch failed: simulated crash")
        if config_file.name.startswith("eval-"):
            _FakeCollector.evaluations += 1
            return (config["training"]["games"], config["training"]["games"], 0, 0, 0)
        games = config["training"]["games"]
        for _, (checkpoint, episodes, _, _) in self.services.targets.items():
            if episodes is None:
                continue
            episodes.mkdir(parents=True, exist_ok=True)
            for g in range(games):
                (episodes / f"episode_{g}.jsonl").write_text(json.dumps(
                    {"terminal": g != 0, "truncated": g == 0, "steps": [{"reward": 0 if g == 0 else 1}]}))
            break
        return (games - 1, games, 1, 0, 0)

    def reset(self, reason):
        _FakeCollector.resets += 1
        self.restarts += 1

    def close(self):
        pass


def _fake_train(current, shards, output, training, device, reference=None):
    model, critic, catalog, data = load_checkpoint(current)
    save_checkpoint(output, model, critic, catalog, updates=data["updates"] + 1)
    return {"decisions": len(shards)}


def _mock_results(monkeypatch, scores):
    """Service outcome counters: evaluation batches take the next scripted score."""
    def results(ports):
        win = int(100 * next(scores)) if _FakeCollector.last_config.startswith("eval-") else 50
        return {"win": win, "loss": 100 - win, "draw": 0, "truncated": 0, "aborted": 0}
    monkeypatch.setattr(runner, "_results", results)


def test_runner_retries_resumes_prunes_and_keeps_the_best(components, tmp_path, monkeypatch):
    model, critic, catalog = components
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, model, critic, catalog, policy_version="initial")
    script = tmp_path / "engine.bat"
    script.write_text("rem fake")
    decks = {}
    for side in ("a", "b"):
        decks[side] = tmp_path / f"{side}.dck"
        decks[side].write_text("4 Forest")
    game = tmp_path / "game.yml"
    game.write_text(yaml.safe_dump({
        "player_a": {"deckPath": str(decks["a"]), "type": "policy"},
        "player_b": {"deckPath": str(decks["b"]), "type": "policy"},
        "training": {"games": 4, "threads": 2, "max_turns": 50, "max_minutes": 5}}))
    policy = tmp_path / "policy.yml"
    policy.write_text(yaml.safe_dump({"training": {"epochs": 1}}))
    monkeypatch.setattr(runner, "_Collector", _FakeCollector)
    monkeypatch.setattr(runner, "_ServicePool", lambda *args: _FakeServices())
    monkeypatch.setattr(runner, "train_from_files", _fake_train)
    scores = iter([0.30, 0.55, 0.45, 0.70])
    _mock_results(monkeypatch, scores)
    output = tmp_path / "run"
    common = dict(games=4, eval_every=1, eval_games=8, eval_ai_skill=6, keep_every=2, snapshot_every=5,
                  min_free_gb=0.0)

    _FakeCollector.fail_next = 1  # the first generation's batch crashes once
    runner.run_selfplay(initial, game, policy, script, output, iterations=2, **common)
    assert _FakeCollector.resets == 1
    state = json.loads((output / "run-state.json").read_text())
    assert state["next_generation"] == 2 and state["best"]["generation"] == 1

    with pytest.raises(ValueError, match="--resume"):
        runner.run_selfplay(initial, game, policy, script, output, iterations=4, **common)
    (output / "generation_0002").mkdir()  # left behind by an interrupted attempt
    runner.run_selfplay(initial, game, policy, script, output, iterations=4, resume=True, **common)

    state = json.loads((output / "run-state.json").read_text())
    assert state["next_generation"] == 4 and state["best"]["generation"] == 3
    assert state["best"]["score"] == pytest.approx(0.70)
    assert list(output.glob("generation_0002.incomplete-*"))
    # Episodes are deleted after training; non-milestone checkpoints are pruned.
    assert not list(output.glob("generation_*/episodes"))
    assert (output / "generation_0000" / "policy.pt").exists()
    assert not (output / "generation_0001" / "policy.pt").exists()
    assert (output / "generation_0002" / "policy.pt").exists()
    assert (output / "generation_0003" / "policy.pt").exists()
    best = torch.load(output / "best.pt", weights_only=True)
    assert best["optimizer_state_dict"] is None
    assert (output / "snapshots" / "generation_0000.pt").exists()
    metrics = json.loads((output / "generation_0003" / "metrics.json").read_text())
    assert metrics["games"] == {"terminal": 3, "requested": 4, "truncated": 1, "errors": 0, "timeouts": 0}
    assert metrics["truncation_fraction"] == 0.25 and "xmage-ai" in metrics["evaluation"]
    assert len((output / "evaluations.jsonl").read_text().splitlines()) == 4


def test_gauntlet_generations_pin_the_learner_deck(components, tmp_path, monkeypatch):
    model, critic, catalog = components
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, model, critic, catalog, policy_version="initial")
    script = tmp_path / "engine.bat"
    script.write_text("rem fake")
    learner, gauntlet = tmp_path / "learner.dck", tmp_path / "kinnan.dck"
    for deck in (learner, gauntlet):
        deck.write_text("1 Forest")
    game = tmp_path / "game.yml"
    game.write_text(yaml.safe_dump({
        "game_mode": "commander",
        "player_a": {"deckPath": str(learner), "type": "policy"},
        "player_b": {"deckPath": str(learner), "type": "policy"},
        "training": {"games": 4, "threads": 2, "max_turns": 50, "max_minutes": 5}}))
    policy = tmp_path / "policy.yml"
    policy.write_text(yaml.safe_dump({"training": {"epochs": 1}}))
    monkeypatch.setattr(runner, "_Collector", _FakeCollector)
    monkeypatch.setattr(runner, "_ServicePool", lambda *args: _FakeServices())
    trained = []
    monkeypatch.setattr(runner, "train_from_files",
                        lambda current, shards, *rest, **options: trained.append(list(shards)) or _fake_train(current, shards, *rest, **options))
    _mock_results(monkeypatch, iter([0.25, 0.75]))
    _FakeCollector.configs, _FakeCollector.fail_next = [], 0
    output = tmp_path / "run"
    runner.run_selfplay(initial, game, policy, script, output, iterations=1, games=8, eval_every=1, eval_games=4,
                        eval_ai_skill=5, min_free_gb=0.0, gauntlet=[gauntlet], gauntlet_fraction=0.25,
                        gauntlet_ai_skill=2, gauntlet_ai_think=1, eval_greedy=True)
    configs = dict(_FakeCollector.configs)
    selfplay, versus = configs["game.yml"], configs["gauntlet.yml"]
    assert selfplay["training"]["games"] == 6 and versus["training"]["games"] == 2
    assert selfplay["player_b"]["type"] == "policy" and selfplay["server"]["sample"] is True
    # The learner keeps its own deck in every seat; the gauntlet deck is XMage's AI.
    assert versus["server"]["learner_keeps_deck"] is True
    assert versus["player_a"]["deckPath"] == str(learner.resolve())
    assert versus["player_b"] == {**versus["player_b"], "type": "xmage-ai", "deckPath": str(gauntlet.resolve())}
    assert versus["opponent_ai"] == {"skill": 2, "think_seconds": 1} and versus["game_mode"] == "commander"
    # Both kinds of trajectories reach one PPO update.
    assert sorted(path.parent.name for path in trained[0]) == ["gauntlet"] * 2 + ["selfplay"] * 6
    metrics = json.loads((output / "generation_0000" / "metrics.json").read_text())
    assert set(metrics["games_by_kind"]) == {"selfplay", "gauntlet:kinnan"}
    assert metrics["games_by_kind"]["gauntlet:kinnan"]["learner_results"]["win"] == 50
    # Evaluation covers the mirror and each gauntlet deck against XMage's AI, greedily.
    evaluations = {name: config for name, config in _FakeCollector.configs if name.startswith("eval-")}
    assert set(evaluations) == {"eval-xmage-ai.yml", "eval-xmage-ai-kinnan.yml"}
    assert evaluations["eval-xmage-ai-kinnan.yml"]["player_b"]["deckPath"] == str(gauntlet.resolve())
    assert all(config["server"]["sample"] is False and config["opponent_ai"] == {"skill": 5}
               for config in evaluations.values())
    assert set(metrics["evaluation"]) == {"xmage-ai", "xmage-ai:kinnan"}
    state = json.loads((output / "run-state.json").read_text())
    assert state["best"]["score"] == pytest.approx(0.5)  # mean over the AI benchmarks
