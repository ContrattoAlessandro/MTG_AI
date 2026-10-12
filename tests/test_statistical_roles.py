import copy
import itertools
import json
import math

import pytest
import torch

from magezero.policy.belief import constrained_hand_belief
from magezero.policy.catalog import CardCatalog
from magezero.policy.checkpoint import load_checkpoint, save_checkpoint
from magezero.policy.learner import PPOConfig, ppo_update
from magezero.policy.model import EntityBeliefPolicy, ModelConfig, PrivilegedCritic
from magezero.policy.roles import ROLE_NAMES
from magezero.policy.schema import collate_observations
from magezero.policy.service import PolicyService, create_app
from magezero.policy.statistics import log_choose, role_statistics, statistical_hand_belief
from test_entity_policy import observation


def components(**options):
    catalog = CardCatalog.from_dict({"format_version": 2, "role_names": list(ROLE_NAMES), "cards": [
        {"id": i, "key": f"card-{i}", "name": f"Card {i}", "features": [0.] * 32, "roles": roles}
        for i, roles in enumerate((["land_access"], ["counterspell", "spot_removal"], ["spot_removal"]), 1)]})
    config = ModelConfig(num_cards=catalog.num_cards, catalog_digest=catalog.digest,
                         width=32, entity_layers=1, max_hand_size=6, **options)
    return EntityBeliefPolicy(config, catalog.tensor(), catalog.role_tensor()).eval(), PrivilegedCritic(config), catalog


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_closed_form_matches_physical_enumeration_and_old_dp(device):
    lower = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], device=device)
    upper = torch.tensor([[3, 2, 1, 0], [0, 0, 0, 0]], device=device)
    size = torch.tensor([3, 0], device=device)
    prior = statistical_hand_belief(lower, upper, size, 5)
    pool = [0, 0, 1, 1, 2]
    hands = list(itertools.combinations(range(len(pool)), 2))
    expected = torch.zeros(4, 5, device=device)
    for hand in hands:
        counts = [1, 0, 0, 0]
        for index in hand:
            counts[pool[index]] += 1
        for kind, count in enumerate(counts):
            expected[kind, count] += 1 / len(hands)
    torch.testing.assert_close(prior.probabilities[0], expected)
    torch.testing.assert_close(prior.probabilities[1, :, 0], torch.ones(4, device=device))
    torch.testing.assert_close(prior.expected_counts.sum(-1), size.float())
    # The previous constrained count DP gives the same physical distribution.
    c = torch.arange(5, device=device)[None, None]
    scores = log_choose((upper - lower).unsqueeze(-1), c - lower.unsqueeze(-1)).float()
    scores = scores.nan_to_num(neginf=-1e9)
    dp = constrained_hand_belief(scores, lower, upper, size)
    torch.testing.assert_close(prior.probabilities, dp.probabilities, atol=2e-6, rtol=1e-5)
    labels = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]], device=device)
    torch.testing.assert_close(prior.nll(labels), dp.nll(labels), atol=2e-6, rtol=1e-5)


def test_overlapping_role_presence_is_union_not_sum_of_card_presences():
    model, _, catalog = components()
    lower, upper, size = torch.zeros(1, 3, dtype=torch.long), torch.tensor([[2, 2, 2]]), torch.tensor([3])
    membership = catalog.role_tensor()[torch.tensor([[1, 2, 3]])]
    presence, expected = role_statistics(lower, upper, size, membership)
    assert presence[0, ROLE_NAMES.index("land_access")].item() == pytest.approx(.8)
    assert presence[0, ROLE_NAMES.index("counterspell")].item() == pytest.approx(.8)
    assert presence[0, ROLE_NAMES.index("spot_removal")].item() == pytest.approx(1.)
    assert expected[0, ROLE_NAMES.index("spot_removal")].item() == pytest.approx(2.)
    assert presence.sum() > 1  # independent role features, never a softmax
    lower[0, 0] = 1
    known, _ = role_statistics(lower, upper, size, membership)
    assert known[0, ROLE_NAMES.index("land_access")].item() == 1
    zero, _ = role_statistics(lower * 0, upper * 0, size * 0, membership)
    assert zero.count_nonzero() == 0


def test_deck_reordering_padding_and_memory_do_not_change_statistical_probabilities():
    model, _, _ = components()
    obs = observation(model.config)
    reordered = copy.deepcopy(obs)
    reordered["opponent_deck"].reverse()
    shorter = copy.deepcopy(obs)
    shorter["opponent_deck"] = shorter["opponent_deck"][:2]
    shorter["opponent_hand_size"] = 2
    with torch.no_grad():
        batch = collate_observations([obs], model.config)
        first = model(batch)
        remembered = model(batch, first.memory)
        reordered_out = model(collate_observations([reordered], model.config))
        mixed = model(collate_observations([obs, shorter], model.config))
        alone = model(collate_observations([shorter], model.config))
    torch.testing.assert_close(first.role_presence, remembered.role_presence)
    assert not torch.allclose(first.logits, remembered.logits)
    torch.testing.assert_close(first.logits, reordered_out.logits)
    torch.testing.assert_close(first.role_presence, reordered_out.role_presence)
    torch.testing.assert_close(mixed.role_presence[1], alone.role_presence[0])
    torch.testing.assert_close(mixed.logits[1], alone.logits[0], atol=1e-6, rtol=1e-5)
    assert not hasattr(model, "belief_head")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_statistical_service_ppo_and_checkpoint(device, tmp_path):
    model, critic, catalog = components()
    model.to(device); critic.to(device)
    service = PolicyService(model, catalog, "stat-test", tmp_path / "episodes")
    client = create_app(service).test_client()
    assert client.get("/healthz").json["architecture"] == "entity_statistical_roles_recurrent_policy"
    for step in range(3):
        payload = {"observation": observation(model.config, step), "sample": True}
        response = client.post("/act", json=payload)
        assert response.status_code == 200
        assert response.json == client.post("/act", json=payload).json
        roles = response.json["belief"]["roles"]
        assert roles["land_access"]["probability_at_least_one"] == pytest.approx(.8)
        assert roles["spot_removal"]["probability_at_least_one"] == pytest.approx(1.)
    finished = client.post("/finish", json={"game_id": "game", "player_id": "player", "terminal": True,
        "result": 1, "labels": [{"opponent_hand_counts": [1, 1, 1]}] * 3})
    episode = json.loads(open(finished.json["episode_file"]).read())
    before = model.role_encoder[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()), lr=1e-3)
    metrics = ppo_update(model, critic, optimizer, [episode], PPOConfig(epochs=1, sequence_length=2), "stat-test")
    assert metrics["decisions"] == 3
    assert metrics["optimizer_steps"] == 1 and metrics["belief_loss_coefficient"] == 0
    assert torch.isfinite(torch.tensor(metrics["belief_nll"]))
    assert not torch.allclose(before, model.role_encoder[0].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, critic, catalog, optimizer=optimizer, updates=1)
    restored, _, restored_catalog, data = load_checkpoint(path)
    assert data["format_version"] == 3 and data["updates"] == 1
    assert restored_catalog.digest == catalog.digest
    batch = collate_observations([observation(model.config)], model.config)
    model.cpu().eval()
    torch.testing.assert_close(restored(batch).logits, model(batch).logits)
    assert restored(batch).role_presence[0, ROLE_NAMES.index("land_access")].item() == pytest.approx(.8)


def test_unknown_roles_missing_catalog_and_legacy_v2_are_explicit(tmp_path):
    _, _, catalog = components()
    invalid = catalog.to_dict()
    invalid["cards"][0]["roles"] = ["magic_good_card"]
    with pytest.raises(ValueError, match="roles"):
        CardCatalog.from_dict(invalid)
    old = catalog.to_dict()
    old["format_version"] = 1
    old.pop("role_names")
    for card in old["cards"]:
        card.pop("roles")
    old = CardCatalog.from_dict(old)
    with pytest.raises(ValueError, match="v2 role catalog"):
        old.role_tensor()
    config = ModelConfig(num_cards=old.num_cards, catalog_digest=old.digest,
        width=32, entity_layers=1, max_hand_size=6, belief_mode="learned")
    model, critic = EntityBeliefPolicy(config, old.tensor()), PrivilegedCritic(config)
    path = tmp_path / "old.pt"
    save_checkpoint(path, model, critic, old)
    raw = torch.load(path, weights_only=True)
    raw["model_config"].pop("belief_mode")  # real pilot files predate this field
    torch.save(raw, path)
    restored, _, _, _ = load_checkpoint(path)
    assert restored.config.belief_mode == "learned" and hasattr(restored, "belief_head")
    batch = collate_observations([observation(config)], config)
    torch.testing.assert_close(restored(batch).logits, model.eval()(batch).logits)


def test_prior_example_large_pool_and_invalid_bounds():
    prior = statistical_hand_belief(torch.tensor([[0, 0]]), torch.tensor([[4, 36]]), torch.tensor([5]), 65)
    assert (1 - prior.probabilities[0, 0, 0]).item() == pytest.approx(1-math.comb(36, 5)/math.comb(40, 5))
    large = statistical_hand_belief(torch.tensor([[0, 0]]), torch.tensor([[10000, 10000]]), torch.tensor([64]), 65)
    torch.testing.assert_close(large.expected_counts.sum(), torch.tensor(64.), atol=1e-4, rtol=1e-6)
    with pytest.raises(ValueError, match="hand size"):
        statistical_hand_belief(torch.tensor([[2, 2]]), torch.tensor([[2, 2]]), torch.tensor([3]), 5)


def test_reviewed_annotation_conversion_preserves_ids_and_rejects_missing_cards():
    from scripts.annotate_policy_catalog import annotate_catalog
    _, _, catalog = components()
    old = catalog.to_dict()
    old["format_version"] = 1
    old.pop("role_names")
    for card in old["cards"]:
        card.pop("roles")
    old = CardCatalog.from_dict(old)
    annotations = {"format_version": 1, "cards": {card["name"]: {
        "roles": card["roles"], "source": "reviewed source", "notes": "reviewed capability"} for card in catalog.cards}}
    updated = annotate_catalog(old, annotations)
    assert updated.digest == catalog.digest and updated.digest != old.digest
    assert [(c["id"], c["key"]) for c in updated.cards] == [(c["id"], c["key"]) for c in old.cards]
    annotations["cards"].pop("Card 2")
    with pytest.raises(ValueError, match="missing reviewed card roles: Card 2"):
        annotate_catalog(old, annotations)


def test_auxiliary_belief_trains_without_changing_actor_outputs(tmp_path):
    torch.manual_seed(5)
    model, critic, catalog = components(auxiliary_belief=True)
    batch = collate_observations([observation(model.config)], model.config)
    with torch.no_grad():
        before = model(batch)
        for parameter in model.auxiliary_belief_head.parameters():
            parameter.add_(torch.randn_like(parameter))
        after = model(batch)
    torch.testing.assert_close(before.logits, after.logits)
    torch.testing.assert_close(before.value, after.value)
    assert after.auxiliary_count_logits.shape == (1, 3, model.config.max_hand_size + 1)
    service = PolicyService(model, catalog, "aux", tmp_path)
    steps = []
    for step in range(3):
        obs = observation(model.config, step)
        response = service.act({"observation": obs, "sample": True})
        steps.append({"observation": obs, "action_id": response["action_id"],
            "log_probability": response["log_probability"], "reward": int(step == 2),
            "labels": {"opponent_hand_counts": [1, 1, 1]}})
    head = model.auxiliary_belief_head[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(critic.parameters()), lr=1e-3)
    metrics = ppo_update(model, critic, optimizer, [{"schema_version": 2, "policy_version": "aux",
        "terminal": True, "steps": steps}], PPOConfig(epochs=1, belief_coefficient=0.1), "aux")
    assert metrics["belief_loss_coefficient"] == 0.1 and math.isfinite(metrics["belief_nll"])
    assert not torch.allclose(head, model.auxiliary_belief_head[0].weight)
    save_checkpoint(tmp_path / "aux.pt", model, critic, catalog, policy_version="aux")
    restored, _, _, _ = load_checkpoint(tmp_path / "aux.pt")
    assert restored.config.auxiliary_belief
