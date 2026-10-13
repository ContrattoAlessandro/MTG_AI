"""Observation and trajectory builders shared by the tests."""

from magezero.schema import ENTITY_FEATURES


def observation(config, step=0, game_id="game", player_id="player"):
    """A valid two-entity, two-candidate observation of the three-card test catalog."""
    return {
        "schema_version": 2,
        "catalog_digest": config.catalog_digest,
        "game_id": game_id,
        "player_id": player_id,
        "step": step,
        "entities": [
            {
                "id": "me",
                "card_id": 0,
                "kind": 0,
                "zone": 0,
                "owner": 0,
                "controller": 0,
                "visibility": "public",
                "features": [0.0] * ENTITY_FEATURES,
            },
            {
                "id": "creature",
                "card_id": 1,
                "kind": 1,
                "zone": 1,
                "owner": 0,
                "controller": 0,
                "visibility": "public",
                "features": [0.1] + [0.0] * (ENTITY_FEATURES - 1),
            },
        ],
        "relations": [{"source": "creature", "target": "me", "type": 0}],
        "globals": [step / 10] + [0.0] * 23,
        "events": [{"seq": step, "type": "pass", "actor": 1, "card_id": 0, "features": [0.0] * 8}],
        "actions": [
            {"id": "pass", "type": "pass", "source": None, "target": None, "card_id": 0, "features": [0.0] * 16},
            {
                "id": "attack",
                "type": "attack",
                "source": "creature",
                "target": "me",
                "card_id": 1,
                "features": [1.0] + [0.0] * 15,
            },
        ],
        "opponent_deck": [
            {"card_id": 1, "lower": 0, "upper": 2},
            {"card_id": 2, "lower": 0, "upper": 2},
            {"card_id": 3, "lower": 0, "upper": 2},
        ],
        "opponent_hand_size": 3,
    }


def varied_trajectory(config, length, game_id, events=True):
    """Observations whose entity, relation, event and candidate counts vary from step to step."""
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


def recorded_episode(service, steps, version, *, terminal=True, reward=1):
    """Play `steps` through the service (sampling) and return the on-policy episode."""
    recorded = []
    for obs in steps:
        response = service.act({"observation": obs, "sample": True})
        recorded.append(
            {
                "observation": obs,
                "action_id": response["action_id"],
                "log_probability": response["log_probability"],
                "reward": 0,
                "labels": {"opponent_hand_counts": [1, 1, 1]},
            }
        )
    if terminal:
        recorded[-1]["reward"] = reward
    episode = {"schema_version": 2, "policy_version": version, "terminal": terminal, "steps": recorded}
    if not terminal:
        episode["truncated"] = True
    return episode
