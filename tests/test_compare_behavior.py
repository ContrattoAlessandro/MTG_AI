from typing import ClassVar

from scripts.compare_policy_behavior import queries


def test_behavior_queries_distinguish_land_choices_and_forced_actions():
    class Catalog:
        cards: ClassVar[list] = [{"id": 1, "roles": ["land_access"], "features": [1, 0]}]

    obs = {
        "entities": [{"id": "forest", "zone": 2, "owner": 0, "card_id": 1}],
        "actions": [
            {"id": "priority-0", "type": "pass", "source": None, "card_id": 0},
            {"id": "priority-1", "type": "activate", "source": "forest", "card_id": 1},
        ],
    }
    assert queries(obs, Catalog())["play_land_when_legal"] == [1]
    obs["actions"] = obs["actions"][:1]
    assert queries(obs, Catalog()) == {}
