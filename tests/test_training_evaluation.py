import json

import pytest

from scripts.report_cpu_training import summarize_arena
from scripts.compare_policy_behavior import queries


def test_swapped_seat_groups_and_errors_do_not_become_wins(tmp_path):
    games = [{"index": i, "seed": i//2, "learner_starts": i%2 == 0,
              "terminal": True, "result": 1} for i in range(20)]
    path = tmp_path/"results.json"
    path.write_text(json.dumps({"games": games}))
    summary = summarize_arena([path])
    assert summary["complete_seed_pairs"] == 10
    assert summary["pair_score_bootstrap_95"] == [1., 1.]
    assert summary["conclusion"] == "supports improvement on this matchup"
    games[0]["finish_error"] = "failure"
    path.write_text(json.dumps({"games": games}))
    summary = summarize_arena([path])
    assert summary["wins"] == 19 and summary["errors_or_truncations"] == 1
    assert summary["complete_seed_pairs"] == 9
    assert "incomplete/error" in summary["conclusion"]


def test_balanced_results_remain_inconclusive(tmp_path):
    games = [{"seed": i//2, "learner_starts": i%2 == 0, "terminal": True,
              "result": 1 if i%2 == 0 else -1} for i in range(20)]
    path = tmp_path/"results.json"
    path.write_text(json.dumps({"games": games}))
    summary = summarize_arena([path])
    assert summary["win_rate"] == .5
    assert summary["pair_score_bootstrap_95"] == [.5, .5]
    assert summary["conclusion"].startswith("inconclusive")


def test_behavior_queries_distinguish_land_choices_and_forced_actions():
    class Catalog:
        cards = [{"id": 1, "roles": ["land_access"], "features": [1, 0]}]
    obs = {"entities": [{"id": "forest", "zone": 2, "owner": 0, "card_id": 1}],
        "actions": [{"id": "priority-0", "type": "pass", "source": None, "card_id": 0},
                    {"id": "priority-1", "type": "activate", "source": "forest", "card_id": 1}]}
    assert queries(obs, Catalog())["play_land_when_legal"] == [1]
    obs["actions"] = obs["actions"][:1]
    assert queries(obs, Catalog()) == {}
