"""Hand-prediction evaluation and decision traces run on recorded trajectories."""

import json

from magezero.checkpoint import save_checkpoint
from magezero.diagnostics import evaluate_predictions, trace_decision
from magezero.service import PolicyService

from helpers import observation, recorded_episode


def test_prediction_report_and_decision_trace(components, tmp_path):
    model, critic, catalog = components
    checkpoint = tmp_path / "policy.pt"
    version = save_checkpoint(checkpoint, model, critic, catalog)
    service = PolicyService(model, catalog, version)
    steps = [observation(model.config, t, game_id="traced") for t in range(4)]
    episode = recorded_episode(service, steps, version)
    path = tmp_path / "episode.jsonl"
    path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    report = evaluate_predictions(checkpoint, [path])
    assert report["states"] == 4 and report["episodes"] == 1
    assert report["choice_states"] == 4 and report["joint_nll"] >= 0

    trace = trace_decision(checkpoint, path, tmp_path / "trace.json", step_index=2)
    assert json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))["step"] == 2
    # Every traced module reported, including the recurrent one that returns a tuple.
    assert {"entity_attention", "decision_memory", "belief_head", "action_scorer"} <= set(trace["stages"])
    assert len(trace["actions"]) == 2 and abs(sum(a["probability"] for a in trace["actions"]) - 1) < 1e-5
    assert len(trace["first_layer_pool_attention"]) == len(steps[2]["entities"]) + 1
