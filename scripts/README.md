# Scripts

Each script documents its arguments with `--help`. They import the installed `magezero` package (`pip install -e .`).

| Script | Purpose |
|---|---|
| `annotate_policy_catalog.py` | Attach reviewed roles from `configs/card-roles.json` to an exported catalog (catalog format 2); fails on any unreviewed card |
| `commander_roles.py` | The reviewed role annotations of the Winota and Kinnan pool, merged into `configs/card-roles.json` |
| `policy_smoke.py` | Synthetic end-to-end check without XMage: catalog, checkpoint, service games, PPO, reload, forward timing |
| `validate_xmage_policy.py` | Play one synthetic-card XMage game over HTTP with the built adapter, then train on it |
| `final-run.sh`, `final_run.py` | The final training run, and its supervisor: crash restarts with `--resume`, keep-awake, `status.log` ([Long runs](../docs/long-run.md#the-commander-run)) |
| `final-evaluate.sh` | The acceptance evaluation of the final run's candidate checkpoints ([Playing the agent](../docs/playing-the-agent.md#how-strong-it-is)) |
| `teacher_stats.py` | Label mix and size of a `mz teacher-collect` output |
| `evaluate_policy_predictions.py` | Opponent-hand prediction quality against the statistical prior, and an optional decision trace |
| `compare_policy_behavior.py` | Action preferences of two checkpoints replayed on identical recorded histories |
| `benchmark_statistical_roles.py` | Matched forward-pass timing of the learned and statistical belief modes |
| `benchmark_policy_learner.py` | CPU versus CUDA timing of one PPO epoch on recorded trajectories |

The `.sh` scripts run under bash; on Windows, use Git Bash. They pick the `.venv` interpreter and the engine launcher for the platform.
