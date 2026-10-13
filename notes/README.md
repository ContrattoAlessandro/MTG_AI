# Development notes

Working notes kept for the record. They describe the project's state when each was written, and the documentation in [`docs/`](../docs) supersedes them where they differ.

| File | Contents |
|---|---|
| [final-run-plan.md](final-run-plan.md) | Plan from the A5 comparison to the final training run and a playable agent |
| [review-handoff.md](review-handoff.md) | Architecture review findings, the format-4 changes, and measurements |
| [next-session-plan.md](next-session-plan.md) | Earlier session plan, including the engine rebuild recipe |
| [cpu-30min-evaluation-plan.md](cpu-30min-evaluation-plan.md) | Pre-registered evaluation plan of the 30-minute CPU pilot |
| [a5/](a5) | Scripts of the A5 warm-start comparison (clone, PPO arms at equal budget, then rescoring and second runs); results in [docs/warm-start.md](../docs/warm-start.md#results-the-a5-comparison) |
| [history/](history) | Reports of the earlier `simplegreen` pilots and a traced decision, made with earlier checkpoint formats that the current code no longer loads |

## Code names used in older notes

The package was reorganized after these notes were written; the behaviour is unchanged. Older notes use these names:

| In the notes | Now |
|---|---|
| `magezero.policy.<module>`, `src/magezero/policy/` | `magezero.<module>`, `src/magezero/` |
| `learner._prepare_chunks`, `_join`, `_windows`, `_carry`, `_Chunk` | `learner.prepare_chunks`, `join_chunks`, `chunk_windows`, `carry_memory`, `Chunk` |
| `runner._launch_server`, `_await_server`, `_start_server`, `_stop_server`, `_reload_server` | `service_client.ServiceProcess` (with `wait_ready`, `reload`, `stop`) and `start_service` |
| `runner._service_stats`, `_results`, `_ServicePool` | `service_client.service_stats`, `collect_results`, `ServicePool` |
| `runner._game_config`, `_gauntlet_config`, `_batch_timeout` | `engine.game_config`, `gauntlet_config`, `batch_timeout` |
| `runner._strip_optimizer`, `_policy_version` | `checkpoint.strip_optimizer`, `read_policy_version` |
| `runner._evaluate`, `_primary_score`, `_Collector` | `runner.evaluate`, `primary_score`, `Collector` |
| the body of `runner.run_selfplay` | `runner.SelfPlayRun` (`run_selfplay` wraps it) |
| `teacher._TeacherGames`, `_split` | `teacher.TeacherGames`, `split_games` |
| `schema._vector` | `schema.numeric_vector` |
| `scripts/evaluate.py` | `mz evaluate` (`runner.evaluate_checkpoints`) |
| `diagnostics.explain_training_sample`, `mz legacy-*`, the MCTS pipeline | removed (in git history and upstream) |
| `docs/next-session-plan.md`, `docs/review-handoff.md`, `docs/final-run-plan.md` | `notes/` |
| `scripts/a5-comparison.sh`, `scripts/a5-extra-arms.sh`, `scripts/a5-followup.sh` | `notes/a5/` |
| `configs/policy-compare-control.yml`, `policy-final-lr1e4.yml`, the pilot configurations | `configs/experiments/` |
| `configs/policy-smoke.yml`, `policy-statistical-smoke.yml`, `real-card-smoke.yml` | `configs/smoke/` |
