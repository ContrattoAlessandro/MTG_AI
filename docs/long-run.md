# Running a long training job

Implemented 12 October 2026. This covers the machinery for an unattended multi-day `mz selfplay` run: the batched learner, crash recovery, disk retention, truncated games, and periodic evaluation. It does not cover deck-pool coverage (role annotations for every card). Mana payment stays with XMage's auto-tapper by decision of 12 October 2026. Those still have to be settled before a final run.

## What changed

| Area | Behaviour |
|---|---|
| Learner | `forward_episodes` runs chunk *w* of every trajectory in a minibatch as one forward pass. `episodes_per_batch` is now the GPU batch width as well as the optimizer minibatch. Targets are computed `target_batch_episodes` (default 64) trajectories at a time. Files read by `train_from_files` are validated once, not twice. |
| Engine (Java) | A failing game no longer fails its batch. Each game has a wall-clock watchdog (`training.max_minutes`). The batch reply is `RESULT terminal requested truncated errors timeouts`. Games that XMage ended after swallowing a rules error are counted as errors, never as truncations. |
| Truncated games | Error-free games that hit `training.max_turns` are recorded with `truncated: true` and bootstrapped by the learner (see [architecture](architecture.md#learning-and-collection)). |
| Recovery | A failed or hung batch (`engine.run` timeout, derived from games/threads/max_minutes) kills the JVM, restarts the policy services, discards that attempt's episodes, and replays the generation, up to `--max-retries` times. `run-state.json` is written atomically after every generation; `--resume` continues from it and moves any half-finished generation folder aside as `generation_NNNN.incomplete-*`. |
| Failure guard | The run stops (resumable) if more than `--max-failure-fraction` (default 25%) of a generation's games error or time out, since training on the survivors would bias the data. |
| Disk | Episodes are deleted after training unless `--keep-episodes`. The checkpoint a generation trained from is deleted unless it is every `--keep-every`-th one. League snapshots are stored as optimizer-free copies in `snapshots/` (about 1/3 the size). The run stops cleanly below `--min-free-gb`. |
| Seats | Odd games swap the two controllers between seats, so the learner plays both decks against historical opponents and benchmarks. The starting player alternates independently. With `server.learner_keeps_deck: true` (set in `configs/commander-mirror.yml`, and always in gauntlet games) the deck moves with the learner, so a single-deck learner never plays another deck. |
| Gauntlet (13 Oct) | `--gauntlet DECK ...` adds a third kind of game: a `--gauntlet-fraction` share of every generation (default 0.5) puts the learner's deck (player_a) against a gauntlet deck piloted by XMage's AI (`--gauntlet-ai-skill`, default 3; `--gauntlet-ai-think` caps its seconds per decision). Decks rotate by generation. Each generation runs the mirror/league batch, then the gauntlet batch, and trains on both in one PPO update; only the learner's side of a gauntlet game is recorded. Trajectories go to `episodes/selfplay` and `episodes/gauntlet`; league win rates use only the self-play ones. `metrics.json` has `games_by_kind` with each batch's engine counts and the learner's results. |
| Evaluation | Every `--eval-every` generations (and after the last one), the learner plays `--eval-games` games against XMage's built-in AI (`ComputerPlayer7`, `--eval-ai-skill`, 0 disables it) and against each `--eval-anchor` checkpoint. These games are never recorded for training. With a gauntlet, each gauntlet deck is also evaluated against XMage's AI (`xmage-ai:<deck>`), and `--eval-greedy` evaluates with the most likely action instead of sampling. Results go to `evaluations.jsonl` and the generation's `metrics.json`. The best checkpoint by mean score over the XMage-AI benchmarks (or the mean anchor score) is copied to `best.pt` without optimizer state. Score = (wins + ½·(draws + truncated)) / scored games; errors are reported, never scored. |

## Commander Duel: Winota learner, Kinnan gauntlet

```powershell
.venv/Scripts/mz.exe selfplay --checkpoint artifacts/commander-initial.pt --game configs/commander-mirror.yml --config configs/policy-final.yml `
  --engine-script xmage/mz-policy.bat --output artifacts/commander-run --iterations 5000 --games 64 `
  --gauntlet xmage/decks/kinnan.dck --gauntlet-fraction 0.5 --gauntlet-ai-skill 1 `
  --engine-threads 16 --device cuda --eval-every 25 --eval-games 64 --eval-ai-skill 3 --eval-greedy `
  --keep-every 100 --min-free-gb 30
```

The numbers (fraction, skills, games) are a starting point to be set from the launch check in [the plan](next-session-plan.md#task-d-launch-check-then-the-final-run).

## Caveats

- XMage's AI searches simulated copies of the game. It is the usual opponent XMage players face, but it is not guaranteed to respect hidden information, so it is a practical yardstick, not a fair-information baseline.
- Evaluation samples from the policy, as in training; it is not greedy play.
- `best.pt` is selected on evaluation results, so its score is optimistically biased. Confirm a final choice with a fresh, larger evaluation.
- The truncation bootstrap gives the final recorded decision zero advantage, because no following observation is recorded. Keep `max_turns` high enough that truncation stays rare, and watch `truncation_fraction`.
- Resuming deletes nothing that reached a checkpoint. A partial generation's logs are kept and its episodes are deleted.

## Measured learner throughput

64 real simplegreen trajectories from the 30-minute run (14,139 decisions), one PPO epoch, sequence length 32, behaviour-probability check disabled because the episodes come from several checkpoints. The script is in the session scratchpad; raw numbers below.

| Device | episodes_per_batch 4 | 16 | 32 |
|---|---:|---:|---:|
| CPU, 4 threads | 271 /s | 263 /s | 246 /s |
| RTX 5070 (**shared with another training job**) | 396 /s | 566 /s | 607 /s |

For comparison, the earlier per-episode learner measured 56 /s CPU (30-minute run, 1 thread) and 360–450 /s CUDA. The CUDA numbers here were measured while another training process used the same GPU, and repeated timings of the same forward pass varied by about 40%. Re-measure on an idle GPU before sizing the run. In the 64-episode CUDA update, Python-side validation (2.3 s, now done once instead of twice) and collation (5.3 s) took about 8 s of 22 s. The statistics module's per-call input checks add host synchronisations inside each forward pass. Collation and those checks are the next optimisation targets if the learner remains the bottleneck.

### Learner changes after this measurement (12 October 2026, evening)

- The model's forward pass no longer re-validates count bounds (`statistics.py` `_validate`, `torch.all`, `.item()`). `validate_observation` already enforces them, and the role matrix is checked when the model is built. Diagnostics still validate.
- `forward_episodes` takes each row's event count as a CPU tensor kept from collation, so GRU packing no longer copies lengths from the GPU.
- Collation was re-profiled on 64 real trajectories (13,351 decisions): 1.2 s on CPU, about 3% of a 42 s one-epoch CPU update. The update is compute-bound: backward 44%, the transformer encoder and linear layers 39%. The 5.3 s figure above predates these changes and included per-window joins on a contended GPU.
- bf16 autocast was not added. It would put the behaviour-probability check and the first PPO ratios at bf16 precision, so measure it on an idle GPU before adopting it.

None of these were timed on the GPU, because another job was using it.

## Validation

- 39 pytest checks pass. New: multi-episode forward vs separate sequences (uneven lengths, eventless trajectories, carried memory), target batching invariance, truncated recording/bootstrapping/validation, service outcome counters, Wilson interval and selection metric, and a runner test with a simulated engine crash, retry, refused non-resume, resume over a half-finished generation, pruning, snapshots, `best.pt` and metrics.
- Engine tests cover the new reply format, the legacy reply format, batch hang timeout and idempotent `kill()`.
- Real end-to-end check (`artifacts/e2e-long-run-check`): the patched XMage distribution with the simplegreen mirror, 3 generations × 8 games, `max_turns: 10` to force truncations, evaluation against XMage's AI and the initial checkpoint each generation. All 24 training games were recorded as truncated and trained. Evaluation games against XMage's AI completed and were scored. There were zero engine errors, and pruning, snapshots, `best.pt` and `run-state.json` were written as specified. This validates the plumbing, not playing strength.

## Example

```powershell
.venv/Scripts/mz.exe selfplay --checkpoint artifacts/initial.pt --game configs/policy-game.yml --config configs/policy.yml `
  --engine-script xmage/mz-policy.bat --output artifacts/final-run --iterations 5000 --games 64 `
  --engine-threads 16 --actor-shards 2 --device cuda `
  --eval-every 25 --eval-games 128 --eval-ai-skill 6 --eval-anchor artifacts/initial.pt `
  --keep-every 100 --min-free-gb 30
# after any interruption, the same command plus --resume
```

The Java classes were rebuilt with `javac` against `xmage/lib/*` and patched into `xmage/lib/mage-magezero-1.4.58.jar` and `mage-player-ai-1.4.58.jar`. The originals are in `artifacts/jar-backup-20261012/`. Sources are synced into `.mz_engine` with `integration/xmage/install.py`; a full Maven rebuild produces the same classes.
