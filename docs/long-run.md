# Long training runs

`mz selfplay` runs synchronous generations: collect games with a frozen checkpoint, run one PPO update on them, repeat. It is built for unattended multi-day runs: it resumes after interruptions, restarts a failed engine, bounds disk use and evaluates periodically. This document describes those mechanisms and the run folder they produce. The learning algorithm itself is in [Architecture](architecture.md#learning-and-collection).

## Mechanisms

| Area | Behaviour |
|---|---|
| Learner | `forward_episodes` runs chunk *w* of every trajectory in a minibatch as one forward pass, so `episodes_per_batch` is both the GPU batch width and the optimizer minibatch. PPO targets are computed `target_batch_episodes` (default 64) trajectories at a time. |
| Engine | A failing game does not fail its batch. Each game has a wall-clock watchdog (`training.max_minutes`) and each turn a separate one (`training.max_turn_seconds`). The batch reply counts terminal, requested, truncated, error and timeout games. Games that XMage ended after swallowing a rules error count as errors, never as truncations. |
| Truncated games | Error-free games that reach `training.max_turns` are recorded with `truncated: true` and bootstrapped by the learner instead of being scored as draws. |
| Recovery | A failed or hung batch (the timeout is derived from games, threads and `max_minutes`) kills the JVM, restarts the policy services, discards that attempt's trajectories, and replays the generation, up to `--max-retries` times. `run-state.json` is written atomically after every generation. `--resume` continues from it and moves a half-finished generation folder aside as `generation_NNNN.incomplete-*`. |
| Failure guard | The run stops (resumably) when more than `--max-failure-fraction` (default 25%) of a generation's games error or time out, since training on the survivors would bias the data. |
| Disk | Trajectories are deleted after training unless `--keep-episodes`. The checkpoint a generation trained from is deleted unless it is a `--keep-every` milestone. League snapshots and `best.pt` are separate optimizer-free copies (about 1/3 the size). The run stops cleanly below `--min-free-gb`. |
| Seats | Odd games swap the two controllers between seats, so against historical opponents and benchmarks the learner plays both decks of an asymmetric matchup. The starting player alternates independently. With `server.learner_keeps_deck: true` (set in `configs/commander-mirror.yml`, and always in gauntlet games) the deck moves with the learner, so a single-deck learner never plays another deck. |
| League | Odd generations play a historical snapshot (one is added every `--snapshot-every` generations) sampled by prioritized fictitious self-play; even generations play the current policy. |
| Gauntlet | `--gauntlet DECK ...` adds games in which the learner's deck (player_a) faces a gauntlet deck piloted by XMage's AI (`--gauntlet-ai-skill`; `--gauntlet-ai-think` caps its seconds per decision). They make up a `--gauntlet-fraction` share of every generation (default 0.5), and the decks rotate by generation. Each generation plays the mirror/league batch, then the gauntlet batch, and trains on both in one PPO update. Only the learner's side of a gauntlet game is recorded. League win rates use only the self-play games. |
| Evaluation | Every `--eval-every` generations, and after the last one, the learner plays `--eval-games` games against XMage's AI (`ComputerPlayer7`, `--eval-ai-skill`; 0 disables it), against XMage's AI with each gauntlet deck, and against each `--eval-anchor` checkpoint. These games are never recorded. `--eval-greedy` plays the most likely action instead of sampling. The score is (wins + ½ · (draws + truncated)) / scored games; errors are reported but never scored. The checkpoint with the best mean score against XMage's AI (or, without it, the best mean anchor score) is copied to `best.pt`. |
| Throughput | The engine JVM and the policy services persist across generations; services swap checkpoints through `/reload`. `--engine-threads` (up to 16) sets concurrent games, and `--actor-shards N` runs N service processes per checkpoint. |

## The run folder

| File | Contents |
|---|---|
| `run-state.json` | Next generation, current checkpoint, best evaluation, league and reference paths (used by `--resume`) |
| `run-summary.json` | Progress, best result and restarts, for a quick look |
| `generation_NNNN/metrics.json` | PPO losses, entropy, KL and reference KL, game outcomes (`games`, `games_by_kind`, `failure_fraction`, `truncation_fraction`), timings and throughput, service statistics, and the evaluation report when one ran |
| `generation_NNNN/policy.pt` | The checkpoint trained in this generation (pruned unless kept) |
| `generation_NNNN/*.yml` | The exact game batches played |
| `evaluations.jsonl` | One line per evaluation: generation, checkpoint, score and per-benchmark report with Wilson intervals |
| `best.pt` | The best evaluated checkpoint, without optimizer state |
| `snapshots/`, `league.json` | League snapshots and the learner's record against each |
| `engine.log`, `actor-*.log`, `opponent-*.log` | Engine and policy-service logs |

## The Commander run

The Winota learner with a Kinnan gauntlet, starting from the behaviour-cloned warm start ([Warm start](warm-start.md)):

```bash
mz selfplay --checkpoint artifacts/commander-cloned.pt --reference artifacts/commander-cloned.pt \
    --game configs/commander-mirror.yml --config configs/policy-final.yml --engine-script $ENGINE \
    --output artifacts/commander-run --iterations 1000 --games 64 \
    --gauntlet xmage/decks/kinnan.dck --gauntlet-fraction 0.5 --gauntlet-ai-skill 1 \
    --engine-threads 16 --device cuda --eval-every 25 --eval-games 64 --eval-ai-skill 3 --eval-greedy \
    --keep-every 100 --min-free-gb 30
```

After an interruption, rerun the same command with `--resume`. The gauntlet fraction, AI skills and game counts are starting points to be tuned from a short launch check: watch games per hour, `failure_fraction` and `truncation_fraction` (both should stay below a few percent), and the share of time spent in the learner.

## Caveats

- XMage's AI searches simulated copies of the game. It is the opponent XMage players usually face, but it is not guaranteed to respect hidden information, so it is a practical yardstick rather than a fair-information baseline.
- Without `--eval-greedy`, evaluation samples from the policy as in training.
- `best.pt` is selected on evaluation results, so its score is optimistically biased. Confirm the final choice with a fresh, larger evaluation.
- The truncation bootstrap gives the final recorded decision zero advantage, because no following observation exists. Keep `max_turns` high enough that truncation stays rare.
- Resuming deletes nothing that reached a checkpoint. A partial generation's logs are kept and its trajectories are deleted.

## Learner throughput

Measured on 64 real `simplegreen` trajectories (14,139 decisions), one PPO epoch, sequence length 32, decisions per second:

| Device | `episodes_per_batch` 4 | 16 | 32 |
|---|---:|---:|---:|
| CPU, 4 threads | 271 | 263 | 246 |
| RTX 5070, shared with another training job | 396 | 566 | 607 |

The GPU figures were taken while another process used the same GPU, and repeated timings varied by about 40%; re-measure on an idle GPU when sizing a run. A CPU profile of the one-epoch update attributes 44% of the time to the backward pass and 39% to the Transformer encoder and linear layers; collation takes about 3%. `scripts/benchmark_policy_learner.py` compares devices on one recorded episode.

## Validation

The test suite runs the runner against a simulated engine: a crash and retry, a refused run without `--resume`, resuming over a half-finished generation, pruning, snapshots, `best.pt`, gauntlet batches and evaluation configurations. Engine tests cover the reply formats, a hung batch timing out, and repeated `kill()`. An end-to-end check on the real distribution (the `simplegreen` mirror, 3 generations of 8 games, `max_turns: 10` to force truncations, evaluation every generation against XMage's AI and the initial checkpoint) recorded and trained all 24 truncated games, scored the evaluations, had zero engine errors, and wrote pruning, snapshots, `best.pt` and `run-state.json` as specified. This validates the machinery, not playing strength.
