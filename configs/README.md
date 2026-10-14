# Configurations

Policy configurations hold a `model:` section (read by `mz init`), a `training:` section (PPO, read by `mz selfplay` and `mz train`) and optionally a `clone:` section (`mz clone`). Game configurations describe an XMage batch: decks, player types, game mode and limits.

| File | Kind | Purpose |
|---|---|---|
| `policy.yml` | policy | Default model and PPO settings |
| `policy-final.yml` | policy | The Commander run: warm start with behaviour cloning, GAE λ 0.95, potential-based shaping 0.5, KL 0.05 towards the cloned policy |
| `commander-mirror.yml` | game | Commander Duel, Winota against Winota, both seats piloted by the policy (self-play and teacher collection) |
| `commander-gauntlet.yml` | game | Commander Duel, the policy's Winota against XMage's AI with Kinnan |
| `policy-game.yml` | game | Example two-player constructed matchup |
| `card-roles.json` | data | Reviewed role annotations for every card of the supported pools, with source and notes |
| `smoke/` | both | Short configurations for integration checks on the `simplegreen` mirror |
| `experiments/` | both | Configurations of the earlier `simplegreen` pilots (`pilot-game.yml`, `policy-pilot.yml`, `policy-cpu-30min.yml`), which used earlier checkpoint formats |

Command-line options override some game settings: `--engine-threads` replaces `training.threads`, and `mz selfplay` writes the actual batch configurations it plays into each generation folder.
