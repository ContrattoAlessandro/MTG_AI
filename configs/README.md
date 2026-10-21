# Configurations

Policy configurations hold a `model:` section (read by `mz init`), a `training:` section (PPO, read by `mz selfplay` and `mz train`) and a `clone:` section (`mz clone`). Game configurations describe an XMage batch: decks, player types, game mode and limits.

| File | Kind | Purpose |
|---|---|---|
| `policy.yml` | policy | The published agent's settings (the default of every `mz` command): statistical-role belief, warm start with a 20-epoch behaviour clone, GAE λ 0.95, potential-based shaping 0.5, KL 0.05 towards the cloned policy |
| `policy-smoke.yml` | policy | The same architecture with a one-epoch learner, for integration checks |
| `commander-mirror.yml` | game | Commander Duel, Winota against Winota, both seats piloted by the policy (self-play, teacher collection and, with `mz evaluate`, the mirror benchmark) |
| `commander-gauntlet.yml` | game | Commander Duel, the policy's Winota against XMage's AI with Kinnan |
| `card-roles.json` | data | Reviewed role annotations for every card of the supported pools, with source and notes |

Command-line options override some game settings: `--engine-threads` replaces `training.threads`, and `mz selfplay` writes the actual batch configurations it plays into each generation folder. The configurations of the earlier pilots are in git history ([docs/history](../docs/history/README.md)).
