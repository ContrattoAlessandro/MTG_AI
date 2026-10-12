# MageZero: entity, history, and opponent-belief policy

This local redesign replaces the default MCTS training path with a direct recurrent policy for two-player constructed Magic. It uses attention over visible entities, persistent history, a statistical known-deck hand prior, 20 overlapping card-role presence probabilities, legal-action scoring, and PPO with a training-only privileged critic. The default has no learned belief head or hand-count dynamic programming.

The design draws on AlphaStar, DouZero/DouZero+, PerfectDou, and search-free imperfect-information learning. Read the [architecture and primary-source research](docs/architecture.md), [protocol](docs/policy-protocol.md), and [XMage integration](docs/xmage-integration.md). This is an implemented experimental architecture; no MTG playing-strength improvement or ideal architecture is claimed.

The [30-minute measured pilot](docs/pilot-2026-10-12.md) completed 20 games and five updates. Belief prediction improved modestly, but stronger play was not demonstrated. The [actual decision-to-training example](docs/flow-example.md) traces every model stage and its losses. A matched learner test found CPU faster than CUDA for the current serial implementation; see the report before starting a larger GPU run.

Those pilot artifacts describe the earlier learned-belief model. The [statistical-role implementation and validation](docs/statistical-roles-implementation.md) describes the current default, its reviewed card annotations, a two-game integration check, and matched forward benchmarks. V2 learned-belief checkpoints remain loadable without converting their weights or trajectories.

The subsequent [30-minute CPU training evaluation](docs/cpu-training-2026-10-12.md) completed 96 games and beat its frozen starting policy **149–51 in 200 fresh games**. It learned more active casting and combat on simplegreen; mulligan preferences stayed nearly 50/50. This establishes improvement against that starting policy on one matchup, not general Magic strength or superiority over trained upstream MageZero.

## Local setup and smoke test

Python 3.10+ is required. On this computer a workspace-local `.venv` reuses the installed CUDA PyTorch and contains the package/HTTP server. For a fresh environment, install a PyTorch build appropriate for your device before running the GPU learner.

```powershell
python -m venv .venv --system-site-packages
.venv/Scripts/python.exe -m pip install -e '.[test]'

# Model, collection, PPO, checkpoint reload, and a synthetic inference benchmark.
.venv/Scripts/python.exe scripts/policy_smoke.py --output artifacts/smoke-new --device cpu

# Unit/regression checks, including CUDA when available.
.venv/Scripts/python.exe -m pytest tests -q
```

Use a fresh artifact directory for each smoke run. All 19 regression checks passed on this computer, including statistical probabilities and CPU/CUDA PPO. The complete 35-module XMage distribution built and is installed in `xmage/`. The statistical-role model completed two real `simplegreen` games, producing four trajectories and 1,223 decisions; CPU PPO saved an updated checkpoint.

## Real-deck training

The old XMage release binary is incompatible with the new protocol. This workspace already contains the rebuilt adapter and a working small real-card experiment. For another computer or deck pool, [install the pinned source adapter](docs/xmage-integration.md), export a fixed catalog, and edit `configs/policy-game.yml`.

```powershell
# The bundled annotations cover simplegreen plus a few reviewed example cards.
# Other deck pools must first receive complete reviewed annotations.
.venv/Scripts/python.exe scripts/annotate_policy_catalog.py --catalog artifacts/real-card-catalog.json --annotations configs/card-roles.json --output artifacts/cards-with-roles.json
.venv/Scripts/mz.exe init --catalog artifacts/cards-with-roles.json --output artifacts/initial.pt
.venv/Scripts/mz.exe selfplay --checkpoint artifacts/initial.pt --game configs/real-card-smoke.yml --engine-script xmage/mz-policy.bat --output artifacts/run-001 --iterations 1 --games 32 --actor-device cpu --device cpu
```

Single-observation inference and the current serial learner measured faster on CPU on this machine, so CLI actors and learners default to CPU. CUDA remains available explicitly. The default has width 256, two attention layers, and roughly 4.2 million actor parameters for a small deck pool. Matched synthetic CPU forward medians were 7.29 ms for learned belief versus 4.72 ms for statistical roles; these exclude engine/HTTP overhead and say nothing about win rate.

`mz serve` runs direct inference; `mz train` updates a checkpoint from fresh complete player trajectories. Old sparse-feature datasets and MCTS checkpoints require new collection and training. Hidden opponent hands are excluded from actor inputs. True counts train the privileged critic and supply diagnostic targets. Belief supervision applies only to explicitly selected learned-belief models.

The [integration limits](docs/xmage-integration.md#coverage-and-remaining-limits) matter for complex card pools: generated/stolen cards, complete visibility tracking, grouped combat, specialized callbacks, and timeout bootstrapping need further work. Validation covered synthetic cards and a simple real-card deck, rather than every Magic mechanic.

## Original pipeline

The original code remains available through explicit `legacy-train`, `legacy-batch`, `legacy-play`, `legacy-import`, and `legacy-export` commands. Install `pip install -e '.[legacy]'` for their additional dependencies. The [original README and results](docs/legacy-readme.md) and [legacy setup guide](setup_guide.md) describe that pipeline; its reported results do not apply to this new policy.
