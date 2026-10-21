# Earlier pilots and validation records

Reports and raw measurements from the development of the current method, before the Commander run. They are kept as a record; the rest of [`docs/`](..) describes the code as it is now and supersedes them where they differ.

All of them were made on MageZero's XMage fork (1.4.58, [WillWroble/mage](https://github.com/WillWroble/mage) commit `cb7e9c6f`) with its `simplegreen` deck, which official XMage releases do not ship, and with earlier checkpoint formats that the current code no longer loads. Their configurations and the scripts that produced the reports are in git history at commit [`228f770`](https://github.com/ContrattoAlessandro/MTG_AI/tree/228f770) (`configs/experiments/`, `scripts/report_cpu_training.py`, `scripts/plot_cpu_training.py`). The package was reorganized after they were written: `magezero.policy.<module>` is now `magezero.<module>`.

| File | Contents |
|---|---|
| [pilot-2026-10-12.md](pilot-2026-10-12.md), [.json](pilot-2026-10-12.json) | First measured training pilot: 20 `simplegreen` mirror games, speed and hand-prediction learning curves |
| [flow-example.md](flow-example.md) | One recorded decision traced from observation through the network to its PPO update |
| [cpu-30min-evaluation-plan.md](cpu-30min-evaluation-plan.md) | Pre-registered evaluation plan of the 30-minute CPU pilot |
| [cpu-training-2026-10-12.md](cpu-training-2026-10-12.md), [.json](cpu-training-2026-10-12.json) | The 30-minute CPU pilot: 149 of 200 held-out games won against the frozen starting policy |
| [validation-2026-10-11.json](validation-2026-10-11.json) | Raw metrics of the first end-to-end validation (synthetic-card XMage game, then `simplegreen` games) |
| [statistical-roles-validation.json](statistical-roles-validation.json) | Measurements taken when the statistical-role belief was introduced ([Statistical roles](../statistical-roles-implementation.md#validation-and-measured-speed)) |
