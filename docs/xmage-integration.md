# XMage policy adapter

The Python policy uses a new entity/history protocol. The original XMage binary sends sparse features and requests MCTS priors, so disabling a search budget in the old YAML is insufficient. This adapter adds a direct `PolicyPlayer`, encoder/watcher, standalone game runner, and catalog exporter.

The reviewed engine is [WillWroble/mage](https://github.com/WillWroble/mage), commit `cb7e9c6fe2dcbe2f650a77dd98a87771e730f22f`. Source is maintained in `integration/xmage/src` because XMage is a separate repository. `install.py` verifies the commit and copies the new classes/launchers into that checkout. It does not alter the original MCTS implementation. Human play uses a separate server configuration (below), not a client change.

## Build a distribution for real decks

Use a complete checkout of the pinned fork, JDK/Maven, and the fork's existing build requirements. This workspace already has the complete `.mz_engine` checkout, installed source overlay, and rebuilt distribution under `xmage/`. All 35 modules built successfully; the final archive is `.mz_engine/Mage.MageZero/target/mage-magezero.zip`. For rebuilding or another machine:

```powershell
# Only for a sparse checkout: fetch the remaining source modules.
# git -C .mz_engine sparse-checkout disable

# Install the reviewed overlay and launcher sources.
.venv/Scripts/python.exe integration/xmage/install.py .mz_engine

# Build the engine distribution and its dependencies.
mvn -B -ntp -f .mz_engine/pom.xml -pl Mage.MageZero -am '-Dmaven.test.skip=true' package
```

For a separate checkout, clone the fork, check out the pinned commit, then pass its path to `install.py`. The resulting `Mage.MageZero/target` distribution ZIP contains `lib/` and `mz-policy.bat`/`.sh`; extract it to `xmage/`. Check the actual ZIP filename produced by Maven. This runner executes `org.mage.magezero.PolicyGames`, not the old generator. `PolicyGames game.yml` plays one batch and exits; `PolicyGames --serve` (used by `mz selfplay`) boots once and plays one batch per `game.yml` path read from stdin, replying with a single `MZ-POLICY RESULT <terminal> <requested>` or `MZ-POLICY ERROR <message>` line; EOF on stdin ends it. An optional `server.ports` / `server.opponent_ports` list in the game YAML spreads games across policy-service shards (game g uses entry g mod n). The launchers assume the distribution's `lib/*` classpath and an 8 GB Java heap, leaving RAM for the Python learner.

## Export the immutable card catalog

Place intended constructed decks in `xmage/decks/`. Export the complete deck pool before initializing a policy. IDs must not change between collection and training or between league opponents.

```powershell
Push-Location xmage
java -Xmx8G '-Dh2.bindAddress=127.0.0.1' -cp 'lib/*' org.mage.magezero.PolicyCatalogExporter ../artifacts/cards.json decks/UWTempo.dck decks/Standard-MonoB.dck
Pop-Location

# Add reviewed roles for every exported card before using the new default.
# Extend configs/card-roles.json for pools beyond its reviewed entries.
.venv/Scripts/python.exe scripts/annotate_policy_catalog.py --catalog artifacts/cards.json --annotations configs/card-roles.json --output artifacts/cards-with-roles.json
.venv/Scripts/mz.exe init --catalog artifacts/cards-with-roles.json --output artifacts/initial.pt
```

The exporter creates sorted name-based IDs and coarse card semantics. Include all intended decks and sideboards. Extra faces, token definitions, and generated card pools require catalog/adapter extensions. Do not append cards to an existing checkpoint catalog or silently regenerate different IDs. For a changed pool, initialize a fresh policy or implement an explicit embedding/optimizer migration. XMage's H2 database needs local socket access; the launcher and exporter command bind it to localhost.

The bundled role annotations cover the validated simplegreen pool and selected example cards; the UWTempo/MonoB example pool requires additional reviews. Missing annotations fail explicitly. The Java client already reads catalog IDs/names independently of extra metadata, so the statistical-role update uses the existing rebuilt distribution without a Java rebuild. See [the current implementation](statistical-roles-implementation.md).

## Collect and train

Edit `configs/policy-game.yml` to use the intended deck files. Both players must be `type: policy` with `hiddenInfo.see_opponent_hand: false`. The known deck list is public; the current hand and library order remain private. Manual mana payment is disabled: XMage handles payment using the base player's automatic tapping logic. Start with four Java game threads; the service serializes inference and retains at most 64 player sessions.

```powershell
.venv/Scripts/mz.exe selfplay --checkpoint artifacts/initial.pt --engine-script xmage/mz-policy.bat --iterations 1 --games 32 --output artifacts/experiment-001 --actor-device cpu --device cpu
```

Each generation starts a fixed policy service on port 50052, optionally a historical opponent on 50053, runs games, stops actors, validates fresh completed trajectories, performs PPO, and saves a versioned checkpoint and metrics. Deck/config paths are resolved before launching Java. Occupied ports, incompatible catalogs, stale trajectories, failed engines, and missing terminal data fail explicitly. Output folders must be fresh. Historical opponents use the same catalog and are not trained from their own behavior records.

The initial actor is random. This command is a training experiment, not a trained strong opponent. Sampling remains enabled during self-play. `serve` also supports inspection requests with `sample=false`, but those are excluded from PPO data.

## Validation that ran here

The core Maven build succeeded, and all new Java classes compiled against it. The HTTP integration test played a **real XMage rules game with synthetic free creatures and lands**, using a 12-card deck, six life, and three starting cards. Two policy players completed it, generated two terminal trajectories with 316 decisions, and trained a CUDA checkpoint. This caught engine API, watcher-copy, batched-event, mulligan, action application, privacy-schema, and transport issues.

The full distribution then built successfully, its database populated 563 sets and 87,707 card printings, and the catalog exporter produced 10 card kinds for the fork's `simplegreen` deck. Two normal 60-card games completed through `mz selfplay` and `xmage/mz-policy.bat`. Four terminal player trajectories with 1,172 decisions passed behavior-policy/label validation and trained a new checkpoint on CUDA. This exercises actual deck import, real cards, mana payment, combat, engine collection, service lifecycle, and learner integration. It is not a win-rate or universal card-coverage test.

The resulting checkpoint is `artifacts/real-card-selfplay/generation_0000/policy.pt`. To repeat this small real-card generation with the installed distribution and existing initial checkpoint, use a fresh output directory:

```powershell
.venv/Scripts/mz.exe selfplay --checkpoint artifacts/real-card-initial.pt --game configs/real-card-smoke.yml --config configs/policy-smoke.yml --engine-script xmage/mz-policy.bat --output artifacts/real-card-repeat --iterations 1 --games 2 --actor-device cpu --device cuda
```

To recreate its catalog/checkpoint on another machine, copy `.mz_engine/Mage.MageZero/decks/simplegreen.dck` into `xmage/decks/`, export it with the catalog command above, and use `mz init`. The normal training preset remains `configs/policy.yml`; the smoke preset uses one epoch to shorten validation.

Reproduce the test using a fresh output directory:

```powershell
# If needed, prepare the small dependency/core build in the workspace cache.
mvn -B -ntp -f .mz_engine/Mage/pom.xml '-Dmaven.repo.local=C:/Users/alexa/Desktop/MTG_AI/artifacts/maven' -DskipTests compile
mvn -N -B -ntp -f .mz_engine/Mage/pom.xml '-Dmaven.repo.local=C:/Users/alexa/Desktop/MTG_AI/artifacts/maven' dependency:get '-Dartifact=org.yaml:snakeyaml:2.2' -Dtransitive=false

.venv/Scripts/python.exe scripts/policy_smoke.py --output artifacts/smoke-new --device cpu
.venv/Scripts/python.exe scripts/validate_xmage_policy.py --checkpoint artifacts/smoke-new/initial.pt --output artifacts/java-smoke-new
```

For another workspace, replace the absolute Maven cache argument and retain its `artifacts/maven` location expected by the validation script. The small test also needs the sparse modules `Mage`, `Mage.Common`, `Mage.MageZero`, `Mage.Server.Plugins/Mage.Player.AI`, and `Mage.Server.Plugins/Mage.Game.TwoPlayerDuel`. Localhost access is required. Logs, episodes, metrics, and checkpoints are under the supplied artifact folder.

## Play against the policy

`mz play` starts a policy service for a checkpoint, then the fork's own XMage server from `xmage/lib` with [`integration/xmage/server-config.xml`](../integration/xmage/server-config.xml). That configuration maps the **"Computer - monte carlo"** opponent type to `PolicyPlayer`. Stock XMage clients already list that type, so no client rebuild is needed (upstream MageZero used the same slot). The server listens on `localhost:17171` only.

```powershell
.venv/Scripts/mz.exe play --checkpoint artifacts/final-run/best.pt
# --greedy plays the most likely action instead of sampling; --device cuda is optional
```

1. Open an XMage client whose version matches the server (1.4.58), and connect to `localhost`, port 17171.
2. Create a Two Player Duel table and add a **Computer - monte carlo** player.
3. Give the policy a deck from its card pool, and play a deck from the same pool yourself. Like self-play, the policy reads its opponent's announced deck list at game start, and every card in that list must be in the checkpoint's catalog. Cards outside the catalog fail the game.

Server games are inference only: the policy sends no privileged labels, nothing is recorded for training, and the policy service runs without an episode directory. `PolicyPlayer` reads its endpoint from `-Dmagezero.policy.endpoint` (or `MAGEZERO_POLICY_ENDPOINT`) and initializes itself in `beginTurn`, which `GameImpl.init` calls for every player before shuffling and opening hands. Each new game of a match re-initializes the policy's memory. Player copies share the live session, so XMage state restores (failed payments, a human's undo) keep the policy's memory. Checked on 12 October 2026 by starting the server (4 player types loaded) and by two non-test-mode games built through the server's three-argument constructor against XMage's AI, with zero engine errors. No interactive client session was tested here.

## Coverage and remaining limits

Policy callbacks cover priority, ordinary targets, use/yes-no, mulligans, named choices, amounts, modes, attacks, blocks, trigger/replacement ordering, spell variants, piles, and target-amount allocation. These decisions use a network-selected candidate and no strategic game-tree search. Base XMage helpers still handle mana payment and some specialized rule callbacks. Rule-engine simulations used to establish playability remain necessary.

Sequential combat covers ordinary blocks and creatures that can block additional attackers: such a blocker chooses again until it stops or reaches its limit. Block candidates show how many blockers the attacker already has and its minimum (menace 2). XMage enforces the minimum after declaration by removing an AI's blockers that leave an attacker under it, so a lone block on a menace creature is legal to choose but has no effect. Banding, compelled combat, and complete attack/block configurations are not covered, and multi-blocking has not been exercised with real cards yet. Pile candidates carry aggregate contents and point to their highest mana-value card. Specialized rule callbacks and amount choices above 2,048 candidates need extensions. This is not a claim of complete support for every XMage card.

Belief bounds assume an announced main deck and track public locations of its original cards. Public returns to hand are remembered conservatively. A complete knowledge system for reveals, stolen cards, face-down exile permissions, generated/copy cards, and transformed names is still required for arbitrary real-deck play. Off-deck hidden hand labels fail rather than expanding actor candidates with private information. Test the intended deck pool for these mechanics before large runs.

Aborted and turn/time-limited games produce no PPO data. Monitor their rate, since excluding them can bias learning. The standalone runner supports constructed two-player normal games; draft, Commander, GUI play, and the old `.mz` export format are outside this adapter. The complete distribution and simple real-card loop were validated; complex deck pools and competitive playing strength still need dedicated experiments.
