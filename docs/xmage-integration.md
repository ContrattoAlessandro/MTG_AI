# XMage policy adapter

The Python policy uses a new entity/history protocol. The original XMage binary sends sparse features and requests MCTS priors, so disabling a search budget in the old YAML is insufficient. This adapter adds a direct `PolicyPlayer`, encoder/watcher, standalone game runner, and catalog exporter.

The reviewed engine is [WillWroble/mage](https://github.com/WillWroble/mage), commit `cb7e9c6fe2dcbe2f650a77dd98a87771e730f22f`. Source is maintained in `integration/xmage/src` because XMage is a separate repository. `install.py` verifies the commit, copies the new classes/launchers into that checkout, and applies the engine fixes in `integration/xmage/patches/` (git diffs; already-applied ones are skipped). It does not alter the original MCTS implementation. Human play uses a separate server configuration (below), not a client change.

Engine patches:

| Patch | Why |
|---|---|
| `computer-player.patch` (1) | `ComputerPlayer` auto-payment recursed without end when mana abilities that cost mana (Prismite, Energy Refractor, Urn of Godfire in the Kinnan deck) paid for each other, overflowing the stack inside XMage AI simulations. Payment now gives up below 8 nested mana abilities. |
| `computer-player.patch` (2) | Every XMage AI player in a JVM shares one pool of 5 search threads, so 16 concurrent gauntlet games queued on it and hit their think-time cap on most decisions. The pool size now reads `-Dxmage.ai.simulationThreads` (default 5); `PolicyGames` sets it to the CPU count unless given. The old value was a compile-time constant inlined into `ComputerPlayer6`, so that class is recompiled into `mage-player-ai-ma` too. |
| `mana-options.patch` | `ManaOptions` (the "what can this player afford" estimate behind `getPlayable`) simulated repeatable mana abilities that cost mana an unlimited number of times and kept every combination, with quadratic duplicate checks. On a Kinnan board one call ran for over 7 CPU-minutes, ignoring the game's timeout. Repeats are now capped at 4 and the options at the 64 largest. It only affects affordability estimates; actual payment is unchanged. |
| `ai-activation-cap.patch` | XMage's AI repeated one activated ability for the whole turn (Basalt Monolith's untap with Kinnan, which is a real infinite-mana loop; Freed from the Real), until the per-turn watchdog or the 8,192-event limit killed the game: about 8% of Kinnan games. `ComputerPlayer6` now passes instead of activating the same ability more than 50 times in one turn (`-Dxmage.ai.maxActivationsPerTurn`). Spells are not counted. |
| `player-fast-fail.patch` | `PlayerImpl`'s copy constructor dropped the fast-fail flag, so the setting below reverted on every state copy. |

Rebuilding after a Java change compiles the overlay plus each patched engine class and updates four jars:

```bash
ROOT=/c/Users/alexa/Desktop/MTG_AI; OUT="$TEMP/claude/mzjava"; rm -rf "$OUT"; mkdir -p "$OUT"; cd "$ROOT"
.venv/Scripts/python.exe integration/xmage/install.py .mz_engine
AI=.mz_engine/Mage.Server.Plugins/Mage.Player.AI/src/main/java/mage/player/ai
MA=.mz_engine/Mage.Server.Plugins/Mage.Player.AI.MA/src/mage/player/ai
MO=.mz_engine/Mage/src/main/java/mage/abilities/mana/ManaOptions.java
PI=.mz_engine/Mage/src/main/java/mage/players/PlayerImpl.java
javac -nowarn -encoding UTF-8 -cp "xmage/lib/*" -d "$OUT" $(find integration/xmage/src -name '*.java') $AI/ComputerPlayer.java $MA/ComputerPlayer6.java $MO $PI
cd "$OUT"
jar uf "$ROOT/xmage/lib/mage-player-ai-1.4.58.jar" mage/player/ai/policy/*.class mage/player/ai/ComputerPlayer.class 'mage/player/ai/ComputerPlayer$PickedCard.class'
jar uf "$ROOT/xmage/lib/mage-player-ai-ma-1.4.58.jar" mage/player/ai/ComputerPlayer6*.class
jar uf "$ROOT/xmage/lib/mage-1.4.58.jar" mage/abilities/mana/ManaOptions*.class mage/players/PlayerImpl*.class
jar uf "$ROOT/xmage/lib/mage-magezero-1.4.58.jar" org/mage/magezero/*.class
```

Check that no `java.exe` runs from `xmage/lib` first. Backups of the four jars from before the Commander work are in `artifacts/jar-backup-20261012-pre-commander/`.

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

The exporter creates sorted name-based IDs and coarse card semantics. Include all intended decks and sideboards (a Commander deck's commander is its sideboard). It also adds every other face of the pool's cards (a transformed planeswalker, an MDFC played as its land) and the tokens the cards can create, under the names XMage gives them on the battlefield (`Goblin Token`, `Treasure Token`; catalog key `xmage-token:`). Tokens are found by walking each card's effects; pass `--sources .mz_engine/Mage.Sets/src` to also scan card sources for tokens built inside custom effects (Legion Warboss, Loyal Apprentice, Sokenzan). Token copies of cards use the card's own ID. Generated card pools beyond these (random cards, conjured cards) still need extensions. Do not append cards to an existing checkpoint catalog or silently regenerate different IDs. For a changed pool, initialize a fresh policy or implement an explicit embedding/optimizer migration. XMage's H2 database needs local socket access; the launcher and exporter command bind it to localhost.

The bundled role annotations cover the validated simplegreen pool and selected example cards; the UWTempo/MonoB example pool requires additional reviews. Missing annotations fail explicitly. The Java client already reads catalog IDs/names independently of extra metadata, so the statistical-role update uses the existing rebuilt distribution without a Java rebuild. See [the current implementation](statistical-roles-implementation.md).

## Commander Duel

Implemented 13 October 2026 for the single-deck goal in [the plan](next-session-plan.md): the learner pilots **Winota, Joiner of Forces** (`xmage/decks/winota.dck`) and the gauntlet opponent is **Kinnan, Bonder Prodigy** (`xmage/decks/kinnan.dck`) piloted by XMage's AI.

- `game_mode: commander` creates XMage's `CommanderDuel` and `CommanderDuelMatch` (loaded by name, like the AI): 100-card decks, 40 life, seven cards, London mulligan without a free mulligan, 21 combat damage from one commander loses. It matches the server's "Commander Two Player Duel" table. `normal` is unchanged; every other mode is refused.
- The commander is the deck's sideboard. `GameCommanderImpl` moves it to the command zone; the encoder shows both commanders there (see [the protocol](policy-protocol.md#feature-layout-in-the-java-adapter)).
- `server.learner_keeps_deck: true` moves the learner's deck with it when seats swap, so the learner always plays player_a's deck. `server.sample: false` makes policy players greedy (evaluation only; unsampled games are never recorded). `opponent_ai.think_seconds` caps XMage's AI search per decision (default 3 s per skill level).
- Configs: `configs/commander-mirror.yml` (Winota vs Winota, both policy) and `configs/commander-gauntlet.yml` (Winota policy vs Kinnan XMage AI).
- Deck lists are written as plain text in `decks-src/` and converted with `org.mage.magezero.DeckListConverter list.txt deck.dck` (lines after the first blank line are the sideboard/commander; unknown names fail). Three cards of the user's lists are not in this XMage build and were substituted: Mayor Tong of Chin Village → Ragavan, Nimble Pilferer (Winota); Avatar Sanctuary → Breeding Pool and Tidus, Zanarkand Fayth → Bounding Krasis (Kinnan).
- Catalog: `artifacts/commander-cards.json` (173 kinds: 158 deck cards, 5 other faces, 10 tokens), roles in `configs/card-roles.json` written by `scripts/commander_roles.py`, annotated catalog `artifacts/commander-cards-roles.json`, fresh checkpoint `artifacts/commander-initial.pt` (seed 20261013).

```powershell
Push-Location xmage
java -Xmx8G '-Dh2.bindAddress=127.0.0.1' -cp 'lib/*' org.mage.magezero.PolicyCatalogExporter ../artifacts/commander-cards.json --sources ../.mz_engine/Mage.Sets/src decks/winota.dck decks/kinnan.dck
Pop-Location
.venv/Scripts/python.exe scripts/commander_roles.py --catalog artifacts/commander-cards.json
.venv/Scripts/python.exe scripts/annotate_policy_catalog.py --catalog artifacts/commander-cards.json --annotations configs/card-roles.json --output artifacts/commander-cards-roles.json
.venv/Scripts/mz.exe init --catalog artifacts/commander-cards-roles.json --config configs/policy-final.yml --output artifacts/commander-initial.pt --seed 20261013
```

Engine fixes found by the Commander smoke games (all in `PolicyPlayer` unless noted):

| Symptom | Fix |
|---|---|
| Phyrexian Revoker's "name a card" offered every card in XMage (over 2,048 candidates) | Oversized named choices are restricted to the names in the card catalog. |
| Sanctum Prelate's "choose a number" is 0 to 2³¹ | Amount ranges above the candidate limit offer their smallest 256 values. |
| XMage lists some abilities as playable that then fail during payment, which aborted the game | A refused activation is treated as no action (as XMage's AI does) and is not offered again in that priority window. |
| The failed activation restored a saved state, including the history watcher, so events were sent twice | The event sequence counter and the "already sent" mark are shared by every copy of `PolicyHistory`. |
| `chooseReplacementEffect` parsed effect IDs as numbers | It returns the index in XMage's map order. |
| Infinite mana-payment recursion in XMage AI simulations, a 5-thread AI search pool, and minutes-long affordability estimates (all from the Kinnan deck) | Engine patches above. |
| XMage's AI looped one action (Kinnan untapping a mana rock) for the whole turn; the single 15-minute interrupt then held the entire batch, and the AI clears interrupts while it waits for its search | `PolicyGames` checks every second and keeps interrupting once a game exceeds `training.max_minutes` or one turn exceeds `training.max_turn_seconds` (default 120). Such games count as timeouts and are discarded. |
| "AI can't find good blocker combination" ended games: players run in XMage's test mode, where that check (and errors inside AI simulations) throw instead of skipping the block as in a real game | `PolicyGames`/`PolicyArena` turn off `fastFailInTestMode` for every player (with `player-fast-fail.patch`). |
| A Kinnan turn emitted more than 1,024 history events between two policy decisions, which the service refuses to truncate | `model.max_events` is 8,192 in `configs/policy-final.yml` (a validation limit; no weights depend on it). |

The engine log ends each game with a summary line (`policy game N TERMINAL: turn T; PlayerA (policy) won life 35 poison 0 commander damage 0 ...`), which shows how games end.

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

Aborted and turn/time-limited games produce no PPO data. Monitor their rate, since excluding them can bias learning. The standalone runner supports two-player constructed games and Commander Duel; draft, multiplayer Commander, and the old `.mz` export format are outside this adapter. The complete distribution and simple real-card loop were validated; complex deck pools and competitive playing strength still need dedicated experiments.
