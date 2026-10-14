# XMage integration

The policy plays inside the XMage rules engine through a Java adapter. The original MageZero engine fork sends sparse hashed features and asks for MCTS priors, so it cannot drive this policy. The adapter adds a direct `PolicyPlayer` with its observation encoder and history watcher, a recording `TeacherPlayer`, a batch game runner, and a catalog exporter.

The adapter targets [WillWroble/mage](https://github.com/WillWroble/mage) at commit `cb7e9c6fe2dcbe2f650a77dd98a87771e730f22f`. XMage is a separate repository, so the sources live in `integration/xmage/` and are copied into a checkout of that commit:

| Path | Contents |
|---|---|
| `src/mage/player/ai/policy/` | `PolicyPlayer` (the policy as an XMage player), `PolicyEncoder`, `PolicyHistory`, `PolicyCandidates`, `PolicyClient`, `HandSummary`, `RecordingPlayer`, `WebPlayer` (a person in the browser, `mz web`) |
| `src-ma/mage/player/ai/TeacherPlayer.java` | XMage's Mad AI with every decision recorded over the policy's candidate list (behaviour cloning) |
| `src/org/mage/magezero/` | `PolicyGames` (batch runner), `PolicyArena` (fixed-opponent arena), `PolicyCatalogExporter`, `DeckListConverter` |
| `patches/` | Engine fixes, as git diffs against the pinned commit |
| `mz-policy.sh`, `mz-policy.bat` | Launchers for `PolicyGames`, copied into the distribution |
| `server-config.xml` | XMage server configuration for human play |
| `test/PolicySmoke.java` | A synthetic-card game over HTTP (`scripts/validate_xmage_policy.py`) |

`install.py` verifies the checkout's commit, copies the sources and launchers, and applies the patches (already-applied ones are skipped). It leaves the original MCTS implementation in the fork untouched.

## Engine patches

| Patch | Why |
|---|---|
| `computer-player.patch` (1) | `ComputerPlayer` auto-payment recursed without end when mana abilities that cost mana (Prismite, Energy Refractor and Urn of Godfire in the Kinnan deck) paid for each other, overflowing the stack inside XMage AI simulations. Payment now gives up below 8 nested mana abilities. |
| `computer-player.patch` (2) | Every XMage AI player in a JVM shares one pool of 5 search threads, so 16 concurrent gauntlet games queued on it and hit their think-time cap on most decisions. The pool size now reads `-Dxmage.ai.simulationThreads` (default 5); `PolicyGames` sets it to the CPU count unless given. The old value was a compile-time constant inlined into `ComputerPlayer6`, so that class is recompiled into `mage-player-ai-ma` too. |
| `mana-options.patch` | `ManaOptions` (the "what can this player afford" estimate behind `getPlayable`) simulated repeatable mana abilities that cost mana an unlimited number of times and kept every combination, with quadratic duplicate checks. On a Kinnan board one call ran for over 7 CPU-minutes, ignoring the game's timeout. Repeats are now capped at 4 and the options at the 64 largest. Only affordability estimates change; actual payment does not. |
| `ai-activation-cap.patch` | XMage's AI repeated one activated ability for the whole turn (Basalt Monolith's untap with Kinnan, a real infinite-mana loop; Freed from the Real) until the per-turn watchdog or the 8,192-event limit ended the game, in about 8% of Kinnan games. `ComputerPlayer6` now passes instead of activating the same ability more than 50 times in one turn (`-Dxmage.ai.maxActivationsPerTurn`). Spells are not counted. |
| `player-fast-fail.patch` | `PlayerImpl`'s copy constructor dropped the fast-fail flag, so the setting described under [Commander Duel](#commander-duel) reverted on every state copy. |

## Building the distribution

Requirements: a JDK (the distribution used here was built with JDK 21), Maven, and a clone of the fork.

```bash
git clone https://github.com/WillWroble/mage .mz_engine
git -C .mz_engine checkout cb7e9c6fe2dcbe2f650a77dd98a87771e730f22f
python integration/xmage/install.py .mz_engine
mvn -B -ntp -f .mz_engine/pom.xml -pl Mage.MageZero -am -Dmaven.test.skip=true package
```

All 35 modules build. Extract the distribution ZIP from `.mz_engine/Mage.MageZero/target/` (check the file name Maven produced) into `xmage/`. It contains `lib/` and the `mz-policy` launchers. The launchers use the `lib/*` classpath and an 8 GB Java heap, which leaves memory for the Python learner. XMage's H2 card database needs local socket access, so every command binds it to localhost (`-Dh2.bindAddress=127.0.0.1`).

`PolicyGames game.yml` plays one batch and exits. `PolicyGames --serve`, which `mz selfplay` and `mz teacher-collect` use, boots once (card database, card scan, JIT warm-up) and then plays one batch per `game.yml` path read from stdin. It replies with a single line, `MZ-POLICY RESULT <terminal> <requested> <truncated> <errors> <timeouts>` or `MZ-POLICY ERROR <message>`, and EOF on stdin ends it. An optional `server.ports` / `server.opponent_ports` list in the game YAML spreads games across policy-service shards (game g uses entry g mod n).

### Rebuilding after a Java change

A full Maven build is slow. After editing the adapter or a patched engine class, compile only those classes against the existing distribution and update the four affected jars. Stop every Java process that uses `xmage/lib` first, and keep a backup of the jars.

```bash
python integration/xmage/install.py .mz_engine
OUT=$(mktemp -d)
AI=.mz_engine/Mage.Server.Plugins/Mage.Player.AI/src/main/java/mage/player/ai
MA=.mz_engine/Mage.Server.Plugins/Mage.Player.AI.MA/src/mage/player/ai
javac -nowarn -encoding UTF-8 -cp "xmage/lib/*" -d "$OUT" \
    $(find integration/xmage/src integration/xmage/src-ma -name '*.java') \
    $AI/ComputerPlayer.java $MA/ComputerPlayer6.java \
    .mz_engine/Mage/src/main/java/mage/abilities/mana/ManaOptions.java \
    .mz_engine/Mage/src/main/java/mage/players/PlayerImpl.java
LIB=$PWD/xmage/lib
(cd "$OUT" &&
 jar uf "$LIB/mage-player-ai-1.4.58.jar" mage/player/ai/policy/*.class mage/player/ai/ComputerPlayer.class 'mage/player/ai/ComputerPlayer$PickedCard.class' &&
 jar uf "$LIB/mage-player-ai-ma-1.4.58.jar" mage/player/ai/ComputerPlayer6*.class mage/player/ai/TeacherPlayer*.class &&
 jar uf "$LIB/mage-1.4.58.jar" mage/abilities/mana/ManaOptions*.class mage/players/PlayerImpl*.class &&
 jar uf "$LIB/mage-magezero-1.4.58.jar" org/mage/magezero/*.class)
```

## Exporting the card catalog

The policy embeds cards by catalog ID, so the catalog must be exported once for the whole deck pool before a policy is initialized, and must not change afterwards. Place the decks in `xmage/decks/`, then:

```bash
cd xmage
java -Xmx8G -Dh2.bindAddress=127.0.0.1 -cp "lib/*" org.mage.magezero.PolicyCatalogExporter \
    ../artifacts/cards.json --sources ../.mz_engine/Mage.Sets/src decks/first.dck decks/second.dck
cd ..
python scripts/annotate_policy_catalog.py --catalog artifacts/cards.json \
    --annotations configs/card-roles.json --output artifacts/cards-with-roles.json
mz init --catalog artifacts/cards-with-roles.json --output artifacts/initial.pt
```

The exporter assigns sorted, name-based IDs and coarse card semantics. Include every intended deck and sideboard (a Commander deck's commander is its sideboard). It also adds the other faces of the pool's cards (a transformed planeswalker, an MDFC played as its land) and the tokens the cards can create, under the names XMage gives them on the battlefield (`Goblin Token`, `Treasure Token`; catalog key `xmage-token:`). Tokens are found by walking each card's effects. `--sources` also scans card sources for tokens built inside custom effects (Legion Warboss, Loyal Apprentice, Sokenzan). Token copies of cards use the card's own ID. Cards generated from outside the pool (random or conjured cards) are not covered.

Do not append cards to an existing checkpoint's catalog or regenerate it with different IDs: a changed pool needs a fresh policy. Every card also needs reviewed roles in `configs/card-roles.json`. Missing annotations fail explicitly rather than being read as "no capability"; see [Statistical roles](statistical-roles-implementation.md#role-metadata-and-coverage). The bundled annotations cover the Winota and Kinnan pool (written by `scripts/commander_roles.py`), the `simplegreen` validation deck, and a few example cards.

## Commander Duel

The learner pilots **Winota, Joiner of Forces** and the gauntlet opponent is **Kinnan, Bonder Prodigy**, piloted by XMage's AI.

- `game_mode: commander` creates XMage's `CommanderDuel` and `CommanderDuelMatch` (loaded by name): 100-card decks, 40 life, seven cards, London mulligan without a free mulligan, and a loss at 21 combat damage from one commander. This matches the server's "Commander Two Player Duel" table. `game_mode: normal` plays a two-player constructed game; other modes are refused.
- The commander is the deck's sideboard. `GameCommanderImpl` moves it to the command zone, and the encoder shows both commanders there ([protocol](policy-protocol.md#feature-layout-in-the-java-adapter)).
- `server.learner_keeps_deck: true` moves the learner's deck with it when seats swap, so the learner always plays player_a's deck. `server.sample: false` makes policy players greedy (evaluation only; unsampled games are never recorded). `opponent_ai.think_seconds` caps XMage's AI search per decision (by default 3 s per skill level).
- Game configurations: `configs/commander-mirror.yml` (Winota against Winota, both policy) and `configs/commander-gauntlet.yml` (the policy's Winota against XMage's AI with Kinnan).
- The deck lists in `decks-src/` are converted with `org.mage.magezero.DeckListConverter list.txt deck.dck`. Lines after the first blank line are the sideboard (the commander), and unknown names fail. Three cards of the original lists are missing from this XMage build and were substituted: Mayor Tong of Chin Village → Ragavan, Nimble Pilferer (Winota); Avatar Sanctuary → Breeding Pool and Tidus, Zanarkand Fayth → Bounding Krasis (Kinnan).
- The resulting catalog has 173 card kinds: 158 deck cards, 5 other faces and 10 tokens. The [README](../README.md#training-the-commander-agent) lists the commands.

Players run in XMage's test mode, where some checks throw instead of being skipped as in a real game ("AI can't find good blocker combination", errors inside AI simulations). `PolicyGames` and `PolicyArena` therefore turn off `fastFailInTestMode` for every player, and `player-fast-fail.patch` keeps that setting across state copies.

Fixes made in the adapter while bringing up the Commander games (in `PolicyPlayer` unless noted):

| Symptom | Fix |
|---|---|
| Phyrexian Revoker's "name a card" offered every card in XMage (over 2,048 candidates) | Oversized named choices are restricted to the names in the card catalog. |
| Sanctum Prelate's "choose a number" ranges from 0 to 2³¹ | Amount ranges above the candidate limit offer their smallest 256 values. |
| XMage lists some abilities as playable that then fail during payment, which aborted the game | A refused activation is treated as no action (as XMage's AI does) and is not offered again in that priority window. |
| A failed activation restored a saved state, including the history watcher, so events were sent twice | The event sequence counter and the "already sent" mark are shared by every copy of `PolicyHistory`. |
| `chooseReplacementEffect` parsed effect IDs as numbers | It returns the index in XMage's map order. |
| XMage's AI looped one action for a whole turn; the single 15-minute interrupt then held the entire batch, and the AI clears interrupts while it waits for its search | `PolicyGames` checks every second and keeps interrupting once a game exceeds `training.max_minutes` or one turn exceeds `training.max_turn_seconds` (default 120). Such games count as timeouts and are discarded. |
| A Kinnan turn emitted more than 1,024 history events between two policy decisions, which the service refuses to truncate | `model.max_events` is 8,192 in `configs/policy-final.yml` (a validation limit; no weights depend on it). |

The engine log ends each game with a summary line (`policy game N TERMINAL: turn T; PlayerA (policy) won life 35 poison 0 commander damage 0 ...`), which shows how games end.

## Collecting and training

Both players of a self-play game YAML must be `type: policy` with `hiddenInfo.see_opponent_hand: false`. The known deck list is public; the current hand and the library order remain private. Mana is paid by XMage's automatic payment. Then:

```bash
mz selfplay --checkpoint artifacts/initial.pt --game configs/policy-game.yml --engine-script $ENGINE \
    --output artifacts/experiment --iterations 1 --games 32
```

Each generation serves the frozen learner checkpoint on port 50052 (and a historical opponent on 50053 when the league supplies one), plays the batch, validates the new complete trajectories, runs one PPO update, and saves a versioned checkpoint and `metrics.json`. Deck and configuration paths are resolved before Java starts. Occupied ports, mismatched catalogs, stale trajectories, failed engines and missing terminal data all fail explicitly, and output folders must be fresh unless resumed. [Long runs](long-run.md) covers the options for long runs.

## Playing against the policy

`mz play` starts a policy service for a checkpoint, then the fork's own XMage server from `xmage/lib` with [`integration/xmage/server-config.xml`](../integration/xmage/server-config.xml). That configuration maps the **"Computer - monte carlo"** opponent type to `PolicyPlayer`. Stock XMage clients already list that type, so no client changes are needed (upstream MageZero used the same slot). The server listens on `localhost:17171` only.

```bash
mz play --checkpoint artifacts/commander-run/best.pt --greedy   # --device cuda is optional
```

1. Open an XMage client of the server's version (1.4.58) and connect to `localhost`, port 17171.
2. Create a table (Commander Two Player Duel for the Commander agent) and add a **Computer - monte carlo** player.
3. Give the policy a deck from its card pool and play a deck from the same pool yourself. The policy reads its opponent's announced deck list at game start, and every card in that list must be in the checkpoint's catalog.

### In the browser

`mz web` needs no XMage client:

```bash
mz web --checkpoint artifacts/anchors/a5-b-gen24.pt     # then open http://127.0.0.1:8765
```

It starts the bot's policy service (port 50071), an engine (`PolicyGames --serve`) and a web server on `127.0.0.1:8765`, and opens the page. Pick your deck and the bot's (only decks in `xmage/decks/` whose cards are all in the checkpoint's catalog are offered; two 99-card decks play Commander Duel), who goes first, and whether the bot plays its most likely action (default) or samples. One game runs at a time. New game abandons the running one, and Concede ends it as a loss.

Your seat is a `WebPlayer` (game YAML `player_b.type: web`, `server.web_endpoint`). It is a `PolicyPlayer` whose decisions go to the browser instead of the policy service, so you choose among exactly the candidates the policy is offered (`PolicyCandidates`), with readable labels, and XMage pays mana automatically for both players. Decisions with one legal option are not asked. The engine posts each decision with a snapshot of what you may see (both battlefields, your hand, public zones, hidden counts, the stack, combat and XMage's game log) to `/engine/decide`, and the server holds the request until you click. The board also refreshes at each of your priority passes, so the bot's actions show while they happen. The page auto-passes priority on an empty stack outside your main phases, and when your own spell is on top of the stack; both can be switched off, and Space or Enter passes. Card images come from Scryfall.

`mz web` compiles the adapter into `artifacts/web-engine/classes` (again whenever a source is newer) and puts that folder ahead of `xmage/lib` on the engine's classpath, so it runs the current sources without rebuilding the jars that training uses. Its logs are in a temporary `mz-web-*` folder printed at start.

Games against people are inference only: nothing is recorded, and no privileged labels are sent. `PolicyPlayer` reads its endpoint from `-Dmagezero.policy.endpoint` (or `MAGEZERO_POLICY_ENDPOINT`) and initializes itself in `beginTurn`, which `GameImpl.init` calls for every player before shuffling and opening hands. Each new game of a match starts a new recurrent session. Player copies share the live session, so XMage state restores (failed payments, a human's undo) keep the policy's memory.

## Validation

- `scripts/validate_xmage_policy.py` compiles the adapter against a core engine build, plays one real XMage rules game with synthetic free creatures and lands over HTTP (12-card decks, six life, three starting cards), and trains on its two trajectories. It caught engine API, watcher-copy, batched-event, mulligan, action-application, privacy-schema and transport issues. It needs the modules `Mage`, `Mage.Common`, `Mage.MageZero`, `Mage.Server.Plugins/Mage.Player.AI` and `Mage.Server.Plugins/Mage.Game.TwoPlayerDuel` built with their dependencies in `artifacts/maven`:

  ```bash
  mvn -B -ntp -f .mz_engine/Mage/pom.xml -Dmaven.repo.local=$PWD/artifacts/maven -DskipTests compile
  mvn -N -B -ntp -f .mz_engine/Mage/pom.xml -Dmaven.repo.local=$PWD/artifacts/maven dependency:get \
      -Dartifact=org.yaml:snakeyaml:2.2 -Dtransitive=false
  python scripts/policy_smoke.py --output artifacts/smoke --device cpu
  python scripts/validate_xmage_policy.py --checkpoint artifacts/smoke/initial.pt --output artifacts/java-smoke
  ```

- With the full distribution, two normal 60-card games of the fork's `simplegreen` mirror (`configs/smoke/real-card-smoke.yml`) completed through `mz selfplay`, producing four terminal trajectories with 1,172 decisions that passed validation and trained a new checkpoint. This exercises deck import, real cards, mana payment, combat, the engine collector, the service lifecycle and the learner; it is not a win-rate or card-coverage test.
- Human play was checked by starting the server (4 player types loaded) and by two games against XMage's AI built through the server's player constructor, with zero engine errors.

## Coverage and remaining limits

Policy callbacks cover priority, ordinary targets, use (yes/no), mulligans, named choices, amounts, modes, attacks, blocks, trigger and replacement-effect ordering, spell variants, piles, and target-amount allocation. Each is a choice among network-scored candidates without game-tree search. XMage's base player still handles mana payment and some specialized rule callbacks, and the rules engine's own simulations still decide what is playable.

Combat is declared one creature at a time. A creature that can block additional attackers chooses again until it stops or reaches its limit. Block candidates show how many blockers the attacker already has and its minimum (2 for menace). XMage enforces the minimum after declaration by removing blockers that leave an attacker under it, so a lone block on a menace creature is legal to choose but has no effect. Banding, compelled combat and whole attack/block configurations are not covered, and multi-blocking has not been exercised with real cards. Pile candidates carry aggregate contents and point to their highest mana-value card. Amount choices above 2,048 candidates and further specialized rule callbacks need extensions.

The opponent-hand bounds assume an announced main deck and track public locations of its original cards. Cards publicly returned to hand are remembered conservatively. Reveals, stolen cards, face-down exile permissions, generated or copied cards and transformed names need a more complete knowledge tracker for arbitrary decks. Off-deck hidden hand labels fail, rather than letting private information expand the actor's candidates. Test a new deck pool for these mechanics before a large run.

Aborted and time-limited games produce no PPO data, and excluding them can bias learning, so their rate is reported per generation. The runner supports two-player constructed games and Commander Duel; draft and multiplayer Commander are outside this adapter.
