# XMage integration

The policy plays inside the XMage rules engine through a Java adapter: a direct `PolicyPlayer` with its observation encoder and history watcher, a recording `TeacherPlayer`, a batch game runner, and a catalog exporter.

The adapter runs on **official XMage releases** ([magefree/mage](https://github.com/magefree/mage/releases)), unmodified. XMage is never patched or rebuilt: the adapter is compiled against a release's jars and only extends XMage's public player API (the `Player` interface methods that a human player answers too). Updating XMage is therefore a one-line change ([below](#updating-xmage)).

| Path | Contents |
|---|---|
| `src/mage/player/ai/policy/` | `PolicyPlayer` (the policy as an XMage player), `PolicyEncoder`, `PolicyHistory`, `PolicyCandidates`, `PolicyClient`, `HandSummary`, `RecordingPlayer`, `WebPlayer` (a person in the browser, `mz web`) |
| `src-ma/mage/player/ai/` | `XMageAiPlayer` (XMage's Mad AI, the gauntlet opponent, with the fixes below) and `TeacherPlayer` (the same AI with every decision recorded over the policy's candidate list, for behaviour cloning) |
| `src/org/mage/magezero/` | `PolicyGames` (batch runner), `PolicyCatalogExporter`, `DeckListConverter` |
| `patches/` | Optional engine fixes, as diffs against the release's own sources (see [below](#engine-fixes)) |
| `xmage-release.txt` | The official release the repository is tested with |
| `build.py` | Builds the distribution from that release |
| `test/PolicySmoke.java` | A synthetic-card game over HTTP (`scripts/validate_xmage_policy.py`) |

## Building the distribution

Requirements: a JDK 21 or newer (`javac` on `PATH`, or `JAVA_HOME`), Python, git and an internet connection. No Maven and no XMage checkout.

```bash
python integration/xmage/build.py
```

The script downloads the release named in `integration/xmage/xmage-release.txt` (its `mage-full` ZIP, about 250 MB, cached in `.mz_cache/`) and writes `xmage/`:

| Path | Contents |
|---|---|
| `lib/`, `plugins/`, `config/` | The release's XMage server, as published |
| `mz/magezero-adapter.jar` | The adapter, compiled against those jars |
| `mz/engine-fixes.jar` | The engine fixes that apply to this release |
| `mz/build.json` | Release, XMage version, adapter source hash and the status of each fix |
| `mz-policy.sh`, `mz-policy.bat` | Launchers for `PolicyGames`, with `mz/` ahead of XMage's jars on the classpath and an 8 GB heap (`MZ_HEAP` overrides it), which leaves memory for the Python learner |
| `config/magezero-server.xml` | The release's `config.xml` for `mz play`: local only, with the policy as a player type |
| `decks/` | The decks in `decks/` |

Run it again after changing the adapter's Java sources (it takes seconds once the release is cached). Rebuilding keeps `decks/`, logs and, for the same XMage version, the card database; stop every Java process that uses `xmage/` first. XMage's H2 card database needs local socket access, so every command binds it to localhost (`-Dh2.bindAddress=127.0.0.1`). The first start of a new version builds the database (about a minute).

`PolicyGames game.json` plays one batch and exits. `PolicyGames --serve`, which `mz selfplay`, `mz evaluate` and `mz teacher-collect` use, boots once (card database, card scan, JIT warm-up) and then plays one batch per game file path read from stdin. It replies with a single line, `MZ-POLICY RESULT <terminal> <requested> <truncated> <errors> <timeouts>` or `MZ-POLICY ERROR <message>`, and EOF on stdin ends it. The game files are JSON, written by the Python runner from the game YAML in `configs/`. An optional `server.ports` / `server.opponent_ports` list spreads games across policy-service shards (game g uses entry g mod n). With `-Dmagezero.gameLog=true` (for example in `JAVA_TOOL_OPTIONS`) the engine log also shows each game's own log, as an XMage client would.

## Updating XMage

```bash
python integration/xmage/build.py --check --release latest   # does the adapter compile against the newest release?
python integration/xmage/build.py --release latest           # build xmage/ on it
```

To move the repository to a new release, put its tag in `xmage-release.txt`. The `xmage` CI workflow compiles the adapter against the pinned release on every change and against the newest release every Monday, so a release that breaks the adapter shows up as a failed run. A compile failure names the XMage API that changed; the adapter uses few of them (see `PolicyPlayer`, `XMageAiPlayer`, `TeacherPlayer`).

After an update, play a short evaluation before relying on it, for example `mz evaluate --games 64` ([Playing the agent](playing-the-agent.md#how-strong-it-is) lists the published scores). Compiling proves the API still fits, not that the rules behave the same: a release can change cards or rules the policy was trained on. Cards of a deck that a release renames or removes fail at deck import.

## Engine fixes

The adapter needs no change to XMage. Three problems seen in training are handled in the adapter's own classes:

| Problem | Fix |
|---|---|
| XMage's AI repeated one activated ability for the whole turn (Basalt Monolith's untap with Kinnan, a real infinite-mana loop; Freed from the Real) until the per-turn watchdog ended the game, in about 8% of Kinnan games | `XMageAiPlayer` passes instead of activating the same ability more than 50 times in one turn (`-Dxmage.ai.maxActivationsPerTurn`), and skips the rest of that priority's plan. Spells are not counted. |
| Every XMage AI player in a JVM shares one pool of 5 search threads, so 16 concurrent gauntlet games queued on it and hit their think-time cap on most decisions | `PolicyGames` resizes the pool to the CPU count (`-Dxmage.ai.simulationThreads`). If a release moves the pool, a warning is logged and the default size stays. |
| `PlayerImpl`'s copy constructor drops the fast-fail flag, so the setting described under [Commander Duel](#commander-duel) reverted on every state copy | `PolicyPlayer` and `XMageAiPlayer` copy it themselves. |

Two fixes are inside XMage classes, so `build.py` applies them to the release's own copy of each file (downloaded for the release's tag), compiles them into `mz/engine-fixes.jar` and puts that jar first on the classpath. A fix that no longer applies or compiles against a release is skipped with a warning (in the build output, `mz/build.json` and the CI run); games still run without it.

| Patch | Why |
|---|---|
| `mana-payment-depth.patch` | `ComputerPlayer` auto-payment recursed without end when mana abilities that cost mana (Prismite, Energy Refractor and Urn of Godfire in the Kinnan deck) paid for each other, overflowing the stack inside XMage AI simulations. Payment now gives up below 8 nested mana abilities. (`PolicyPlayer` and `XMageAiPlayer` also stop there for their own payments.) |
| `mana-options.patch` | `ManaOptions` (the "what can this player afford" estimate behind `getPlayable`) simulated repeatable mana abilities that cost mana an unlimited number of times and kept every combination, with quadratic duplicate checks. On a Kinnan board one call ran for over 7 CPU-minutes, ignoring the game's timeout. Repeats are now capped at 4 and the options at the 64 largest. Only affordability estimates change; actual payment does not. |

### History

Until October 2026 the adapter was installed into [WillWroble/mage](https://github.com/WillWroble/mage) at commit `cb7e9c6f` (MageZero's fork of XMage 1.4.58) and patched five engine files. The trained agent, its training run and the evaluations in [Playing the agent](playing-the-agent.md) used that fork. Two differences mattered when moving to official XMage: official XMage runs game code only on threads named as game threads (`PolicyGames` names its workers so), and its test mode deals no opening hands (the fork had changed that), so `PolicyGames` plays real games instead.

## Exporting the card catalog

The policy embeds cards by catalog ID, so the catalog must be exported once for the whole deck pool before a policy is initialized, and must not change afterwards. Place the decks in `xmage/decks/`, then (on Windows, separate the classpath with `;`):

```bash
git clone --depth 1 --branch "$(cat integration/xmage/xmage-release.txt)" https://github.com/magefree/mage xmage-src
cd xmage
java -Xmx8G -Dh2.bindAddress=127.0.0.1 -cp "mz/*:lib/*:plugins/*" org.mage.magezero.PolicyCatalogExporter \
    ../artifacts/cards.json --sources ../xmage-src/Mage.Sets/src decks/first.dck decks/second.dck
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
- The deck lists in `decks-src/` are converted with `org.mage.magezero.DeckListConverter list.txt deck.dck`. Lines after the first blank line are the sideboard (the commander), and unknown names fail. Three cards of the original lists were missing from XMage 1.4.58 and are still missing in 1.4.62, so they were substituted: Mayor Tong of Chin Village → Ragavan, Nimble Pilferer (Winota); Avatar Sanctuary → Breeding Pool and Tidus, Zanarkand Fayth → Bounding Krasis (Kinnan). `decks-src/ingris.txt` (Ingris Stingerquill, needs 1.4.62) uses the Oracle names of two Universes Beyond printings XMage lacks: A Promise Fulfilled → Light Up the Stage, Zidane Tribal → Ragavan, Nimble Pilferer. `decks-src/vraska.txt` (Vraska, Soul of Stone, also 1.4.62) converts unchanged. Both are outside the trained agent's catalog.
- The resulting catalog has 173 card kinds: 158 deck cards, 5 other faces and 10 tokens. The [README](../README.md#training-the-commander-agent) lists the commands.

Players run in XMage's player test mode, where some checks throw instead of being skipped as in a real game ("AI can't find good blocker combination", errors inside AI simulations). `PolicyGames` therefore turns off `fastFailInTestMode` for every player, and the adapter's players keep that setting across state copies. The game itself is a real game, not XMage's game test mode (which deals no opening hands).

Fixes made in the adapter while bringing up the Commander games (in `PolicyPlayer` unless noted):

| Symptom | Fix |
|---|---|
| Phyrexian Revoker's "name a card" offered every card in XMage (over 2,048 candidates) | Oversized named choices are restricted to the names in the card catalog. |
| Sanctum Prelate's "choose a number" ranges from 0 to 2³¹ | Amount ranges above the candidate limit offer their smallest 256 values. |
| XMage lists some abilities as playable that then fail during payment, which aborted the game | A refused activation is treated as no action (as XMage's AI does) and is not offered again in that priority window. |
| A failed activation restored a saved state, including the history watcher, so events were sent twice | The event sequence counter and the "already sent" mark are shared by every copy of `PolicyHistory`. |
| `chooseReplacementEffect` parsed effect IDs as numbers | It returns the index in XMage's map order. |
| XMage's AI looped one action for a whole turn; the single 15-minute interrupt then held the entire batch, and the AI clears interrupts while it waits for its search | `PolicyGames` checks every second and keeps interrupting once a game exceeds `training.max_minutes` or one turn exceeds `training.max_turn_seconds` (default 120). Such games count as timeouts and are discarded. |
| A Kinnan turn emitted more than 1,024 history events between two policy decisions, which the service refuses to truncate | `model.max_events` is 8,192 in `configs/policy.yml` (a validation limit; no weights depend on it). |

The engine log ends each game with a summary line (`policy game N TERMINAL: turn T; PlayerA (policy) won life 35 poison 0 commander damage 0 ...`), which shows how games end.

## Collecting and training

Both players of a self-play game YAML must be `type: policy` with `hiddenInfo.see_opponent_hand: false`. The known deck list is public; the current hand and the library order remain private. Mana is paid by XMage's automatic payment. Then:

```bash
mz selfplay --checkpoint artifacts/initial.pt --game configs/commander-mirror.yml --engine-script $ENGINE \
    --output artifacts/experiment --iterations 1 --games 32
```

Each generation serves the frozen learner checkpoint on port 50052 (and a historical opponent on 50053 when the league supplies one), plays the batch, validates the new complete trajectories, runs one PPO update, and saves a versioned checkpoint and `metrics.json`. Deck and configuration paths are resolved before Java starts. Occupied ports, mismatched catalogs, stale trajectories, failed engines and missing terminal data all fail explicitly, and output folders must be fresh unless resumed. [Long runs](long-run.md) covers the options for long runs.

## Playing against the policy

`mz play` starts a policy service for a checkpoint, then the release's XMage server from `xmage/` with `xmage/config/magezero-server.xml` (written by `build.py` from the release's own `config.xml`). That configuration maps the **"Computer - monte carlo"** opponent type to `PolicyPlayer`. Stock XMage clients already list that type, so no client changes are needed. The server listens on `localhost:17171` only.

```bash
mz play --checkpoint artifacts/winota-agent.pt --greedy   # --device cuda is optional
```

1. Open an XMage client of the server's version (`xmage_version` in `xmage/mz/build.json`; the client is in the same release's `mage-full` ZIP) and connect to `localhost`, port 17171.
2. Create a table (Commander Two Player Duel for the Commander agent) and add a **Computer - monte carlo** player.
3. Give the policy a deck from its card pool and play a deck from the same pool yourself. The policy reads its opponent's announced deck list at game start, and every card in that list must be in the checkpoint's catalog.

### In the browser

`mz web` needs no XMage client ([Playing in the browser](web-play.md) is the user guide):

```bash
mz web --checkpoint artifacts/winota-agent.pt     # then open http://127.0.0.1:8765
```

It starts the bot's policy service (port 50071), an engine (`PolicyGames --serve`) and a web server on `127.0.0.1:8765`, and opens the page. Pick your deck and the bot's (only decks in `xmage/decks/` whose cards are all in the checkpoint's catalog are offered; two 99-card decks play Commander Duel), who goes first, and whether the bot plays its most likely action (default) or samples. One game runs at a time. New game abandons the running one, and Concede ends it as a loss.

Your seat is a `WebPlayer` (game file `player_b.type: web`, `server.web_endpoint`). It is a `PolicyPlayer` whose decisions go to the browser instead of the policy service, so you choose among exactly the candidates the policy is offered (`PolicyCandidates`), with readable labels, and XMage pays mana automatically for both players. Decisions with one legal option are not asked. The engine posts each decision with a snapshot of what you may see (both battlefields, your hand, public zones, hidden counts, the stack, combat and XMage's game log) to `/engine/decide`, and the server holds the request until you click. The board also refreshes at each of your priority passes, so the bot's actions show while they happen. The page auto-passes priority on an empty stack outside your main phases, and when your own spell is on top of the stack; both can be switched off, and Space or Enter passes. Card images come from Scryfall.

`mz web` compiles the adapter into `xmage/mz/web-classes` (again whenever a source is newer; `build.py` clears it) and puts that folder ahead of the adapter jar on the engine's classpath, so it runs the current sources without rebuilding the jar that training uses. Its logs are in a temporary `mz-web-*` folder printed at start.

Games against people are inference only: nothing is recorded, and no privileged labels are sent. `PolicyPlayer` reads its endpoint from `-Dmagezero.policy.endpoint` (or `MAGEZERO_POLICY_ENDPOINT`) and initializes itself in `beginTurn`, which `GameImpl.init` calls for every player before shuffling and opening hands. Each new game of a match starts a new recurrent session. Player copies share the live session, so XMage state restores (failed payments, a human's undo) keep the policy's memory.

## Validation

- `scripts/validate_xmage_policy.py` plays one real XMage rules game with the built adapter and synthetic free creatures and lands over HTTP (12-card decks, six life, three starting cards), and trains on its two trajectories. It caught engine API, watcher-copy, batched-event, mulligan, action-application, privacy-schema and transport issues:

  ```bash
  python scripts/policy_smoke.py --output artifacts/smoke --device cpu
  python scripts/validate_xmage_policy.py --checkpoint artifacts/smoke/initial.pt --output artifacts/java-smoke
  ```

- With the full distribution (on the fork), two normal 60-card games of the fork's `simplegreen` mirror completed through `mz selfplay`, producing four terminal trajectories with 1,172 decisions that passed validation and trained a new checkpoint. This exercises deck import, real cards, mana payment, combat, the engine collector, the service lifecycle and the learner; it is not a win-rate or card-coverage test.
- Human play was checked by starting the server (4 player types loaded) and by two games against XMage's AI built through the server's player constructor, with zero engine errors.

## Coverage and remaining limits

Policy callbacks cover priority, ordinary targets, use (yes/no), mulligans, named choices, amounts, modes, attacks, blocks, trigger and replacement-effect ordering, spell variants, piles, and target-amount allocation. Each is a choice among network-scored candidates without game-tree search. XMage's base player still handles mana payment and some specialized rule callbacks, and the rules engine's own simulations still decide what is playable.

Combat is declared one creature at a time. A creature that can block additional attackers chooses again until it stops or reaches its limit. Block candidates show how many blockers the attacker already has and its minimum (2 for menace). XMage enforces the minimum after declaration by removing blockers that leave an attacker under it, so a lone block on a menace creature is legal to choose but has no effect. Banding, compelled combat and whole attack/block configurations are not covered, and multi-blocking has not been exercised with real cards. Pile candidates carry aggregate contents and point to their highest mana-value card. Amount choices above 2,048 candidates and further specialized rule callbacks need extensions.

The opponent-hand bounds assume an announced main deck and track public locations of its original cards. Cards publicly returned to hand are remembered conservatively. Reveals, stolen cards, face-down exile permissions, generated or copied cards and transformed names need a more complete knowledge tracker for arbitrary decks. Off-deck hidden hand labels fail, rather than letting private information expand the actor's candidates. Test a new deck pool for these mechanics before a large run.

Aborted and time-limited games produce no PPO data, and excluding them can bias learning, so their rate is reported per generation. The runner supports two-player constructed games and Commander Duel; draft and multiplayer Commander are outside this adapter.
