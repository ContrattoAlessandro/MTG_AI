# Next session: single-deck Commander Duel agent

Written 12 October 2026 as a handoff. Read this file first, then `docs/architecture.md`, `docs/long-run.md`, `docs/policy-protocol.md` and `docs/xmage-integration.md`.

## Goal

One agent that pilots **one Commander deck** (example: Winota, Joiner of Forces) as well as possible in **Commander Duel** (two players). It is not a general player of every deck. A human must be able to play against it in the XMage client; that already works through `mz play`.

## Decisions already made

| Decision | Detail |
|---|---|
| Mana payment | XMage's auto-tapper (`autoTap = true`). Do not add policy-controlled payment. |
| Human play | Done: `mz play`, using the "Computer - monte carlo" opponent slot. |
| Learner deck | Exactly one deck. The policy never trains on another deck. |
| Opponents | Mirror self-play plus league snapshots, **plus** a gauntlet of decks piloted by XMage's AI (see task C). Not mirror only: a pure mirror only meets the deck's own interaction. |
| Format | Commander **Duel** only. Multiplayer pods would need a redesign: one opponent is assumed throughout the observation, belief and seats. |

## Inputs needed from the user before starting

1. **The learner's deck list** (100 cards, commander marked). Put it in `xmage/decks/` as a `.dck`; XMage reads the commander from the sideboard section (`SB:` lines).
2. **3–5 gauntlet decks** typical of the user's playgroup, also as `.dck` with commanders.
3. **Learning from the user's own games (task E): yes or no.** Recommended for a synergy deck.

## State at handoff

- Nothing is committed. The working tree holds the whole policy redesign plus the 12 October changes. Ask the user whether to commit before starting; if yes, branch first and use the attribution in the session's system reminder.
- 39 tests pass: `.venv/Scripts/python.exe -m pytest -q`.
- Patched jars: `xmage/lib/mage-player-ai-1.4.58.jar` and `mage-magezero-1.4.58.jar`. Backups: `artifacts/jar-backup-20261012/` (original) and `artifacts/jar-backup-20261012-pre-coverage/` (before the 12 October evening changes).
- The 12 October work, all in the working tree:
  - Human play: `PolicyPlayer` server constructor and `beginTurn` initialization, `integration/xmage/server-config.xml`, `mz play` in `src/magezero/cli.py`.
  - Mulligan option features: `HandSummary.java`, `PolicyPlayer.chooseMulligan`.
  - Entity features: current keywords, counters by type, token and copy flags, land drops (`PolicyEncoder.addObject`).
  - Multi-block and menace block features, pile aggregates, target prompt text.
  - Learner host-sync removal (`statistics.py` `validate=False`, `forward_episodes(event_counts=...)`).
  - `configs/policy-final.yml`.
  - The feature layouts are in `docs/policy-protocol.md`. **Any checkpoint made before 12 October reads the new slots with their old meanings: run a fresh `mz init`.**
- GPU: another project's training job (`ablations/campaign.py`) was using the RTX 5070. Ask before any long GPU job and treat GPU timings as contended.

## Rebuild loop after any Java change

```bash
ROOT=/c/Users/alexa/Desktop/MTG_AI; OUT="$TEMP/claude/mzjava"; rm -rf "$OUT"; mkdir -p "$OUT"
cd "$ROOT" && javac -nowarn -encoding UTF-8 -cp "xmage/lib/*" -d "$OUT" $(find integration/xmage/src -name '*.java')
(cd "$OUT" && jar uf "$ROOT/xmage/lib/mage-player-ai-1.4.58.jar" mage/player/ai/policy/*.class \
           && jar uf "$ROOT/xmage/lib/mage-magezero-1.4.58.jar" org/mage/magezero/*.class)
python integration/xmage/install.py .mz_engine
```

Check that no `java.exe` is running from `xmage/lib` first. Back up both jars before the first patch of the session. The XMage engine source for reference is in `.mz_engine/` (pinned fork commit `cb7e9c6f`).

Quick real-game smoke (CPU only, about 30 s):

```bash
.venv/Scripts/mz.exe selfplay --checkpoint <ckpt> --game <game.yml> --config configs/policy-smoke.yml --engine-script xmage/mz-policy.bat --output "$TEMP/claude/<fresh>" --iterations 1 --games 4 --engine-threads 2 --device cpu --actor-device cpu --eval-every 1 --eval-games 2 --eval-ai-skill 3 --keep-episodes
```

---

## Task A: Commander Duel support (about 1.5 days)

**A1. Game creation.** `integration/xmage/src/org/mage/magezero/PolicyGames.java`, `runBatch` and `play`:
- Accept `game_mode: commander` alongside `normal`. Keep refusing everything else.
- For commander, create `mage.game.CommanderDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE, mulligan, 40, 7)`, with minimum deck size 100 and 40 life, and a `CommanderDuelMatch`. Load it by reflection, like `xmageAi`, so `mage-magezero` gets no compile-time dependency on the commander module, or add the Maven dependency.
- Keep loading the sideboard (`load()` already does): `GameCommanderImpl` moves the commander from the sideboard to the command zone at init.
- Look for the same hard-coded 60/20/7 in `PolicyArena.java` (it is legacy; leave it unless it is used).
- `mz play` needs no change: the server config already lists "Commander Two Player Duel".

**A2. Observation of the command zone.** `PolicyEncoder.observation`:
- Add command-zone objects as public entities (zone 7, kind 2) for both players: `game.getState().getCommand()`, or each player's `game.getCommandersIds(player, CommanderCardType.ANY, false)`, mapped to cards in the command zone.
- Entity features. The free object slots are none left in 0–31 after 12 October, so choose carefully and update `docs/policy-protocol.md`:
  - an "is commander" flag
  - commander tax, or times cast (`CommanderPlaysCountWatcher`)
  - commander damage. On the player entity: damage this player has taken from each opposing commander, via `CommanderInfoWatcher.getDamageToPlayer()`. Player entity slots 15–20 and 21–31 are free.
  - Freeing object slots: slot 26 "transformed" could double as a flags field, or `ENTITY_FEATURES` could grow from 32 to 48. Growing it is a schema change in `schema.py` and `model.py` (`entity_numeric` input size). Fresh init is happening anyway, so growing is the clean option. Update the tests that build 32-wide vectors.
- Globals: life is divided by 40, so 40 life already maps to 1.0. Fine.

**A3. Opponent deck belief.** `PolicyEncoder` constructor:
- It excludes sideboard cards, so the opponent's commander is not in the hidden-deck candidates. That is correct: the commander is public in the command zone.
- `trainingLabel` throws if an off-deck card is in the opponent's hand. A commander bounced to hand would trigger it. Either add commanders to the candidate list with upper bound 1, or treat the commander as a known card (public lower bound) when it moves to hand.
- `max_deck_kinds` 128 is enough for 99 singletons. Check `max_entities`, `max_actions` and `max_events` against measured commander games.

**A4. Limits.** `training.max_turns` (start at 40–60 and measure), `max_minutes`, `PPOConfig.max_episode_steps` (10000; 40-life games are longer), and the service's `max_steps`.

**A5. Tests and validation.**
- Java: a Commander Duel smoke with the user's deck against itself, 4 games, zero engine errors.
- Python: schema tests for any feature-width change.
- Record decisions/game, games/hour and truncation fraction.

## Task B: Deck-pool audit and catalog (1–3 days, depends on the lists)

1. Export the catalog with **all** decks: learner plus gauntlet, sideboards included so commanders are in it.
   ```powershell
   Push-Location xmage
   java -Xmx8G '-Dh2.bindAddress=127.0.0.1' -cp 'lib/*' org.mage.magezero.PolicyCatalogExporter ../artifacts/commander-cards.json decks/<learner>.dck decks/<g1>.dck ...
   Pop-Location
   ```
2. Write a reviewed role annotation for every card in `configs/card-roles.json`. Twenty roles are listed in `docs/statistical-roles-implementation.md`. Roles matter most for the opponents' cards, since they drive the belief over the opponent's hand. Then:
   ```bash
   .venv/Scripts/python.exe scripts/annotate_policy_catalog.py --catalog artifacts/commander-cards.json --annotations configs/card-roles.json --output artifacts/commander-cards-roles.json
   ```
3. **Tokens.** Today a token gets card ID 0 and only flag slot 13. For the decks' tokens, extend `PolicyCatalogExporter` to add the tokens each card can create, by scanning `CreateTokenEffect`s in card abilities, under their token names. Check that the names match `object.getName()` on the battlefield. Winota decks rely on tokens, which are non-Human attackers.
4. Mechanics to check with real games: "look at the top N" choices (Winota's trigger uses `visibleChoices` from the library; confirm the chosen Human is pointed at correctly); copies; stolen or exchanged control; transform/MDFC; face-down cards; reveals (only partially tracked in `PolicyHistory`). Fix only what the pool actually uses.
5. Fresh `mz init --catalog artifacts/commander-cards-roles.json --config configs/policy-final.yml --output artifacts/commander-initial.pt --seed <new>`.

## Task C: Gauntlet opponents in training (about ½ day)

`src/magezero/policy/runner.py`, the generation loop around "Alternate latest-policy self-play and historical league games":
- Today the opponent is the current policy or a league snapshot. XMage's AI (`type: xmage-ai`, already supported by `PolicyGames`) is used only in `_evaluate`.
- Add a third generation kind: **learner deck vs a gauntlet deck piloted by XMage's AI**.
  - New `selfplay` options, for example `--gauntlet deck1.dck deck2.dck ...`, `--gauntlet-every N` or a mixing fraction, and `--gauntlet-ai-skill`.
  - In those generations, `player_a` is the learner deck (`type: policy`) and `player_b` the gauntlet deck (`type: xmage-ai`).
  - Seat swapping must keep the learner on **its own deck**. Today `alternate_seats` swaps controllers between the two decks, so with one learner deck the swap must move the deck with the controller. Check `PolicyGames.play`: it swaps the players but loads `listA` into seat A. Add a config flag so the learner always gets its deck.
  - Only the learner side records trajectories, so there is half the data per game compared with a mirror.
- Mirror and league generations: `player_a` and `player_b` are both the learner deck. Every trajectory is the learner deck, and the swap logic is irrelevant.
- Suggested mix to start: 50% mirror/league, 50% gauntlet, rotating through gauntlet decks. Tune it from the per-deck evaluation.
- Evaluation: report each gauntlet deck separately (also on the lower-priority list) and add a greedy-play option for evaluation.
- Tests: extend `tests/test_long_run.py`'s simulated-engine runner test for the new generation kind and the deck pinning.

## Task D: Launch check, then the final run

Follow `docs/long-run.md` and the original checklist:
1. Re-measure learner throughput on an idle GPU (ask first) with `episodes_per_batch` 16/32. Consider bf16 only if the learner is the bottleneck, and measure it first.
2. Run 1–2 h with the final config and real decks, `--eval-every` small. Record games/hour, decisions/game, GB per generation, error/timeout/truncation fractions, and the learner vs collection balance.
3. Set `--iterations`, `--games`, `--eval-every`, `--keep-every` and `--min-free-gb` from those numbers. 95 GB were free on 12 October.
4. Final run with `--resume` after any interruption. Confirm `best.pt` with a fresh, larger evaluation.

## Task E (optional, recommended): learn from the user's games first (2–3 days)

Purpose: synergy lines such as "build non-Humans, then Winota, then attack with all of them, keeping protection up" are found slowly by random exploration. Imitation gives PPO a good starting point.

- Record a human's decisions in server games as the **same candidate lists** the policy sees. Wrap or observe `HumanPlayer` callbacks (priority, targets, attacks, blocks, mulligans) and build the candidates with the same code as `PolicyPlayer`. Refactor candidate construction out of `PolicyPlayer` into a shared builder.
- Write the trajectories in the existing episode format with the chosen action ID, `log_probability` unused, and a flag marking demonstrations.
- Add `mz pretrain`: behaviour cloning (cross-entropy on the chosen candidate) over the recurrent model, reusing `forward_episodes`. Then PPO from that checkpoint. Keep a small behaviour-cloning or KL term early so PPO does not immediately forget it.
- The hard part: human actions in XMage do not always map 1:1 to the policy's sequential choices, for example mana payment or attacking with several creatures in one click. Map each to the policy's sequence (one attack decision per creature), or skip what does not map.

## Not doing (decided)

- Policy-controlled mana payment.
- Multiplayer Commander (pods).
- Training the policy on decks other than the learner's.
