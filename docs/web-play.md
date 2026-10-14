# Playing against the bot in the browser

`mz web` runs a local web app where you play a game of Magic against a trained MageZero checkpoint. You need no XMage client, but you do need the XMage distribution in `xmage/` (see [XMage integration](xmage-integration.md#building-the-distribution)) and a JDK (`javac` and `java` on the PATH).

## Starting it

```bash
mz web --checkpoint artifacts/anchors/a5-b-gen24.pt
```

This starts three things and opens http://127.0.0.1:8765 in your browser:

1. the bot: a policy service for the checkpoint on port 50071 (inference only, nothing is recorded);
2. the rules engine: an XMage JVM (`PolicyGames --serve`) that plays one game at a time;
3. the web server, on `127.0.0.1` only.

The first start compiles the Java adapter into `artifacts/web-engine/classes` (about 20 s); later starts recompile only when a source changed. Booting the engine takes up to a minute; the page shows "Preparing the table" meanwhile. Ctrl+C stops everything.

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint` | (required) | the policy that plays the bot |
| `--port` | 8765 | web server port |
| `--policy-port` | 50071 | the bot's policy service port (use a free one if a training run is up) |
| `--device` | cpu | `cuda` to run the bot on the GPU |
| `--xmage` | xmage | XMage distribution folder |
| `--heap` | 4G | engine Java heap |
| `--no-browser` | off | do not open a browser window |

It does not touch the jars in `xmage/lib` or the training ports, so it can run beside a training run (it competes for CPU).

## Starting a game

The **New game** window lists the decks in `xmage/decks/` whose cards are all in the checkpoint's card catalog (for the Commander agent: Winota and Kinnan).

- **Your deck** and **MageZero's deck**: two 99-card decks with a commander play a Commander Duel (40 life, 21 commander damage); other decks play a normal two-player game.
- **Who goes first**: random, you, or MageZero.
- **MageZero plays**: *Its best move* (the policy's most likely action; its strongest play) or *Varied* (sampled from the policy, so it plays differently each game).

The bot was trained as Winota against Kinnan, so playing Kinnan against the Winota bot is the intended matchup.

## The table

- **Top**: MageZero's seat panel (commander art, life, hand, library, graveyard and exile buttons, command zone, cards in hand as card backs) and battlefield.
- **Middle bar**: turn number and whose turn it is, the phase tracker (the current step is lit in the active player's colour), and the stack (top first; hover an item to see its targets).
- **Bottom**: your seat panel, your battlefield and your hand. Battlefields put creatures in the front row and other permanents and lands in the back row. Tapped cards turn sideways, identical lands and tokens stack with a ×N count, auras and equipment sit under their creature, and badges show power/toughness (red when damaged), counters, summoning sickness, attacking and blocking.
- **Right**: the decision panel, the auto-pass controls and the game log (written to you; new lines flash; it follows the game unless you scroll up).

Hover any card for a large image and its current state. Card images come from Scryfall (internet needed; cards without an image show a text frame). When MageZero casts a spell it is shown in the middle of the table for a moment.

## Making decisions

Whenever the engine needs you, the decision panel lights up gold with the question and your options. Cards and players you can act on glow on the table; you can click them instead of the buttons (a card with several options opens a small menu). Choices with only one legal option are not asked.

| Decision | How it looks |
|---|---|
| Priority | your playable spells, abilities and land drops, plus **Pass priority** |
| Mulligan, yes/no | two large buttons |
| Targets, modes, choices | one button per option, with card art; **Done choosing** when you may stop |
| Attackers | asked one creature at a time: **Attack MageZero** or **Don't attack**; **All attack** / **No more attacks** answer the rest of this combat |
| Blockers | asked one blocker at a time: one button per attacker it can block, or **Don't block**; **No blocks** skips the rest |
| Numbers (X costs) | a slider and a number box for long ranges |

Mana is always paid automatically (XMage's auto-payment), for you and the bot. Some sources still ask a question while paying, such as which colour Command Tower should make.

### Keyboard

| Key | Action |
|---|---|
| Space / Enter | the gold button: pass priority, don't attack, don't block, keep… |
| 1–9 | the numbered option |
| E | pass priority until the end of this turn |
| T | pass priority until your next main phase |
| Esc | close a window or menu |

Keys are ignored for a moment after a new decision appears, and for a little longer after a new turn starts, so a key meant for the previous situation is not applied to a new one.

### Auto-pass

Like XMage, the page passes priority for you when you would rarely act:

- **Auto-pass on an empty stack outside my main phases** (on by default): you stop in your own main phases and whenever something is on the stack, so you can respond to the bot.
- **Auto-pass while my own spell or ability is on top of the stack** (on by default).

Turn either off to stop at every priority (for instant-speed play in the bot's turn). **Pass turn** and **To my turn** pass until the end of the turn or your next turn; **Stop** cancels them.

## Ending a game

**Concede** ends the game as a loss. **New game** starts another one; a game in progress is abandoned. When a game ends, a result window offers **Play again** or **View the board**.

Only one browser tab plays at a time: the most recently opened one. Others show "Playing in another tab" with a **Play here** button.

## Troubleshooting

- **"The game stopped"**: the engine hit a rules error. The window names the engine log; logs are in a temporary `mz-web-*` folder printed when `mz web` starts (`engine.log`, `policy.log`).
- **A decision seemed to be skipped**: `http://127.0.0.1:8765/api/history` lists the recent decisions the engine asked and the answers it received; in the browser console, `mzTrace` lists the answers the page sent and why (click, key, auto-pass).
- **Port in use**: pass `--port` or `--policy-port`.
- **No card images**: Scryfall is unreachable; the game still works with text frames.

## How it works

Your seat is `WebPlayer` (`integration/xmage/src/mage/player/ai/policy/WebPlayer.java`), a `PolicyPlayer` whose decisions go to the browser instead of the policy service. You therefore choose among exactly the candidates the policy is offered (`PolicyCandidates`), with readable labels. For each decision the engine posts the options and a snapshot of what you may see (never the bot's hand or library) to `/engine/decide`; the server holds the request until you answer. The page long-polls `/api/state` and answers through `/api/answer`. The server is `src/magezero/webplay.py` and the page is `src/magezero/web/index.html` (plain HTML and JavaScript, no build step).
