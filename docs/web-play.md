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

- **Left**: the seat panels (commander art, life, hand, library, graveyard and exile piles, command zone; MageZero's hand as card backs).
- **Centre**: MageZero's battlefield above yours, and your hand. Battlefields put creatures in the front row and other permanents and lands in the back row. Tapped cards turn sideways, identical lands and tokens stack with a ×N count, auras and equipment sit under their creature, and badges show power/toughness (red when damaged), counters, summoning sickness, attacking and blocking.
- **The stack**: a floating panel (top right of the board; drag its header to move it, double-click to put it back) that appears while something is on the stack. The bottom of the stack is on the left, the item that resolves next is on the right, marked **Next**. Spells show their card; abilities show their source, whether they are triggered or activated, their text and their targets.
- **Right**: the decision panel, the auto-pass controls and the game log (written to you; new lines flash; it follows the game unless you scroll up), then the match flow: whose turn it is, who has priority, and the steps of the turn with the current one lit in the active player's colour (blue for you, red for MageZero) and the steps already played dimmed.

### What a stack item points at

Lines drawn from the stack show what the top item (or the one under the mouse) interacts with:

- **gold line and number**: the permanent or card that put the ability on the stack. Every source on the stack gets a number, shown on the stack item and on the source, so several triggers of one card share a number;
- **violet arrow with a ring**: its targets (cards, players, other spells on the stack);
- **dashed violet line**: what it affects without targeting, such as "that creature" in a trigger.

A card that is not on the table (in a graveyard, in exile, in a hand or library) is pointed at through its pile, which lights up.

Cards revealed by either player, or that only you look at, open in a window as it happens (several queue up: **Next**, then **Close**).

Hover any card for a large image and its current state. Card images come from Scryfall (internet needed; cards without an image show a text frame). When MageZero casts a spell it is shown in the middle of the table for a moment.

## Making decisions

Whenever the engine needs you, the decision panel lights up gold with the question and your options. Cards and players you can act on glow on the table; you can click them instead of the buttons (a card with several options opens a small menu). Choices with only one legal option are not asked.

- **Activating an ability** of a permanent: when you have priority, a permanent with an ability you can activate now glows; click it (or its line in the decision panel). Mana abilities are not offered here: you tap sources while paying a cost (below), so a plain land does not glow.
- **Choosing targets**: after you cast or activate something that targets, the panel asks for each target; legal targets glow on the table. A graveyard or exile pile that holds legal targets glows too: open it and click the card.

| Decision | How it looks |
|---|---|
| Priority | your playable spells, abilities and land drops, plus **Pass priority** |
| Mulligan, yes/no | two large buttons |
| Targets, modes, choices | one button per option, with card art; **Done choosing** when you may stop |
| Attackers | the whole attack at once: click your glowing creatures to attack with them (a chosen one turns a little and burns brighter; click it again to hold it back). A creature that can also attack a planeswalker opens a menu. **All attack** chooses every one. **Pass priority** (Space) declares the attack; with nothing chosen, no creature attacks. A creature that attacks lunges forward in sparks and flames (no particles when the system asks for reduced motion) |
| Blockers | the whole block at once: click one of your glowing creatures, a line follows the pointer, then click the attacker it blocks; the line stays. Click a blocker again to aim it elsewhere (click the table or Esc to leave it unblocking). **Pass priority** (Space) declares the blocks. When they break a restriction (menace), you are asked again with your blocks kept. During combat, lines join each blocker to its attacker (blue yours, red MageZero's) |
| Numbers (X costs) | a slider and a number box for long ranges |
| Choices among cards (a library search, "look at the top six…") | a window with the cards: those you can choose glow, the others are dimmed; **Done choosing** when you may stop. **Hide** (or Esc) shows the table; **Show the cards** in the decision panel brings them back |
| Cards you look at with nothing to choose (Winota without a Human among the six) | the same window with **OK** |
| Naming a card, or another choice among many names (a creature type) | a search box: type part of the name (accents and punctuation do not matter), ↑/↓ and Enter or click a result. While the box is empty it lists the cards of this game. Every name XMage offers can be chosen (the bot only considers the cards of its catalog) |
| Several of your triggers at once | they wait under the stack panel: click them in the order they go on the stack (the last one placed is on top and resolves first; 1–9 place them too), drag placed ones to reorder or click one to take it back, **Place all** for the rest, then **Confirm order** (Space). The whole order is sent at once and the engine then puts them on the stack in that order |

### Paying mana

Whenever you owe mana (casting a spell, activating an ability, a "pay {1}?" trigger), mana already in your pool is used first; for the rest the panel shows **Pay {2}{R}** and you choose how:

- **By hand**: click a glowing source on your battlefield, or its line in the panel (identical lands are listed once, with ×N). A source with several mana abilities, such as a dual land, opens a menu of them. A source that makes mana of any colour (Command Tower) then shows its colours as mana symbols. You are asked again until the cost is paid. Only sources whose mana can go towards the cost are offered.
- **Auto pay** (Space): the rest of the cost is paid for you. For each part of the cost it taps an artifact if one can pay it, else a land, else another permanent (a mana creature), and only last a source that is sacrificed for mana (Treasure, Lotus Petal) or a card in your hand (Simian Spirit Guide). Colours of any-colour sources are picked to fit the cost. When none of those can pay, XMage's own payment tries everything, including Phyrexian mana with life and convoke. If it still cannot finish, the panel says so and you pay the rest by hand or cancel.
- **Cancel**: the spell or ability is not cast and everything is undone; you can cast it again in the same window.

The toggle **Always auto pay mana costs** answers every payment with Auto pay (except one Auto pay could not finish). The bot always uses XMage's automatic payment.

### Keyboard

| Key | Action |
|---|---|
| Space / Enter | the gold button: pass priority (also declares the chosen attackers or blockers), keep… |
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

Your seat is `WebPlayer` (`integration/xmage/src/mage/player/ai/policy/WebPlayer.java`), a `PolicyPlayer` whose decisions go to the browser instead of the policy service. You therefore choose among exactly the candidates the policy is offered (`PolicyCandidates`), with readable labels. For each decision the engine posts the options and a snapshot of what you may see (never the bot's hand or library) to `/engine/decide`. Each stack item in the snapshot names its `source` and its `targets`, each with its zone and owner: the targets of every chosen mode (`kind: "target"`) and the objects its effects' target pointers name (`kind: "related"`), which is what the page draws its lines from; the server holds the request until you answer. The page long-polls `/api/state` and answers through `/api/answer`. The server is `src/magezero/webplay.py` and the page is `src/magezero/web/index.html` (plain HTML and JavaScript, no build step).
