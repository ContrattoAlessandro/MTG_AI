# Playing the Winota agent

`artifacts/winota-agent.pt` is the finished agent: it pilots **Winota, Joiner of Forces** (`xmage/decks/winota.dck`) in Commander Duel. It is generation 999 of the final run, without optimizer state (SHA-256 `73b28381f4f8de6a2d93ad6bdd12922c33ec3512809daebd99ca1e16f4bd263a`). The `artifacts/` folder is not in git, so keep a copy of the file.

## Playing it

In the browser (no XMage client needed; see [Playing in the browser](web-play.md)):

```bash
mz web --checkpoint artifacts/winota-agent.pt
```

Choose your deck (Kinnan, or Winota for the mirror) and Winota for MageZero in the **New game** window. The browser app plays the most likely action by default.

In the XMage client:

```bash
mz play --checkpoint artifacts/winota-agent.pt --greedy
```

Then create a **Commander Two Player Duel** table in the client and put "Computer - monte carlo" in the opponent seat; that seat is played by the agent. Use `--greedy`: the agent was selected on greedy play, and sampling adds deliberate randomness meant for training.

## Deck rules

- The agent only knows the cards of its catalog: the Winota and Kinnan lists, their tokens and other faces. Your deck must use only those cards, which in practice means `xmage/decks/kinnan.dck` or `winota.dck`.
- Three cards missing from this XMage build were substituted in both lists: Mayor Tong → Ragavan, Nimble Pilferer; Avatar Sanctuary → Breeding Pool; Tidus, Zanarkand Fayth → Bounding Krasis.
- Mana is paid by XMage's auto-tapper, for the agent and in its training.

## How strong it is

Measured on 16 October 2026 with `scripts/final-evaluate.ps1`: 400 fresh greedy games per opponent against XMage's AI (`ComputerPlayer7`) at skill 3, with seats and the starting player alternating. Scores count a win as 1 and a draw as ½; the brackets are Wilson 95% intervals.

| Checkpoint | vs XMage AI, Kinnan | vs XMage AI, Winota (mirror) | Mean |
|---|---|---|---|
| generation 500 | 0.85 [0.82–0.89] | 0.78 [0.74–0.82] | 0.82 |
| `best.pt` (generation 549) | 0.83 [0.79–0.87] | 0.70 [0.65–0.74] | 0.77 |
| generation 600 | 0.87 [0.84–0.90] | 0.77 [0.73–0.81] | 0.82 |
| generation 850 | 0.86 [0.82–0.89] | 0.79 [0.75–0.83] | 0.83 |
| **generation 999 (the agent)** | **0.88 [0.85–0.91]** | 0.77 [0.73–0.81] | **0.83** |

For comparison, with the same opponent at skill 3 but sampled play over 256 games, the behaviour-cloned starting point scored 0.56 against Kinnan and the best 25-generation PPO arm 0.63 ([Warm start](warm-start.md#results-the-a5-comparison)).

- The agent beats XMage's AI piloting Kinnan in about 7 of 8 games. That clears the target set before the run (at least 0.60 against skill 3, with the interval above 0.5) by a wide margin.
- Generations 500 to 999 are statistically tied. The improvement flattened after about generation 450, so generation 999 was chosen for the highest Kinnan score, the matchup the agent is meant for.
- `best.pt` scored 0.895 in its own 64-game evaluation during the run but 0.77 here: it was chosen as the maximum of many noisy evaluations. This is why the final choice comes from fresh games.
- XMage's AI searches simulated copies of the game and may see hidden information; a human plays differently. Expect the agent to be weaker against a strong human, and note what it misplays.

## The final run

`scripts/final-run.sh` (tag `final-run-1`, `configs/final.yml`): 1,000 generations of 64 games from the 20-epoch behaviour clone, half Winota mirror against itself and league snapshots, half against XMage's AI at skill 1 piloting Kinnan, with a KL term towards the clone. It took 30 hours on 16 engine threads and an RTX 5070 (14–16 October 2026), with no failed or truncated games in any saved generation. The run folder is `artifacts/final/run` (`evaluations.jsonl` holds the 64-game evaluation every 25 generations, `provenance.txt` the commit and jar hashes).

Issues seen during the run, all recovered automatically by the supervisor or the runner, and still to be fixed:

- 16 times (generations 431 to 985, more often late in the run), the learner's check that its recomputed action probabilities match the ones recorded during play (tolerance 0.005) failed, and `mz selfplay` exited. The supervisor resumed it and the generation was replayed on fresh games. Likely cause: a CPU (actor) against GPU (learner) rounding difference on a very unlikely sampled action.
- 35 engine restarts within generations. Some `/finish` calls to the opponent's policy service failed and left a session open, so the next `/reload` answered HTTP 400. The replacement service then sometimes could not bind its port while the old process still held it, and the generation was replayed.
