# MageZero: a search-free recurrent policy for Magic: The Gathering

MageZero trains an agent to play two-player Magic: The Gathering inside the [XMage](https://github.com/magefree/mage) rules engine. The agent chooses directly among the legal actions at every decision, with no game-tree search. It reads a player-relative observation of the game and remembers the game's history, and it estimates the opponent's hidden hand from the publicly known deck list.

The target setting is a Commander Duel: the learner pilots **Winota, Joiner of Forces** against **Kinnan, Bonder Prodigy**, with XMage's own AI as the gauntlet opponent and benchmark. The method is general: any two-player constructed or Commander Duel matchup whose cards the catalog covers can be trained.

This repository is a redesign of [WillWroble/MageZero](https://github.com/WillWroble/MageZero), which trained MCTS-guided networks on sparse hashed features. That pipeline is not part of this code base; it remains available upstream and in this repository's git history.

## Method

| Component | Summary | Details |
|---|---|---|
| Observation | Visible entities (cards, permanents, players, stack objects, hidden-zone counts) with directed relations; public globals; the events since the player's previous decision; the legal candidate actions; the opponent's announced deck with public count bounds. | [Protocol](docs/policy-protocol.md) |
| Actor | Card embeddings plus static card semantics → relation messages → Transformer over entities with a pooling token. New events are pooled per decision into an event GRU, and a decision GRU carries the memory. Every legal candidate is scored against the actor context, with pointers to its source and target entities. | [Architecture](docs/architecture.md) |
| Hidden hand | A closed-form without-replacement (hypergeometric) prior over the opponent's hand from public bounds, plus the probability of holding each of 20 reviewed, overlapping card roles. A learned constrained hand-count model is available as an alternative mode. | [Statistical roles](docs/statistical-roles-implementation.md) |
| Learning | Recurrent PPO on one player's complete trajectories with GAE, a public value head, and a privileged critic that sees the true hand during training only. Optional potential-based reward shaping. | [Architecture](docs/architecture.md#learning-and-collection) |
| Warm start | Behaviour cloning from XMage's AI recorded over the policy's own candidate lists, then PPO with a decaying KL penalty towards the cloned policy (as in AlphaStar). Against XMage's AI, the cloned policy alone outscores 25 generations of PPO from a random start. | [Warm start](docs/warm-start.md#results-the-a5-comparison) |
| Training runs | Resumable self-play generations with a historical league, a gauntlet piloted by XMage's AI, crash recovery, disk retention and periodic evaluation with best-checkpoint selection. | [Long runs](docs/long-run.md) |
| Engine adapter | Java classes compiled against official, unmodified XMage releases: the policy player, observation encoder, history watcher, teacher recorder, batch game runner, catalog exporter and the browser player, plus two optional engine fixes. | [XMage integration](docs/xmage-integration.md) |
| Play | The trained agent in the browser (`mz web`) or in the XMage client (`mz play`). | [Playing the agent](docs/playing-the-agent.md), [Playing in the browser](docs/web-play.md) |

## Repository layout

```text
src/magezero/          Python package (the `mz` command)
  schema.py              observation schema, validation and tensor collation
  catalog.py, roles.py   immutable card catalog and the 20 card roles
  statistics.py          hypergeometric hand prior and role presence
  belief.py              learned constrained hand-count belief (alternative mode)
  model.py               actor (EntityBeliefPolicy) and privileged critic
  checkpoint.py          checkpoint format and loading
  service.py             HTTP inference and recording service used by XMage
  service_client.py      service subprocesses, reload and result counters
  engine.py              persistent XMage engine process and batch configurations
  shaping.py             potential-based reward shaping
  learner.py             recurrent PPO
  teacher.py             teacher-game collection and behaviour cloning
  league.py, runner.py   historical opponents and resumable self-play runs
  diagnostics.py         hand-prediction evaluation and decision traces
  webplay.py, web/       `mz web`: play against a checkpoint in the browser
  cli.py                 command-line interface
integration/xmage/     Java adapter, optional engine fixes and the build script (see docs/xmage-integration.md)
configs/               model, training and game configurations (see configs/README.md)
decks-src/, decks/     the Winota and Kinnan deck lists, and the same lists as XMage .dck files
scripts/               catalog annotation, smoke tests, the final run and evaluation, benchmarks (see scripts/README.md)
tests/                 pytest suite (no XMage needed)
docs/                  design and reference documentation; docs/history/ holds reports of earlier pilots
```

## Installation

Python 3.10 or newer. Install PyTorch for your platform first ([pytorch.org](https://pytorch.org/get-started/locally/)), then the package:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"         # ".[dev]" adds the linter
pytest                           # unit and regression tests; CUDA tests are skipped without a GPU
python scripts/policy_smoke.py --output artifacts/smoke --device cpu
```

The smoke script builds a synthetic catalog and checkpoint, plays two in-process games through the inference service, runs a PPO update, reloads the result and times a forward pass. It needs no XMage installation.

Playing real games requires XMage. The adapter runs on official, unmodified XMage releases: with a JDK 21 on the `PATH`, `python integration/xmage/build.py` downloads the release pinned in `integration/xmage/xmage-release.txt`, compiles the adapter against it and writes the distribution to `xmage/` ([XMage integration](docs/xmage-integration.md), which also covers updating XMage). Below, `ENGINE` stands for the launcher in that distribution: `xmage/mz-policy.sh`, or `xmage/mz-policy.bat` on Windows.

## Playing the trained agent

The trained Winota agent is published as a release asset: [`winota-agent.pt`](https://github.com/ContrattoAlessandro/MTG_AI/releases/download/v0.3.0-alpha/winota-agent.pt) (20 MB, release [v0.3.0-alpha](https://github.com/ContrattoAlessandro/MTG_AI/releases/tag/v0.3.0-alpha)). It pilots Winota, Joiner of Forces in Commander Duel and beats XMage's AI (skill 3) piloting Kinnan, Bonder Prodigy in about 7 of 8 games. To play it:

1. Install the package (above) and build the XMage distribution: `python integration/xmage/build.py` (it also installs the two decks the agent knows, `decks/winota.dck` and `decks/kinnan.dck`, in `xmage/decks/`).
2. Download the weights and start the browser app (http://127.0.0.1:8765):

```bash
curl -L -o winota-agent.pt https://github.com/ContrattoAlessandro/MTG_AI/releases/download/v0.3.0-alpha/winota-agent.pt
mz web --checkpoint winota-agent.pt
```

Choose Kinnan for yourself and Winota for MageZero. [Playing the agent](docs/playing-the-agent.md) has the deck rules, the measured strength and the XMage-client alternative (`mz play`).

## Training the Commander agent

These steps reproduce the published agent. Steps 5 and 6 correspond to `scripts/final-run.sh` and `scripts/final-evaluate.sh`, which produced and selected it (the run took 30 hours on 16 engine threads and an RTX 5070; [Playing the agent](docs/playing-the-agent.md#the-final-run)). On Windows, separate the Java classpath with `;`.

```bash
# 1. Decks: build.py already installed decks/*.dck in xmage/decks/. To regenerate them from the
#    lists in decks-src/ (the commander is the sideboard):
cd xmage
CP="mz/*:lib/*:plugins/*"
java -Dh2.bindAddress=127.0.0.1 -cp "$CP" org.mage.magezero.DeckListConverter ../decks-src/winota.txt decks/winota.dck
java -Dh2.bindAddress=127.0.0.1 -cp "$CP" org.mage.magezero.DeckListConverter ../decks-src/kinnan.txt decks/kinnan.dck

# 2. Card catalog with tokens and other faces (token scan over the release's card sources),
#    then the reviewed roles of every card.
git clone --depth 1 --branch "$(cat ../integration/xmage/xmage-release.txt)" https://github.com/magefree/mage ../xmage-src
java -Xmx8G -Dh2.bindAddress=127.0.0.1 -cp "$CP" org.mage.magezero.PolicyCatalogExporter \
    ../artifacts/commander-cards.json --sources ../xmage-src/Mage.Sets/src decks/winota.dck decks/kinnan.dck
cd ..
python scripts/annotate_policy_catalog.py --catalog artifacts/commander-cards.json \
    --annotations configs/card-roles.json --output artifacts/commander-cards-roles.json

# 3. A fresh policy.
mz init --catalog artifacts/commander-cards-roles.json --config configs/policy.yml \
    --output artifacts/commander-initial.pt --seed 20261013

# 4. Warm start: record XMage's AI playing both decks, then clone it (20 epochs).
mz teacher-collect --checkpoint artifacts/commander-initial.pt --game configs/commander-mirror.yml \
    --engine-script $ENGINE --output artifacts/teacher --games 2000 --engine-threads 16 --ai-skill 1
mz clone --checkpoint artifacts/commander-initial.pt --episodes artifacts/teacher/episodes \
    --output artifacts/commander-cloned.pt --config configs/policy.yml

# 5. Self-play PPO from the clone: Winota mirror and league games, plus a Kinnan gauntlet
#    piloted by XMage's AI; evaluated against XMage's AI every 25 generations.
mz selfplay --checkpoint artifacts/commander-cloned.pt --reference artifacts/commander-cloned.pt \
    --game configs/commander-mirror.yml --config configs/policy.yml --engine-script $ENGINE \
    --output artifacts/commander-run --iterations 1000 --games 64 \
    --gauntlet xmage/decks/kinnan.dck --gauntlet-fraction 0.5 --gauntlet-ai-skill 1 \
    --engine-threads 16 --device cuda --eval-every 25 --eval-games 64 --eval-ai-skill 3 --eval-greedy \
    --keep-every 100 --min-free-gb 30
# After any interruption: the same command with --resume.

# 6. Score candidate checkpoints on fresh games against XMage's AI (no training).
mz evaluate --checkpoint artifacts/commander-run/best.pt artifacts/commander-cloned.pt \
    --gauntlet xmage/decks/kinnan.dck --engine-script $ENGINE --output artifacts/final-evaluation \
    --games 400 --ai-skill 3 --engine-threads 16 --greedy

# 7. Play against the agent in the browser (http://127.0.0.1:8765) ...
mz web --checkpoint artifacts/winota-agent.pt
#    ... or in the XMage client ("Computer - monte carlo" seat).
mz play --checkpoint artifacts/winota-agent.pt --greedy
```

`mz <command> --help` lists every option. [Long runs](docs/long-run.md) explains the run folder (`run-state.json`, `metrics.json` per generation, `evaluations.jsonl`, `best.pt`). `best.pt` is selected on evaluation scores, so its own score is optimistic: confirm the final choice with `mz evaluate` on fresh games.

## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Design choices and their sources, the actor, belief, learning and collection |
| [Policy protocol](docs/policy-protocol.md) | Observation schema, feature layouts, service endpoints, trajectory format |
| [Statistical roles](docs/statistical-roles-implementation.md) | The hand prior and card roles as implemented, and their validation |
| [Statistical belief: research](docs/statistical-belief-and-card-roles.md) | Literature and reasoning behind the statistical prior and roles |
| [Warm start](docs/warm-start.md) | Teacher recording, behaviour cloning, reward shaping and the KL term |
| [Playing the agent](docs/playing-the-agent.md) | The finished Winota agent: how to play it, deck rules, measured strength, the final run |
| [Long runs](docs/long-run.md) | Unattended training runs: recovery, retention, gauntlet and evaluation |
| [XMage integration](docs/xmage-integration.md) | Building the engine, engine patches, catalogs, Commander Duel, human play, coverage limits |
| [Playing in the browser](docs/web-play.md) | `mz web`: starting the app, the table, decisions, keyboard, auto-pass, troubleshooting |
| [Earlier pilots](docs/history/README.md) | Reports and raw measurements of the pilots that preceded the Commander run |

## Scope and limitations

No playing-strength claim is made beyond measured evaluations. The policy depends on a fixed card catalog, so it cannot play cards outside the deck pool it was trained on. Card semantics are coarse features plus learned embeddings rather than rules text. The engine adapter covers the decisions these decks require; [XMage integration](docs/xmage-integration.md#coverage-and-remaining-limits) lists the mechanics it does not cover. Mana payment uses XMage's automatic payment rather than a learned policy.

## Acknowledgements

MageZero was created by Will Wroble ([WillWroble/MageZero](https://github.com/WillWroble/MageZero), [WillWroble/mage](https://github.com/WillWroble/mage)). Games run on [XMage](https://github.com/magefree/mage) by the XMage developers. The design draws on AlphaStar, DouZero/DouZero+, PerfectDou, DeepNash, PPO and GAE; [Architecture](docs/architecture.md) gives the references.

## License

MIT; see [LICENSE](LICENSE). Magic: The Gathering is a trademark of Wizards of the Coast. This project is not affiliated with or endorsed by Wizards of the Coast.
