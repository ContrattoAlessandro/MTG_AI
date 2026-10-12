# Entity policy protocol, version 2

`src/magezero/policy/schema.py` is the executable specification. JSON requests use finite numbers and exact fields. Entity/action IDs are opaque references; the neural network never embeds their strings. Observations are player-relative: owner/controller/actor `0=self`, `1=opponent`, `2=neutral`.

## Endpoints

`GET /healthz` returns schema and checkpoint policy versions. `GET /catalog` returns immutable catalog contents and digest. `/act` requires exactly:

```json
{"observation": {"schema_version": 2}, "sample": true}
```

The example only illustrates the envelope; the observation must contain every field below. `sample=true` samples the stochastic legal policy; `false` uses its argmax for inspection and is excluded from training recording. Responses include `action_id`, `log_probability`, public `value`, legal action probabilities, and belief card IDs/count marginals/expected counts. Values do not drive search. A service started with `--no-belief` omits the diagnostic `belief` object (about 97% of response bytes); the self-play and evaluation runners use it because the engine reads only `action_id`. `GET /stats` reports request, inference and batch counters. A service started with `--allow-reload` also accepts `POST /reload` with exactly `{"checkpoint": path, "episodes": path or null}`: it loads the checkpoint (same card catalog required), switches the episode directory, and restarts the counters. It is refused while any game session is active, so a recurrent game never spans two policies.

Concurrent `/act` requests from different sessions are evaluated together: one inference worker takes every decision queued while it was busy (up to `--max-batch`, default 32) and runs a single padded forward pass, with no added wait when the service is idle. A retry that arrives while its decision is still being computed receives the same result; `/finish` is rejected while a decision is in flight.

The observation schema remains v2 for both architectures. Statistical-role checkpoints use checkpoint format v3 and catalog format v2. `belief.mode` identifies `learned` or `statistical_roles`. Statistical responses additionally contain `belief.roles`, mapping each stable role name to `probability_at_least_one` and `expected_count`. Presence probabilities overlap and do not sum to one. These are possession probabilities under the count prior, not probabilities of casting an effect or having a currently legal response.

Role catalogs add a fixed `role_names` ordering and a `roles` list to each card. `scripts/annotate_policy_catalog.py` attaches reviewed annotations while preserving card IDs; roles change the digest, so initialize a fresh policy and collect fresh trajectories. Catalog v1 and pre-existing checkpoint v2 files remain supported in `learned` mode.

An observation contains `schema_version`, `catalog_digest`, `game_id`, `player_id`, zero-based `step`, `entities`, `globals`, `events`, `actions`, `opponent_deck`, `opponent_hand_size`, and optional `relations`.

| Collection | Fields |
|---|---|
| Entity | `id`, `card_id`, `kind`, `zone`, `owner`, `controller`, `visibility`, `features[32]` |
| Relation | `source`, `target`, `type` |
| Event | increasing `seq`, `type`, `actor`, `card_id`, `features[8]` |
| Action | `id`, `type`, `source`, `target`, `card_id`, `features[16]` |
| Opponent deck entry | unique positive `card_id`, public `lower`, public `upper` |

Entity kinds used by the adapter: `0=player`, `1=permanent`, `2=card`, `3=stack object`, `4=hidden aggregate`; 5–7 are reserved. Zones: `0=unspecified`, `1=battlefield`, `2=hand`, `3=graveyard`, `4=exile`, `5=stack`, `6=library`, `7=command`. Relation types emitted: `0=attachment → host`, `1=stack object → target`, `2=attacker → defender`, `3=blocker → attacker`; 4–7 are reserved for additional public links.

Visibility is `public`, `own`, or `hidden`. Own private entities require owner 0. Hidden aggregate entities have card ID 0 and only quantity feature 7. A face-down permanent uses public visibility with identity 0 and its rules-visible attributes. Opponent hand identities do not appear as entities, apart from explicitly inspectable engine choice sets or public information. The Java encoder, not JSON validation alone, must enforce game visibility.

Action types are `pass`, `cast`, `activate`, `target`, `use`, `mode`, `attack`, `block`, `order`, `amount`, `mulligan`, `mana`, `discard`, `choose`. Event types are `draw`, `reveal`, `zone_change`, `cast`, `activate`, `resolve`, `attack`, `block`, `damage`, `life`, `pass`, `shuffle`, `choice`. Opponent/neutral draw events must have card ID 0. Not all accepted types are emitted by the initial adapter.

## Feature layout in the Java adapter

Unused slots are zero. Divisors normalize inputs rather than impose game limits.

| Vector | Slots |
|---|---|
| Card semantics, 32 | 0–7 land/creature/instant/sorcery/artifact/enchantment/planeswalker/battle; 8 mana value /16; 9–13 W/U/B/R/G; 14–15 power/toughness /20; 16–31 flying/first strike/double strike/deathtouch/lifelink/trample/vigilance/haste/hexproof/ward/indestructible/reach/flash/menace/defender/protection |
| Entity, 32 (objects) | 0 power /20; 1 toughness /20; 2 mana value /16; 3 tapped; 4 summoning sickness; 5 damage /20; 6 counters other than +1/+1, −1/−1 and loyalty /20; 7 +1/+1 counters /20; 8 attacking; 9 blocking; 11 −1/−1 counters /20; 12 loyalty counters /20; 13 token; 21–25 creature/land/artifact/enchantment/planeswalker; 26 transformed; 31 copy. Current keywords, including abilities gained during play: 10 flying, 14 first strike, 15 double strike, 16 deathtouch, 17 lifelink, 18 trample, 19 vigilance, 20 haste, 27 hexproof (any), 28 indestructible, 29 reach, 30 menace |
| Entity, 32 (players) | 7 hand size /60; 10 lands played this turn /4; 11 life /40; 12 poison /10; 13 energy /20; 14 land drops left this turn /4. Hidden aggregates use only 7 (quantity) |
| Global, 24 | 0 turn /100; 1 self active; 2 stack /16; 3–4 life self/opponent /40; 5–6 hand sizes /60; 7–8 libraries /60; 9 phase ordinal /32 (0 before turns); 10–15 self W/U/B/R/G/colorless pool /20; 16–21 opponent pool /20 |
| Action, 16 | 0 argument/ordinal /64 for every type, then per type: priority 1 payable mana value /16, 3–15 lexical bins of the ability text; target/use/mode/order/choose (named) 3–15 lexical bins of the prompt or option text; block 1 blockers already on that attacker /4, 2 attacker's minimum blockers (menace 2) /4, 3 attackers this blocker already blocks /4; pile 2 size /60, 3 lands /8, 4 nonlands /8, 5 creatures /8, 6 total mana value /32, 7 highest mana value /16, 8 creature power /20, source/card ID = its highest mana-value card; mulligan 1 keep flag, 2 mulligan flag, 3 hand size /7, 4 lands /7, 5 spells /7, 6 colour-fit spells /7, 7–8 spells castable by turn 2/3 /7, 9–13 spell mana-value buckets 0–1/2/3/4/5+ /7, 14 mulligans taken /7, 15 on the play. On the keep option 3–13 describe the current hand; on the mulligan option 3 is the next hand size and 4–5, 9–13 the expected counts of a uniform redraw from the own deck (6–8 zero). Lexical bins use Java `floorMod(token.hashCode(),13)` followed by `tanh(count/4)` |
| Event, 8 | 0 amount /20; 1 turn /100; 2–3 from/to zone /7 for zone changes. Observer choice events copy the first eight action features. |

Card features are a coarse catalog baseline; keyword substring extraction does not parse Oracle rules. Entity keyword slots read the object's current abilities, so gained keywords are visible. Tokens are flagged but get card ID 0 unless the catalog contains their name; transformed faces, continuous effects other than keywords, and trigger context still need richer encodings. These layouts changed on 12 October 2026 without a shape change: checkpoints initialized earlier load, but read the new slots with old meanings, so initialize a fresh policy.

## Recurrent and training lifecycle

Steps must be contiguous for one player, starting at zero. The adapter does not send decisions with exactly one legal action; they consume no step, and their choice event arrives with the next observation. Events contain only new entries, in increasing sequence order; empty history is valid. Retrying the identical last request returns the same action without sampling again or advancing memory. Changing a retry fails. Services cap active sessions, steps, body size, and inactivity. The protocol does not support resuming an expired live game.

After the game, `/finish` receives exactly:

```json
{
  "game_id": "unique-game",
  "player_id": "observer",
  "result": 1,
  "terminal": true,
  "labels": [{"opponent_hand_counts": [1, 0, 2]}]
}
```

There is one label per recorded decision, with counts in that observation's `opponent_deck` ordering, matching its hand size and public bounds. Labels are absent from `/act`. Completion retries are idempotent. An abort/turn limit uses `terminal=false` and an empty label list, releases memory, and records no training episode. A real draw uses `terminal=true,result=0`.

Training writes one JSONL episode per player: `schema_version`, frozen `policy_version`, `terminal=true`, and `steps`. Each step has observation, selected action ID, behavior log probability, separate labels, and reward. Earlier rewards are zero; the final reward is the game outcome. An episode cannot mix players, skip steps, repeat events, or use stale checkpoint data.

Old hashed-state HDF5 files, feature vocabularies, `.pt.gz` checkpoints, fixed-head policies, and MCTS targets do not convert to this representation. Start a new catalog/checkpoint and recollect games.
