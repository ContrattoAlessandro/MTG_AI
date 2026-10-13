# Statistical hand belief and semantic card roles

Research note and design recommendation (12 October 2026) that led to the default `statistical_roles` belief mode; [Statistical roles](statistical-roles-implementation.md) describes what was implemented.

## Recommendation

For announced opponent deck lists, use a deterministic without-replacement count prior as the next baseline. Feed interpretable, overlapping role summaries alongside card-level information and the existing recurrent history. Remove the learned hand head in that experimental variant. Compare it with a prior plus a small learned history correction before deciding whether an explicit neural belief head earns its cost.

The idea is promising because deck multiplicities and public zone changes are known structure. Our pilot's final joint hand NLL was 3.928 versus 3.958 for the count prior, on only two held-out games. That narrow difference does not establish a useful strategic advantage for the current learned head. It also does not establish that history is unnecessary: a recurrent actor can exploit history even when its explicit probability features come from statistics.

The proposed groups are **semantic roles**, not mutually exclusive clusters. A card may be land access and removal, or a creature body and mana acceleration. The probabilities describe availability of each capability and need not sum to one.

## What the literature supports

| Primary source | What it supports | What it does not establish |
|---|---|---|
| [DouZero+, CoG 2022](https://arxiv.org/html/2204.02558), section IV-A | Predicting opponent per-card counts from observed history, supervised by true cards during training, and excluding impossible counts. | That learning known shuffle combinatorics from scratch is necessary, or that its card categories transfer to Magic. |
| [Bayesian Action Decoder, ICML 2019](https://proceedings.mlr.press/v97/foerster19a.html), [paper](https://proceedings.mlr.press/v97/foerster19a/foerster19a.pdf), section 5.2 | Separating a basic count/constraint belief from inference conditioned on observed actions. | Direct transfer of Hanabi's cooperative communication conventions to an adversarial Magic opponent. |
| [Near Optimal Behavior via Approximate State Abstraction, ICML 2016](https://proceedings.mlr.press/v48/abel16.html) | Abstraction can reduce complexity, subject to conditions on the information it preserves. | That any particular Magic role list preserves values or optimal decisions. |
| [Value Preserving State-Action Abstractions, AISTATS 2020](https://proceedings.mlr.press/v108/abel20a.html) | Discarding distinctions can compromise useful policies; preservation requires specific assumptions. | That two cards with the same coarse label are strategically interchangeable. |

For the role vocabulary, [Mark Rosewater's Mechanical Color Pie 2021](https://magic.wizards.com/en/news/making-magic/mechanical-color-pie-2021) supplies a design vocabulary of effects. It is a historical design snapshot, not a current restriction on which colors or individual cards can have a role. Classification must use actual card abilities.

Reid Duke's [Permission Spells](https://magic.wizards.com/en/news/feature/permission-spells-2015-08-17) supports separating stack interaction from battlefield answers and considering target restrictions. His [Tempo](https://magic.wizards.com/en/news/feature/tempo-2015-07-20) explains why temporary answers differ from permanent removal. [The Basics of Card Advantage](https://magic.wizards.com/en/news/feature/basics-card-advantage-2014-08-25) motivates separating card replacement from net card gain. [When to Cast Your Spells](https://magic.wizards.com/en/news/feature/when-cast-your-spells-2015-08-31) motivates timing and combat relevance. [Inevitability](https://magic.wizards.com/en/news/feature/inevitability-2014-12-08) motivates distinguishing long-term engines from immediate pressure.

These sources inform the vocabulary; **the taxonomy and architecture below are our design proposal**, not a published validated Magic AI architecture.

## The statistical calculation

Let `r_i` be publicly unaccounted copies of card kind `i`, across the unknown hand slots and unknown library. Let `N = sum(r_i)`, `m` be the number of unknown hand slots, and `R_c` be the number of these copies carrying role `c`. Set aside individually known hand cards first; let `k_c` be their count for this role.

Under the assumption that the unknown hand is a uniformly sampled subset of that pool:

```text
P(total role count = k_c + j)
    = choose(R_c, j) choose(N - R_c, m - j) / choose(N, m)

E[total role count] = k_c + m R_c / N

P(at least one role card) = 1, if k_c > 0
                         = 1 - choose(N - R_c, m) / choose(N, m), otherwise
```

This is the [hypergeometric distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html). Out-of-support combinations have probability zero; an empty unknown hand is deterministic. Each role's copies are counted once even when a card has several qualifying abilities or modes.

For an illustration, take 40 unaccounted cards, five unknown hand slots, 20 land-access cards, four counterspell cards, and six spot-removal cards:

| Role | Expected fraction of the hand | Expected copies | P(at least one) | P(at least two) |
|---|---:|---:|---:|---:|
| Land access | 50% | 2.50 | 97.64% | 82.92% |
| Counterspell | 10% | 0.50 | 42.71% | 6.90% |
| Spot removal | 15% | 0.75 | 57.71% | 15.42% |

These are illustrative counts, not predictions for the pilot deck. A 10% expected hand fraction is very different from a 10% chance of holding a counterspell. Presence is often the useful signal for playing around an answer; expected count and P(at least two) help represent repeated interaction. Role overlap means neither expected role counts nor presence probabilities have a sum-to-hand-size constraint.

Do not sum per-card presence probabilities to calculate presence of a role. They overlap as events and are dependent under sampling without replacement. Calculate the union of qualifying copies first. For two different roles, shared cards and competition for hand slots also make their presences dependent.

## Why the prior is not the full posterior

Observed cards impose hard constraints, but observed choices contain additional evidence. Opponents preferentially play lands, cast selected threats, retain counters, and wait for valuable targets. Mulligans select opening hands; scry, filtering, tutors, and choices of discard also change composition. Subtracting public cards and redistributing the rest uniformly ignores these selection mechanisms.

[Constructed mulligans](https://magic.wizards.com/en/news/feature/mulligans-part-iii-constructed-2015-06-29) illustrate the selection of hands for a deck's plan. Its historical rules examples should not be used as a specification of the current mulligan procedure. [Hold Up!](https://magic.wizards.com/en/news/feature/hold-2017-06-15) illustrates why an available card may deliberately remain unplayed.

A full filter would update a distribution through game transitions and weight possible private states by the likelihood of the observed choice:

```text
posterior(private state after action)
  proportional to
  sum over previous private states:
    previous belief × opponent-action likelihood × state-transition probability
```

A numerical toy example makes the distinction clear. Suppose the prior chance of a counter is 40%, and an illustrative opponent passes with open mana with probability 80% when holding a counter versus 20% otherwise. Bayes' rule gives `0.8*0.4 / (0.8*0.4 + 0.2*0.6) = 72.7%`. Those likelihoods are invented for illustration, not a calibrated Magic model. Real passes may represent card draw, removal, a bluff, or an irrelevant counter.

I recommend testing statistics first, while retaining history in the actor. A small learned correction should be an evidence-driven addition. An explicit full Bayesian opponent-policy filter is more involved and need not be the initial implementation.

## Proposed role taxonomy

Use twenty broad roles, with the final deck-specific role enabled only when annotations support it. Store exact card types separately. Counts are of physical cards capable of supplying a role, not numbers of effects.

| ID | Role | Meaning and useful refinements |
|---|---|---|
| land_access | Land access | A card can be played as a land from the relevant zone. Include a legal land face; record colors, enters-tapped conditions, and utility abilities separately. |
| mana_acceleration | Mana acceleration / fixing | Permanent or temporary mana, extra land deployment, and color fixing. Record which resource increases, conditions, and when it becomes available. |
| board_presence | Board presence | Creature bodies, token production, and other deployable combat presence. Refine offense/defense, haste, evasion, size, and flash; a creature is not automatically a meaningful threat. |
| repeatable_engine | Repeatable engine | A deployed card can repeatedly produce material advantage or pressure. Record the trigger/resource and relevant synergies; do not label every one-shot ETB effect an engine. |
| card_draw | Card access / draw | Draw or temporary access to cards. Distinguish one-card replacement, net gain, repeatability, expiry, and conditional requirements. |
| card_selection | Card selection | Scry, surveil, looting, rummaging, and choosing among looked-at cards. Record look depth and net card change separately. |
| tutor | Search / tutor | Searches for a restricted or unrestricted card. Record destination, reveal status, eligible targets, and whether it advances mana or a combo. |
| counterspell | Stack interaction | Counters or otherwise stops spells/abilities on the stack. Refine spell versus ability, allowed spell types, mana-value limits, taxing versus unconditional answers, and alternative costs. |
| spot_removal | Individual permanent answer | Destroy, exile, sacrifice, damage, shrink, control change, or sustained disablement affecting individual permanents. Preserve mechanism and target restrictions. |
| sweeper | Broad board answer | Mass destruction/exile/damage/bounce/disablement. Refine affected types, asymmetry, thresholds, and whether the effect is temporary. |
| tempo_interaction | Temporary disruption | Bounce, tapping, freezing, phasing, or temporary theft that changes immediate availability. Track duration and whether ETB reuse or replay remains possible. |
| direct_reach | Direct reach / alternate pressure | Damage or life loss to the opponent, or an annotated mill/alternate-win route. Distinguish amount, repeatability, and whether the route is actually relevant to this deck. |
| combat_modifier | Combat modifier | Pump, evasion, strike changes, forced blocks, or team buffs. Record whether the effect is available during combat and its possible swing; permanent buffs are distinct from tricks. |
| protection | Protection / rescue | Hexproof, protection, indestructibility, countering an answer, blink, or self-bounce that can preserve a resource. Record what it actually protects against. |
| hand_disruption | Hand disruption / information | Targeted or mass discard and hand inspection. Refine randomness, selection, and eligible card types; self-discard belongs to selection or synergy annotations. |
| recursion_cheating | Recursion / bypassing deployment costs | Return cards from a graveyard, reanimate, or deploy objects through another route. Record origin, destination, target restrictions, and required supporting resources. |
| graveyard_hate | Graveyard interaction against the opponent | Exile graveyard cards or prevent graveyard access. Refine one card versus broad effects, timing, and replacement versus removal. |
| persistent_restriction | Persistent restriction | Taxes, casting/activation restrictions, attack restrictions, resource denial, or replacement effects. Record the specific prohibited or modified action. |
| survival | Survival / damage mitigation | Life gain, prevention, fog-like effects, and loss prevention. Distinguish immediate combat prevention from a life cushion or a conditional permanent. |
| deck_synergy | Deck-specific enabler / payoff | Annotated sacrifice outlets, combo pieces, tribal/counter/artifact support, and other plan-specific roles. Keep enabler and payoff identifiers, partner requirements, and combo membership. |

Role membership often overlaps. A sweeper may also provide spot removal in a mode; a counter can protect a threat; a damage spell can supply removal and reach. These roles are capabilities, not a recommendation for how to spend a card.

The broad list is intentionally extensible. Copy effects, extra turns, energy generation, equipment, land animation, and unusual alternate wins need effect descriptors and deck-specific refinements. There is no universal fixed list that fully captures every Magic card.

## Preserve the distinctions that change decisions

1. **Object types:** land, creature, instant, sorcery, artifact, enchantment, planeswalker, battle, and applicable multiple types/faces. These are rules facts, not substitutes for roles.
2. **Timing:** sorcery timing, instant/flash timing, activated or triggered route, originating zone, and window restrictions.
3. **Costs:** colored pips, generic requirements, X, taxes/reductions, alternative costs, life, sacrifice, discard, and other conditions. Printed mana value alone does not establish affordability.
4. **Targets and mechanism:** legal target kinds, power/toughness/mana-value restrictions, damage threshold, destroy versus exile versus sacrifice, color restrictions, and temporary versus persistent effect.
5. **Conditions and magnitude:** supporting creatures, graveyard content, resources, threshold abilities, conditional draws, and numerical effect strength.
6. **Synergy:** enabler/payoff links, combo partners, tribal support, and deck plan. A generic `combo` probability is too coarse to mean imminent combo availability.

The current [Comprehensive Rules, 25 September 2026](https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt) distinguish alternative costs (118.9), flash (702.8), modal choices (700.2), and double-faced cards (712). These justify handling a card's separate action routes rather than inferring usability from card type and mana value alone.

Examples checked against this workspace's XMage source:

| Card | Why one exclusive group fails |
|---|---|
| Llanowar Elves | Creature body and mana acceleration. Newly deploying it does not imply immediate access to its tap-for-mana ability. |
| Brazen Borrower | Flash creature presence plus an instant-speed bounce Adventure. The available route depends on its current zone/history. |
| Boseiju, Who Endures | Land access plus a channel ability answering artifacts, enchantments, or nonbasic lands. The channel route is an ability rather than a spell. |
| Negate | Counterspell role, but only noncreature spells qualify. A probability of `counterspell` alone overstates the risk to a creature spell. |
| Growth Spiral | One-card replacement plus conditional extra land deployment; acceleration depends on having an eligible land. |

See the local implementations in `../.mz_engine/Mage.Sets/src/mage/cards/`. Source effects and target filters are preferable to a keyword-only text classifier. However, effects can be wrapped, conditional, modal, or custom, so engine inspection alone does not guarantee complete automatic classification.

## What should enter the actor

For each role, start with four quantities: expected count, P(at least one), P(at least two), and probability of a qualifying card whose route appears usable from public resources in the specified window. Twenty roles yield about 80 scalar features, plus optional masks identifying unsupported or uncertain availability checks. Feed these through a small encoder into the recurrent actor context.

Call the last quantity **public-resource availability**, not exact private-state legality. Mana restrictions, alternative costs involving other hidden cards, and joint requirements can prevent an exact check from public information alone. Keep unknown conditions marked; never query the opponent's real hidden hand to produce actor features.

For a candidate action, add focused queries: probability of an affordable answer to this spell, removal applicable to this creature, protection against this removal, or an effective trick in this combat. Under the statistical prior, a fixed public predicate defines a subset of eligible cards and can reuse the same hypergeometric presence calculation. Conditions involving several hidden cards require a joint calculation or an explicitly approximate estimate. Existing battlefield answers are observed features, not hidden-hand probabilities.

Keep card-level information through deck-card tokens containing identity, effect descriptors, remaining counts, known-hand counts, and statistical marginals. Pool or attend to those tokens alongside role summaries. They describe the announced possibilities, not a guessed enemy hand. A rare relevant card must remain distinguishable from the rest of its role.

No role softmax is appropriate: overlapping roles can all be present. Joint distributions for combos or multiple answers cannot be reconstructed by multiplying independent role presences. Modal cards also cannot supply all their modes simultaneously after being spent once.

## Optional learned correction

If the statistics-only variant misses useful history information, add a small residual count scorer:

```text
log score_i(n_i) = log choose(r_i, n_i - known_i)
                  + history_residual_i(n_i)
```

Use the existing count constraints and normalization. Zero residual recovers the combinatorial prior. A bounded or regularized correction can learn retention, selection, and passing patterns while preserving impossible-state zeros. The residual is not guaranteed to equal a Bayesian opponent-action likelihood. Group features should be derived consistently from the resulting joint model; summing marginal expectations is valid, but summing marginal presence probabilities is not.

This option retains the count-DP cost. Independent learned role heads would be cheaper but could produce mutually inconsistent counts and probabilities; they should not be presented as an exact hand posterior.

## Information tracking and evaluation

Maintain zone and knowledge transitions, not a monotonically growing list of played cards. Public bounce to hand creates a known hand card; subsequent departure removes or weakens that knowledge. Track draws, discard, mill, reveal, tutors, shuffle, theft, generation, and transformed/adventure routes. A shuffle resets library-order knowledge, not every fact about cards currently known in hand. Known top/bottom cards can make library positions nonexchangeable. The statistical formula must either track those constraints or flag its approximation. Announced deck lists must describe the actual post-sideboard game deck.

Generated or stolen cards and private pool expansion remain an integration gap in the present adapter. Adding a role layer does not fix that gap or permit using private labels to expand actor input.

Compare four fresh-seed variants: current learned head; statistics with card-level inputs; statistics with card-level inputs and roles; and prior-plus-residual with the same role inputs. Retain the same history, observation information, matchups, reward, and evaluation opponents. Report both equal-game and equal-time results, initialization seeds, model sizes, and compute. Use diverse matchups exercising counters, removal, combat tricks, and synergy; the simplegreen pilot alone cannot test all these roles.

Measure full-hand NLL where defined, role-presence Brier score and calibration, count error, action latency, learner throughput, and wins against fixed opponents. Evaluate across independent games; states within a game are correlated. Include rare but decisive capabilities rather than optimizing only average prediction error. Better hand prediction need not improve decisions, and a smaller input need not improve throughput until measured.

The existing implementation's serial GPU behavior is a separate bottleneck. Removing the learned head may save work, especially its DP, but no new speedup is claimed without a matched benchmark.
