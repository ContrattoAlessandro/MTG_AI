# Statistical roles: the default hand belief

This document describes the default `statistical_roles` belief mode as implemented and validated. It realizes the [research proposal](statistical-belief-and-card-roles.md) as a deliberately small first variant: deterministic hand-count statistics, reviewed overlapping roles, card identity, and recurrent history. It does not implement learned history corrections or context-dependent affordability/target queries.

## Architecture

```mermaid
flowchart LR
    D[Known deck and public count bounds] --> P[Without-replacement prior]
    P --> C[Card count marginals]
    D --> R[Role-copy counts]
    T[Reviewed catalog roles] --> R
    R --> S[20 probabilities of at least one]
    C --> B[Card summary encoder]
    S --> E[Linear role encoder and SiLU]
    B --> A[Actor context]
    E --> A
    H[Entity attention and persistent history] --> A
    A --> L[Legal candidate scores]
```

The default is `ModelConfig.belief_mode = statistical_roles`. There is no trainable `belief_head` and no count DP in this path. `statistics.py` calculates hypergeometric card marginals and role presence from public counts. Computation is vectorized across cards and roles, in float64 for combinatorial arithmetic followed by float32 features. An empty pool/hand, known cards, zero-capacity padding, infeasible bounds, and large public pools are handled explicitly.

Each role supplies **one actor input: probability of holding at least one qualifying card**. A linear 20-to-256 encoder with SiLU adds its result to the existing card-summary vector. Expected role counts are returned for inspection but do not add another actor input. The existing card summaries retain learned card embeddings, expected card counts, card-presence probabilities, and uncertainty. Entity attention, the event GRU, decision GRU, action scorer, and critics remain in place.

Let `l_i` be known hand copies and `u_i` publicly remaining copies, including those known copies. Unknown pool counts are `r_i = u_i - l_i`, pool size `N = sum(r_i)`, and unknown hand slots `m = hand_size - sum(l_i)`. For a role with `R` qualifying unknown copies and no known qualifying card:

```text
P(holding at least one) = 1 - choose(N - R, m) / choose(N, m)
```

If a known hand card supplies the role, presence is one. Overlapping roles reuse physical cards across categories, but count a copy only once within a category. The implementation computes a union of qualifying copies; it never sums per-card presence probabilities or normalizes roles with a softmax.

For example, four counterspell copies among 40 unknown cards and five unknown hand slots produce presence 0.4271, not 0.10. Twenty land-access copies produce 0.9764. These are prior probabilities under exchangeable sampling, not exact posteriors conditioned on strategic retention, selection, and mulligans. History remains in the actor for those patterns.

## Role metadata and coverage

The 20 stable role names are defined in `src/magezero/roles.py`. Catalog format 2 adds these names in fixed order plus reviewed role labels on every card. `configs/card-roles.json` records 187 reviewed entries, each with its XMage source reference and notes. They cover the whole Winota and Kinnan Commander pool (including tokens and other faces; written by `scripts/commander_roles.py` from the XMage implementation of each card), every card of the 10-kind `simplegreen` validation deck, and several interaction examples.

Annotations preserve identities and IDs, but change the catalog digest. Unknown annotations fail explicitly rather than interpreting an unreviewed card as lacking every capability. `scripts/annotate_policy_catalog.py` converts an engine-exported v1 catalog to v2 and refuses to overwrite an existing output.

Some subtle distinctions are intentional: ward on Hulking Raptor is not labelled as a counterspell/protection card for other objects; deathtouch on Ichorspit Basilisk is not an independent removal spell; an ordinary Forest is land access rather than extra mana acceleration. A conditional future capability such as Flourishing Bloom-Kin's search is labelled but is not presented as immediately usable from hand. Negate's broad counter role does not imply it answers a creature spell.

The role layer currently describes possession of capabilities. It does **not** claim mana affordability, timing permission, legal targets, independent availability of multiple modal effects, or combo completion. Those require additional descriptors/queries and controlled evidence that they help. Generated/stolen cards, richer reveal tracking, and other pre-existing engine limitations are unchanged.

## Learning and compatibility

PPO still trains the policy and the two value estimators. The statistical prior has no trainable parameters. Its hand NLL is a diagnostic only; the learner forces the effective belief-loss coefficient to zero for `statistical_roles`, even if an old training config requests a nonzero coefficient. True hand labels remain absent from `/act` inputs, and are used only after collection for the privileged critic and evaluation.

Checkpoints record their architecture name (`entity_statistical_roles_recurrent_policy` or `entity_belief_recurrent_policy`), and loading checks it against the belief mode. Strict state-dict loading and catalog digest checks are always enabled. On-policy trajectories of one actor are never reused as fresh data for another.

## Validation and measured speed

The tests compare the prior with exhaustive enumeration of small pools and with the count-DP model given binomial count scores, and cover overlapping role unions, known and empty hands, deck reordering and batch padding, role metadata validation, service retries, CPU and CUDA PPO gradients, and statistical checkpoint reloads.

The measurements below were taken when the mode was introduced, with the checkpoint format of that time; [aggregate validation data](statistical-roles-validation.json) records them. Comparing checkpoints before and after one update confirmed nonzero changes in both the role encoder and the action scorer, while the role probabilities for the same public observation stayed identical, as they have no trainable parameters.

On the real engine, two normal `simplegreen` mirror games produced four complete player trajectories with **1,223 decisions**, and a CPU PPO update trained the role encoder and policy. Collection took 30.95 seconds, learning 23.98 seconds, and the whole generation 55.20 seconds. This is integration evidence, not evidence of playing strength. In this mode the hand NLL is a diagnostic of the fixed prior, not a learned-head loss.

The matched synthetic forward benchmark used width 256, two attention layers, 64 entities, 32 actions, 24 deck kinds, hand size seven, batch one, one CPU thread, and 50 warmed timed samples:

| Device | Learned-count model median | Statistical-role model median |
|---|---:|---:|
| CPU | 7.29 ms | 4.72 ms |
| CUDA / RTX 5070 | 66.97 ms | 20.54 ms |

The statistical model's CPU median was approximately 35% lower for this workload. Actor parameter count at 24 kinds fell from 4,344,387 to 4,200,962. Timings exclude HTTP, encoding, Java, and learning; CUDA p95 varied substantially. `scripts/benchmark_statistical_roles.py` reproduces the benchmark.

## Reproduction

With the `simplegreen` catalog exported as in [XMage integration](xmage-integration.md#exporting-the-card-catalog) and fresh output paths:

```bash
python scripts/annotate_policy_catalog.py --catalog artifacts/simplegreen-cards.json \
    --annotations configs/card-roles.json --output artifacts/simplegreen-roles.json
mz init --catalog artifacts/simplegreen-roles.json --config configs/policy.yml \
    --output artifacts/roles-initial.pt --seed 20261012
mz selfplay --checkpoint artifacts/roles-initial.pt --game configs/smoke/real-card-smoke.yml \
    --config configs/smoke/policy-statistical-smoke.yml --engine-script $ENGINE \
    --output artifacts/roles-run --iterations 1 --games 2 --actor-device cpu --device cpu
python scripts/benchmark_statistical_roles.py --output artifacts/roles-benchmark.json
```

New deck pools need reviewed annotations for their entire catalog. A strength comparison needs fresh seeds and several relevant matchups, including counters, removal and combat tricks; `simplegreen` alone cannot establish a benefit for those categories.
