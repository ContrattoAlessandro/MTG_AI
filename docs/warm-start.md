# Warm start: behaviour cloning from XMage's AI, then regularized PPO

AlphaStar's ablations show that reinforcement learning from scratch fails without a supervised start and a KL penalty towards it. Without them, the only learning signal here would be the ±1 result after hundreds of decisions. Three pieces address that:

| Piece | Where | What it does |
|---|---|---|
| Teacher recording | `integration/xmage/src-ma/mage/player/ai/TeacherPlayer.java`, `POST /record` | XMage's Mad AI (ComputerPlayer7) plays; each decision becomes a label over the policy's own candidate list (`PolicyCandidates`, shared with `PolicyPlayer`), with the observation the policy would have seen. |
| Behaviour cloning | `src/magezero/teacher.py`, `mz clone` | Cross-entropy on the teacher's choices, memory rebuilt from step zero as in PPO; both value heads pre-trained on the games' shaped outcomes. Keeps the epoch with the lowest held-out NLL (whole games held out). Streams from disk: files are indexed once (byte offsets, return targets) and one minibatch is parsed and collated at a time, so memory does not grow with the number of games. |
| Regularized PPO | `learner.py`, `shaping.py` | `reference_kl_coefficient` × KL(cloned ‖ policy) over legal candidates, decayed per update; potential-based shaping; `gae_lambda` 0.95. |

## How the teacher's decisions are mapped

The AI decides first; its answer is then located in the candidate list the policy would get in the same state.

- **Priority.** The AI's search returns a sequence of abilities, which it activates one by one, then passes. Each activation is one recorded decision (matched by source and ability identity), and so is the pass.
- **Combat.** Its attack and block declarations are captured, then replayed one creature at a time in the policy's order, each as a stay/attack or none/block decision.
- **Targets** are chosen on a copy of the target, then added one at a time with a final stop. **Use, mulligan, named choice, amount, mode and trigger order** record the returned answer.
- **Not recorded:** answers outside the candidate list (most often targets the AI's search fixed inside a simulated cast, which the engine then never asks for), piles, cast variants, damage division and replacement order. Those decisions are played, so teacher trajectories can be missing some steps a policy game contains. Forced single-candidate decisions go into the history only, exactly as for the policy.

The teacher is deterministic, so the cloned policy is sharp (low entropy) where the AI is consistent. The KL term keeps PPO near it while the entropy bonus and sampling explore.

## Shaping and λ

`shaping.py` defines φ(s) = tanh(health difference + 0.25·hand difference/7 + 0.25·non-land permanents/10 + 0.5·creature power/20) from the observer's public state. Health is the nearest loss condition: life/40, poison, or commander damage. The learner adds γφ(s′) − φ(s) per decision, with φ = 0 after a finished game and no term after a truncated one. By Ng et al. (1999) this leaves the optimal policy unchanged. With γ = λ = 1 it only shifts the value targets; it matters together with λ < 1, where it acts like a value initialization. `clone.shaping_coefficient` must equal `training.shaping_coefficient`, so the pre-trained values estimate the returns PPO regresses on.

## Workflow

```bash
# 1. Teacher games: both decks, recorded from both seats (resumable: rerun to continue).
mz teacher-collect --checkpoint artifacts/commander-initial.pt --game configs/commander-mirror.yml \
    --engine-script $ENGINE --output artifacts/teacher --games 2000 --engine-threads 16 --ai-skill 1
python scripts/teacher_stats.py artifacts/teacher     # label mix and size of the recorded data

# 2. Behaviour cloning (the clone: section of the config).
mz clone --checkpoint artifacts/commander-initial.pt --episodes artifacts/teacher/episodes \
    --output artifacts/commander-cloned.pt --config configs/policy.yml

# 3. PPO from the clone, regularized towards it (full command in long-run.md).
mz selfplay --checkpoint artifacts/commander-cloned.pt --reference artifacts/commander-cloned.pt \
    --game configs/commander-mirror.yml --config configs/policy.yml ...
```

`mz clone` prints the training and held-out negative log-likelihood and accuracy per epoch; the checkpoint's `metrics.behavior_cloning` keeps the full history, including accuracy by action type, hand NLL per card and the public value error.

`teacher-collect` uses its checkpoint only for the card catalog and limits; it never chooses an action. `selfplay` refuses a positive `reference_kl_coefficient` without `--reference`, and keeps the reference in `run-state.json` across `--resume`. PPO from a cloned checkpoint with `gae_lambda < 1` refuses a `shaping_coefficient` or `gamma` different from the ones the clone was trained with (at λ 1 the critic is only a baseline, so the control arm may differ). Metrics gain `reference_kl`, `reference_kl_coefficient`, `shaping_coefficient` and `gae_lambda`.

## Results: the A5 comparison

Three arms of 25 PPO generations (64 games each, half of them against XMage's AI with Kinnan at skill 1) separate the warm start from the credit-assignment settings:

| Arm | Start | λ / shaping / KL |
|---|---|---|
| a | fresh `mz init` checkpoint | 1 / 0 / 0 |
| b | behaviour-cloned checkpoint | 0.95 / 0.5 / 0.05 (`configs/policy.yml`) |
| c | behaviour-cloned checkpoint | 1 / 0 / 0 (`policy.yml` with these three values) |

The teacher data were 2,000 games of XMage's AI at skill 1 playing both decks (3,987 trajectories, 418,521 decisions). The clone reached 92.3% held-out accuracy (NLL 0.216): 98.5% on passes, 71.0% on casts, 62.7% on activations, 96.9% on attacks and 81.6% on blocks. After training, every checkpoint below played 256 fresh sampled games per opponent deck against XMage's AI at skill 3, with `mz evaluate`. b2 and c2 are second runs of b and c. Scores count a win as 1 and a draw or truncation as ½, with Wilson 95% intervals of the win rate:

| Checkpoint | vs XMage AI, Winota (mirror) | vs XMage AI, Kinnan |
|---|---|---|
| clone, no PPO | 0.42 [0.36–0.49] | 0.56 [0.48–0.61] |
| a | 0.24 [0.19–0.29] | 0.36 [0.29–0.40] |
| b | 0.54 [0.48–0.60] | 0.63 [0.56–0.68] |
| b2 | 0.51 [0.45–0.57] | 0.59 [0.52–0.64] |
| c | 0.44 [0.38–0.50] | 0.67 [0.60–0.72] |
| c2 | 0.50 [0.44–0.56] | 0.62 [0.55–0.67] |

- **The warm start is decisive at this budget.** The clone with no PPO at all outscores 25 generations of PPO from a fresh checkpoint on both matchups.
- **PPO improves on the clone, modestly:** by about 0.05 to 0.10 after 25 generations, in all four cloned runs.
- **λ, shaping and the KL term made no measurable difference.** Averaged over the two runs, the new settings lead on the mirror (0.52 against 0.47) and trail against Kinnan (0.61 against 0.64), and two runs of the same settings differ by up to 0.06. `configs/policy.yml` keeps them, since the decaying KL term is meant to prevent drift from the clone over thousands of generations, which a 25-generation comparison cannot test.

The comparison's control configuration and scripts were removed once it was decided; they are in git history (commit `0f2acb7`, `configs/experiments/policy-compare-control.yml` and `notes/a5/`). The clone in this comparison trained for 10 epochs; the final run's clone for 20 (`clone.epochs` in `configs/policy.yml`).

## Pipeline check

A 4-game Commander smoke run (Winota and Kinnan, AI skill 1, 2 s think cap) took 13 s for 4 terminal games and 8 trajectories with 712 recorded decisions. Of the 369 priority decisions, 110 cast or activate; 93 of 163 attack decisions attack. Cloning, followed by PPO generations with the KL term, ran end to end. This showed that the pipeline works; the comparison above shows that it helps.
