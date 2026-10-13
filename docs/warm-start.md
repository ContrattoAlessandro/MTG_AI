# Warm start: behaviour cloning from XMage's AI, then regularized PPO

Implemented 13 October 2026, from the architecture review (findings 3 and 4). AlphaStar's ablations show RL from scratch fails without a supervised start and a KL penalty towards it; this repository had neither, and its only reward was the ±1 result after hundreds of decisions. Three pieces address that:

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

```powershell
# 1. Teacher games: both decks, recorded from both seats (resumable: rerun to continue).
.venv/Scripts/mz.exe teacher-collect --checkpoint artifacts/commander-initial.pt --game configs/commander-mirror.yml `
  --engine-script xmage/mz-policy.bat --output artifacts/teacher --games 2000 --engine-threads 16 --ai-skill 1

# 2. Behaviour cloning (clone: section of the config).
.venv/Scripts/mz.exe clone --checkpoint artifacts/commander-initial.pt --episodes artifacts/teacher/episodes `
  --output artifacts/commander-cloned.pt --config configs/policy-final.yml

# 3. PPO from the clone, regularized towards it.
.venv/Scripts/mz.exe selfplay --checkpoint artifacts/commander-cloned.pt --reference artifacts/commander-cloned.pt `
  --game configs/commander-mirror.yml --config configs/policy-final.yml ... (as in long-run.md)
```

`teacher-collect` uses its checkpoint only for the card catalog and limits; it never chooses an action. `selfplay` refuses a positive `reference_kl_coefficient` without `--reference`, and keeps the reference in `run-state.json` across `--resume`. PPO from a cloned checkpoint with `gae_lambda < 1` refuses a `shaping_coefficient` or `gamma` different from the ones the clone was trained with (at λ 1 the critic is only a baseline, so the control arm may differ). Metrics gain `reference_kl`, `reference_kl_coefficient`, `shaping_coefficient` and `gae_lambda`.

## Measured so far

A 4-game Commander smoke (Winota/Kinnan, AI skill 1, 2 s think cap) took 13 s for 4 terminal games and 8 trajectories with 712 recorded decisions. Of the 369 priority decisions, 110 cast or activate; 93 of 163 attack decisions attack. Cloning, followed by PPO generations with the KL term, ran end to end. This shows the pipeline works, not that it helps: the next step is a controlled comparison of the cloned start against `commander-initial.pt` at equal PPO budget, evaluated against XMage's AI.
