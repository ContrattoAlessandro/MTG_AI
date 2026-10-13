# Architecture review follow-ups: handoff (13 October 2026)

An architecture review on 13 October compared the policy with AlphaStar, OpenAI Five, DouZero/DouZero+ and PerfectDou. It produced nine numbered findings plus low-priority items. Findings 1–4 are implemented and verified; the rest remain. Read this file before the [next-session plan](next-session-plan.md) when continuing this work.

## State at handoff

**Nothing is committed.** Everything below is in the working tree of branch `policy-redesign`, on top of `f42f557`. The first task is reviewing the diff and committing it (`git status` lists 14 modified and 7 new paths).

The engine jars in `xmage/lib/` are already rebuilt with all Java changes: `mage-player-ai`, `mage-player-ai-ma` and `mage-magezero`. Backups are in `artifacts/jar-backup-20261013-pre-feature-fix/` (before fixes 1–2) and `artifacts/jar-backup-20261013-pre-teacher/` (before the teacher). `.mz_engine/` has the overlay installed.

Tests: 47 pass (`.venv/Scripts/python.exe -m pytest -q`), including 7 new ones in `tests/test_teacher.py`.

### Done

| Finding | Change | Verified by |
|---|---|---|
| 1. Opponent pass events existed only against policy opponents | `PolicyEncoder.recordChoice` writes passes only to the passing player's own history | Real games: every pass event has actor 0 in self-play and gauntlet games |
| 2. Candidate position (UUID-sorted, random per game) in action slot 0 | Slot 0 is the `amount` value only. Positions stable across games (yes/no, mode printed order, named-choice alphabetical, cast variant, pile) go in slot 2. Pile aggregates moved to 3–9. Layout in `docs/policy-protocol.md` | Episode check: slot 0 nonzero only for `amount` |
| 3. Credit assignment (γ = λ = 1, terminal reward only) | `src/magezero/policy/shaping.py`: potential-based shaping from public state. `training.shaping_coefficient`, plus `gae_lambda` 0.95 in `configs/policy-final.yml` | Unit tests (telescoping; baseline-only effect at λ = 1) and a real PPO smoke |
| 4. No supervised warm start | `TeacherPlayer` (XMage's AI, recorded over the policy's own candidates via the shared `PolicyCandidates`), `POST /record`, `mz teacher-collect`, `mz clone`, a reference-KL term in PPO (`training.reference_kl_coefficient` and `_decay`), `selfplay --reference`. Workflow in `docs/warm-start.md` | A 4-game real teacher collection (712 decisions), cloning, and 2 PPO generations from the clone with the KL term. All games terminal, no errors |

The smoke outputs are under the session scratchpad and are throwaway: `teacher-smoke/`, `cloned-smoke.pt`, `ppo-smoke/`.

## Remaining work, in order

### A. Before the final run

1. **Commit** the working tree, after reading the diff. The `PolicyPlayer` refactor onto `PolicyCandidates` is meant to preserve behaviour: same candidate order, ids and features. Check it in review.
2. ~~Decide what `artifacts/commander-initial.pt` is.~~ Resolved: it is a fresh initialization (`updates: 0`, no optimizer state, no metrics; `mz init`, seed 20261013, model section of `configs/policy-final.yml`, which is unchanged). No weights were trained on the old input meanings, so it is valid both as the clone's source and as the from-scratch baseline.
3. **Teacher data at scale.** Run `mz teacher-collect --games 2000 --engine-threads 16 --ai-skill 1` (see `docs/warm-start.md`). Record games/hour, decisions per game, the label mix (`inspect_teacher.py`-style summary: share of casts/activations among priority decisions, attacks among attack decisions) and disk use (about 8.5 KB per decision). Decide the AI skill: skill 1 is fast, higher skills are stronger teachers but slower. `--ai-think` caps seconds per decision.
4. **Clone.** Run `mz clone` with the `clone:` section of `configs/policy-final.yml` (10 epochs, 10% of games held out). Report held-out NLL/accuracy overall and by action type (`accuracy_by_type`). Watch for over-sharp priority heads: in the smoke, the clone predicted "pass" on every priority decision, though 712 decisions is far too few to judge.
5. **The controlled comparison (the important one).** At equal PPO budget, evaluated against XMage's AI on Winota vs Kinnan:
   - (a) the old setup: `commander-initial.pt`, λ 1.0, shaping 0, KL 0;
   - (b) cloned start with the new settings;
   - ideally also (c) cloned start with λ 1, shaping 0, KL 0, to separate the warm start from the λ/shaping change.

   The only earlier learning evidence (`docs/cpu-training-2026-10-12.md`) was for λ 1 from scratch. The new values (λ 0.95, shaping 0.5, KL 0.05 decaying ×0.999 per update) are untested starting points and should be tuned from this comparison.
6. Then the final run, as in `docs/long-run.md` (its command now starts from `commander-cloned.pt` with `--reference`).

### B. Known limitations of the warm start (decide if they matter after A5)

- **Unrecorded teacher decisions.** Targets fixed inside the AI's simulated casts, piles, cast variants, damage division (`chooseTargetAmount`) and replacement-effect order are played but not recorded. Teacher trajectories can therefore skip decisions a policy game contains, which shifts the decision-memory step distribution slightly. A fix would synthesize target decisions from the ability's pre-filled targets.
- **Deferred combat declarations.** During the AI's `selectAttackers`/`selectBlockers`, `TeacherPlayer` captures its `declareAttacker`/`declareBlocker` calls and applies them afterwards in the policy's order. The AI's code reads `isAttacking()` while declaring, so deferral can change its plan slightly; the first assignment per attacker wins. Not observed to break anything in the smoke.
- **The teacher is deterministic,** so the clone is low-entropy wherever the AI is consistent. Mulligan was always "keep", `use` always "yes", trigger order always the first option. PPO entropy and sampling must explore past this, while the KL term pulls back toward the teacher; that tension is why the KL weight decays.
- **Shaping is stated from the observer's view only.** The weights in `shaping.py` (health 1, hand 0.25, non-land permanents 0.25, creature power 0.5, through tanh) are hand-picked. `clone.shaping_coefficient` must equal `training.shaping_coefficient`.
- **Value semantics changed.** With shaping on, the critic and public value head estimate the shaped return V − φ, so value diagnostics are no longer win expectations.

### C. Review findings not yet addressed

| # | Severity | Finding | Where | Suggested fix |
|---|---|---|---|---|
| 5 | Medium | Belief is a fixed hypergeometric prior that ignores the opponent's behaviour; DouZero+ / ReBeL use posteriors. **Concrete bug:** opponent hand cards seen through `visibleChoices` (Thoughtseize-style effects) become entities but never update `knownHand`, so the belief contradicts the visible input | `src/magezero/policy/statistics.py`; `PolicyEncoder.observation` (`visibleChoices`) and `PolicyHistory.knownHand` | First fix the bug: mark opponent-hand cards revealed by a choice as known. Then train the hand predictor from memory and feed its prediction to the actor (DouZero+), with an ablation |
| 6 | Medium | The event GRU steps over every raw event (up to 8,192 between decisions; Kinnan loops), so BPTT through tens of thousands of steps | `model.py` `forward`, `forward_sequence`, `forward_episodes` | Pool each decision's events with attention (a small set transformer), then one recurrent step per decision. Keep `forward` / `forward_episodes` equivalence (test `test_multi_episode_forward_matches_separate_sequences`) and the log-prob replay check |
| 7 | Medium | One gradient-clip norm over actor and critic together, so a large critic loss shrinks policy steps | `learner.py` (clip before `optimizer.step()`; same in `teacher.py` `_pass`) | Clip the two parameter groups separately, or use two optimizers |
| 8 | Medium | Action head sees the current state only through the GRU, global features and pointers | `model.py` `_head` (`context_encoder`) | Concatenate `encoded.pooled` into the context input (a shape change: needs a fresh checkpoint or an upgrade hook) |
| 9 | Medium | Entropy bonus not normalized by candidate count (256-option amounts vs binary choices) | `learner.py` entropy term | Divide per-decision entropy by log(number of legal candidates) |
| – | Low | No final LayerNorm after the pre-norm transformer | `model.py` `entity_attention` | Add `norm=nn.LayerNorm(d)` to `TransformerEncoder` |
| – | Low | Turn phase as a scalar ordinal (`ordinal()/32`) | `PolicyEncoder.java` `global[9]` | Phase embedding or one-hot |
| – | Low | Rules text in 13 hashed word bins (heavy collisions) | `PolicyCandidates.describe` | Pretrained text embeddings in the card catalog |
| – | Low | Default final-layer init on the policy/value heads | `model.py` | Small init of the last policy layer (Andrychowicz et al. 2021); fresh runs only |
| – | Low | Failed activations stay in trajectories as no-op decisions | `PolicyPlayer.priority` (`rejected`) | Mask them from the policy loss, or drop the step |

Changes to findings 6 and 8, the LayerNorm and the init change the model's state dict: plan them as a new checkpoint generation, not a mid-run change. The phase-embedding item changes what the observation means, as fixes 1 and 2 did.

## Rebuilding after Java changes

Recipe in `docs/xmage-integration.md`. It now compiles `integration/xmage/src` plus `integration/xmage/src-ma` (`TeacherPlayer`, installed into the AI.MA module by `install.py`), and adds `TeacherPlayer*.class` to `mage-player-ai-ma-1.4.58.jar`. Check that no `java.exe` runs from `xmage/lib` first.
