# Architecture review follow-ups: handoff (13 October 2026)

An architecture review on 13 October compared the policy with AlphaStar, OpenAI Five, DouZero/DouZero+ and PerfectDou. It produced nine numbered findings plus low-priority items. Findings 1–4 are implemented and verified; the rest remain. **From 13 October evening, start with [final-run-plan.md](final-run-plan.md)**, which orders all remaining work; this file keeps the history and measurements.

## State at handoff

**Nothing is committed.** Everything below is in the working tree of branch `policy-redesign`, on top of `f42f557`. The first task is reviewing the diff and committing it (`git status` lists 14 modified and 7 new paths).

The engine jars in `xmage/lib/` are already rebuilt with all Java changes: `mage-player-ai`, `mage-player-ai-ma` and `mage-magezero`. Backups are in `artifacts/jar-backup-20261013-pre-feature-fix/` (before fixes 1–2) and `artifacts/jar-backup-20261013-pre-teacher/` (before the teacher). `.mz_engine/` has the overlay installed.

Tests: 47 pass (`.venv/Scripts/python.exe -m pytest -q`), including 7 new ones in `tests/test_teacher.py`.

### Done

| Finding | Change | Verified by |
|---|---|---|
| 1. Opponent pass events existed only against policy opponents | `PolicyEncoder.recordChoice` writes passes only to the passing player's own history | Real games: every pass event has actor 0 in self-play and gauntlet games |
| 2. Candidate position (UUID-sorted, random per game) in action slot 0 | Slot 0 is the `amount` value only. Positions stable across games (yes/no, mode printed order, named-choice alphabetical, cast variant, pile) go in slot 2. Pile aggregates moved to 3–9. Layout in `docs/policy-protocol.md` | Episode check: slot 0 nonzero only for `amount` |
| 3. Credit assignment (γ = λ = 1, terminal reward only) | `src/magezero/shaping.py`: potential-based shaping from public state. `training.shaping_coefficient`, plus `gae_lambda` 0.95 in `configs/policy-final.yml` | Unit tests (telescoping; baseline-only effect at λ = 1) and a real PPO smoke |
| 4. No supervised warm start | `TeacherPlayer` (XMage's AI, recorded over the policy's own candidates via the shared `PolicyCandidates`), `POST /record`, `mz teacher-collect`, `mz clone`, a reference-KL term in PPO (`training.reference_kl_coefficient` and `_decay`), `selfplay --reference`. Workflow in `docs/warm-start.md` | A 4-game real teacher collection (712 decisions), cloning, and 2 PPO generations from the clone with the KL term. All games terminal, no errors |

### In progress (13 October, second session)

- Committed as `40e175b` (A1). The `PolicyPlayer` refactor was checked decision by decision: same order, ids and early returns; feature changes are only the intended fix 2.
- A3 running: `artifacts/teacher/` (log `artifacts/teacher-collect.log`; skill 1, `--ai-think 2`). First batch: 32/32 terminal, ~500 games/h (≈4 h for 2,000), 226 decisions/game, **18 KiB/decision** (≈8 GB total, not 8.5 KB). Label mix: 22% of priority decisions cast/activate, 43% of attack decisions attack. Summary: `python scripts/teacher_stats.py artifacts/teacher`.
- A4/A5 queued behind it: `scripts/a5-comparison.sh` (log `artifacts/a5-chain.log`) clones to `artifacts/commander-cloned.pt`, then runs arms a, b, c in `artifacts/a5/` at 25 generations × 64 games each, evaluated at the end over 128 games vs XMage's AI (skill 3). Arms a and c use `configs/policy-compare-control.yml`. Rerun the script to resume. Arm (c)'s critic was pre-trained on shaped returns; at λ 1 it is only a baseline.

### A3–A5 results (13 October, finished 17:48)

- Teacher data: 2,000 games (3,987 trajectories, all terminal, 7 game errors), 418,521 decisions, 7.2 GiB. 23% of priority decisions cast/activate; 48% of attack decisions attack.
- Clone (`artifacts/commander-cloned.pt`, best epoch 9 of 10): held-out accuracy 92.3%, NLL 0.216. By type: pass 98.5%, cast 71.0%, activate 62.7%, attack 96.9%, block 81.6%, target 97.3%, amount 7% (576 decisions); the rest ≥ 99.6%. No all-pass collapse.
- Comparison, 25 generations × 64 games each, Winota learner, scored over 128 sampled games vs XMage's AI at skill 3 (Wilson 95% in brackets):

| Arm | Start | λ / shaping / KL | vs AI Winota (mirror) | vs AI Kinnan |
|---|---|---|---|---|
| a | `commander-initial.pt` | 1 / 0 / 0 | 0.24 [0.18–0.32] | 0.44 [0.35–0.52] |
| b | clone | 0.95 / 0.5 / 0.05 | **0.61** [0.52–0.69] | 0.56 [0.46–0.63] |
| c | clone | 1 / 0 / 0 | 0.48 [0.39–0.56] | **0.67** [0.59–0.75] |

  The warm start clearly helps: both cloned arms beat (a) on both matchups. The λ/shaping/KL change is unresolved: (b) leads on the mirror, (c) on the Kinnan matchup that the final run targets, with overlapping intervals, one seed each, and no score for the clone itself before PPO. Next: score the clone with no PPO, then rerun b vs c with more games or seeds (or longer) before fixing the final config.

### A5 follow-up (13 October, finished 21:23)

256 fresh sampled games per deck vs XMage's AI at skill 3 (`scripts/evaluate.py`, `artifacts/a5/rescore/`), plus a second run of b and c (`b2`, `c2`, evaluated in-run over 256 games). Scores with Wilson 95%:

| Checkpoint | vs AI Winota (mirror) | vs AI Kinnan |
|---|---|---|
| clone, no PPO | 0.42 [0.36–0.49] | 0.56 [0.48–0.61] |
| a (initial, control) | 0.24 [0.19–0.29] | 0.36 [0.29–0.40] |
| b (clone, new settings) | 0.54 [0.48–0.60] | 0.63 [0.56–0.68] |
| b2 | 0.51 [0.45–0.57] | 0.59 [0.52–0.64] |
| c (clone, control) | 0.44 [0.38–0.50] | 0.67 [0.60–0.72] |
| c2 | 0.50 [0.44–0.56] | 0.62 [0.55–0.67] |

- **Warm start: confirmed.** The clone alone beats 25 generations of PPO from the initial checkpoint on both matchups.
- **PPO on top of the clone helps a little:** about +0.05 to +0.10 after 25 generations, in all four runs.
- **New settings vs control: no measurable difference.** Mean of the two runs: mirror 0.52 (b) vs 0.47 (c), Kinnan 0.61 (b) vs 0.64 (c). Opposite signs, and the run-to-run spread (up to 0.06) is as large as the gaps. Mean over both matchups: 0.565 vs 0.555.
- Decision for the final run: keep `policy-final.yml` (b). It is no worse at this budget, and the decaying KL is there to prevent drift from the clone over thousands of generations, which a 25-generation test cannot show. Revisit λ/shaping if long-run evaluations stall.

The smoke outputs are under the session scratchpad and are throwaway: `teacher-smoke/`, `cloned-smoke.pt`, `ppo-smoke/`.

### Format-4 architecture (13 October, third session)

A second review confirmed the GAE recursion, PPO clipping and KL estimator, the reference-KL masking, the hypergeometric prior and role presence, the shaping potential (every feature index matches `PolicyEncoder.java`) and the recurrent batching. It found no math errors. Findings 6–9 and the low-priority model items were then implemented together as checkpoint **format 4**. Format 2/3 files no longer load (`mz init` creates a new one). `artifacts/commander-initial.pt` was re-created as format 4 with the same command and seed, while it was still an untrained initialization; the format-3 file is kept as `artifacts/commander-initial-v3.pt`. The teacher data is unaffected: it stores observations, not weights, and no observation changed.

| Change | Where |
|---|---|
| Finding 6: each decision's new events are pooled with attention (PMA block: per-event residual MLP, learned query, log2 recency embedding, log count); the event GRU steps once per decision with events | `model.py` `EventPool`, `_encode`, the three forwards |
| Finding 7: actor and privileged critic gradients clipped separately | `learner.py` `clip_gradients` (PPO and `teacher.py` cloning) |
| Finding 8: pooled entities concatenated into the actor context | `model.py` `context_encoder` (4·d input) |
| Finding 9: entropy bonus divided by log(legal candidates); `entropy` (raw) and `normalized_entropy` both logged | `learner.py` |
| Final LayerNorm on the pre-norm entity transformer | `model.py` `entity_attention` |
| Turn step one-hot into the global encoder (decoded from `globals[9]` in Python, no Java change), global state layer-normalized | `model.py` `global_encoder`, `global_norm` |
| Small init of the last policy layer | `model.py` `action_scorer` |
| Mean instead of summed relation messages | `model.py` `_encode` |
| KL early stopping per minibatch (1.5 × `target_kl`, before the step; 0 disables) instead of after a whole epoch; `early_stopped` metric | `learner.py` |
| PPO from a clone with `gae_lambda < 1` refuses a `shaping_coefficient`/`gamma` different from the clone's | `learner.py` `check_warm_start` |

Verified: 52 tests pass, including `tests/test_architecture_review.py` (stepwise, sequence and multi-trajectory forwards identical with 0 to 300 events per decision; pooling unaffected by batch padding; separate clipping; early stop; warm-start check; format-2/3 rejection). On real teacher games, live and learner forwards agree to 5e-10 in the logits. The three decision-GRU inputs now have equal scale (L2 norm 16 each; before, the entity summary was 10.2 against 1.2 for global features). Learner forward+backward on CUDA: 3,400–4,400 decisions/s against 2,400 for format 3, at the same peak memory (3.4 GiB for 48 games). With four 8,000-event bursts in one 16-trajectory batch: 1.5 s and 2.1 GiB against 27.5 s and 21.3 GiB (format 3 overflowed the 12 GB card).

Behaviour cloning on the same 300 teacher games (4 epochs, 15% held out, seed 7), format 4 against format 3:

```text
format 4 (new)
epoch 0: held-out nll 0.5742 acc 0.759 value mse 0.6904
epoch 1: held-out nll 0.4709 acc 0.820 value mse 0.5985
epoch 2: held-out nll 0.4342 acc 0.835 value mse 0.5611
epoch 3: held-out nll 0.4364 acc 0.829 value mse 0.5504
by type: {'pass': 0.984, 'cast': 0.536, 'activate': 0.403, 'target': 0.895, 'use': 0.41, 'attack': 0.832, 'block': 0.671, 'order': 0.973, 'amount': 0.1, 'mulligan': 1.0, 'choose': 0.571}
510 train / 90 held-out trajectories, 76 s
format 3 (old)
epoch 0: held-out nll 0.5933 acc 0.750 value mse 0.7514
epoch 1: held-out nll 0.5484 acc 0.788 value mse 0.6495
epoch 2: held-out nll 0.5071 acc 0.814 value mse 0.6734
epoch 3: held-out nll 0.4847 acc 0.815 value mse 0.7255
by type: {'pass': 0.972, 'cast': 0.522, 'activate': 0.42, 'target': 0.866, 'use': 0.378, 'attack': 0.78, 'block': 0.67, 'order': 0.932, 'amount': 0.308, 'mulligan': 1.0, 'choose': 0.502}
510 train / 90 held-out trajectories, 82 s
```

Format 4 is better on every held-out total (best NLL 0.434 vs 0.485, accuracy 0.835 vs 0.815, value MSE 0.55 vs 0.67) and per type except `amount` (0.10 vs 0.31) and slightly `activate`. Amount decisions are rare, so this is noisy, but it is worth checking in the full clone's `accuracy_by_type`: amounts are candidates whose only distinguishing feature is slot 0. One seed and 300 games: evidence that format 4 learns at least as well, not a measured strength gain.

Finding 5's concrete bug (opponent hand cards seen through `visibleChoices` not updating `knownHand`) cannot occur in Winota vs Kinnan: neither list has a card that shows or picks from the opponent's hand. It stays open for other decks. Vineglimmer Snarl's self-reveal belongs to the separate generic-reveal gap.

### Open review items closed (13 October, fourth session)

- **`mz clone` streamed from disk.** It loaded every teacher trajectory and kept all collated chunks on the GPU. At 600 trajectories the process held 8.1 GB of RAM, so the 2,000-game set (~4,000 trajectories, ~450,000 decisions) would have exceeded this machine's 31 GB and 12 GB VRAM right after collection. `_TeacherGames` now indexes files once (validation, byte offsets, return targets) and parses and collates one minibatch at a time, with the next one prefetched on a thread. Same 600 trajectories, 4 epochs: peak RSS 2.0 GB and VRAM 4.5 GB, independent of the data size; about 70 s per epoch, so roughly 80 min for the full 10-epoch clone. Clone metrics now include `hand_nll_per_card`.
- **Finding 5, belief ablation by cloning** (same 600 teacher trajectories, fresh format-4 initializations with seed 20261013, 4 epochs, 15% of games held out, belief weight 0.1 where trained):

  | Variant | Held-out NLL | Accuracy | Value MSE | Hand NLL per card | Time |
  |---|---|---|---|---|---|
  | Fixed hypergeometric prior (current) | 0.394 | 0.847 | 0.535 | 0.187 | 243 s |
  | Learned belief fed to the actor (`belief_mode: learned`, DouZero+) | 0.389 | 0.852 | 0.508 | 0.169 | 548 s |
  | Prior + auxiliary hand loss (`auxiliary_belief: true`) | 0.394 | 0.848 | 0.510 | 0.169 | 401 s |

  A learned predictor beats the prior on the actual hand by about 10%, so play history does carry hand information. Feeding it to the actor changed imitation by 0.005 NLL, within what one seed can resolve, at 2.3× the training time; the learned mode also drops the role-presence features. Decision: keep `statistical_roles` for A5 and the final run. Whether the better hand estimate matters for winning (not for imitating a teacher that never models hands) can only be measured with PPO arms; add one to `scripts/a5-extra-arms.sh` if A5 leaves time. The visibleChoices bug stays open for other decks (not reachable with these lists).
- **Failed activations measured:** in recorded policy games 2 of 877 decisions (0.23%; 0.9% of chosen casts/activations) were refused by the engine (`rejected`). Each is a no-op step with near-zero gradient. Masking needs a Java protocol change and a jar rebuild; deferred until no run uses the jars.
- **Rules-text bins: not changed.** Within the fixed 174-card pool every candidate already carries its card identity (learned embedding) and, for modes and named choices, its printed order; bins only need to separate abilities of the same card. Text embeddings mostly help unseen cards, and any change to the candidate features changes observations, which would discard the teacher data.
- **Learning-rate arm:** `configs/policy-final-lr1e4.yml` (policy-final with PPO learning rate 1e-4) and `scripts/a5-extra-arms.sh`, which waits for `== done` in `artifacts/a5-chain.log` and then runs arm (d) with the same budget and evaluation. Start it with `bash scripts/a5-extra-arms.sh > artifacts/a5-extra.log 2>&1` (it is a separate script because a running bash script must not be edited). Never run; removed with the other A5 files after the comparison.

## Remaining work, in order

### A. Before the final run

1. **Commit** the working tree, after reading the diff. The `PolicyPlayer` refactor onto `PolicyCandidates` is meant to preserve behaviour: same candidate order, ids and features. Check it in review.
2. ~~Decide what `artifacts/commander-initial.pt` is.~~ Resolved: it is a fresh initialization (`updates: 0`, no optimizer state, no metrics; `mz init`, seed 20261013, model section of `configs/policy-final.yml`, which is unchanged). No weights were trained on the old input meanings, so it is valid both as the clone's source and as the from-scratch baseline.
3. **Teacher data at scale.** Run `mz teacher-collect --games 2000 --engine-threads 16 --ai-skill 1` (see `docs/warm-start.md`). Record games/hour, decisions per game, the label mix (`inspect_teacher.py`-style summary: share of casts/activations among priority decisions, attacks among attack decisions) and disk use (about 8.5 KB per decision). Decide the AI skill: skill 1 is fast, higher skills are stronger teachers but slower. `--ai-think` caps seconds per decision.
4. **Clone.** (Streams from disk now, ~80 min for 2,000 games; format 4: `commander-initial.pt` was re-created before cloning; see above.) Run `mz clone` with the `clone:` section of `configs/policy-final.yml` (10 epochs, 10% of games held out). Report held-out NLL/accuracy overall and by action type (`accuracy_by_type`). Watch for over-sharp priority heads: in the smoke, the clone predicted "pass" on every priority decision, though 712 decisions is far too few to judge.
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

Findings 6–9, the final LayerNorm, the phase encoding and the policy-layer init are done (format 4, above).

| # | Severity | Finding | Where | Suggested fix |
|---|---|---|---|---|
| 5 | Low here | Bug: opponent hand cards seen through `visibleChoices` never update `knownHand`. Unreachable with Winota/Kinnan. The belief ablation is done (above) | `PolicyEncoder.observation`, `PolicyHistory.knownHand` | Fix with the next Java rebuild; optional PPO arm with `belief_mode: learned` |
| – | Low | Failed activations stay in trajectories as no-op decisions (0.23% of decisions) | `PolicyPlayer.priority` (`rejected`) | With the next Java rebuild: report the refused step to the service and mask it from the policy loss |
| – | Low | Rules text in 13 hashed word bins | `PolicyCandidates.describe` | Only if the card pool grows (would need new teacher data) |

Further changes to the model's layers again need a new checkpoint format and a fresh `mz init` before cloning; changes to the Java observation need re-collected teacher data.

## Rebuilding after Java changes

Recipe in `docs/xmage-integration.md`. It now compiles `integration/xmage/src` plus `integration/xmage/src-ma` (`TeacherPlayer`, installed into the AI.MA module by `install.py`), and adds `TeacherPlayer*.class` to `mage-player-ai-ma-1.4.58.jar`. Check that no `java.exe` runs from `xmage/lib` first.
