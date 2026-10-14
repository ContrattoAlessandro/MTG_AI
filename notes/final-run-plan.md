# Plan to the final training run and a working agent

Written 13 October 2026, end of the fourth session. This is the plan for the next sessions. It replaces the ordering in [review-handoff.md](review-handoff.md) section A; that file keeps the history and measurements. Read this file first.

**Done means:** a checkpoint (`best.pt` of the final run) that pilots Winota against Kinnan in Commander Duel, confirmed by a large, fresh evaluation against XMage's AI, and playable by the user in the XMage client through `mz play`.

## Questions for the user (answer before Phase 3)

1. **Time budget for the final run.** At the measured 1,100–2,000 games/hour, 1,000 generations of 64 games take about 1.5–2.5 days, and 5,000 generations (the command in `long-run.md`) take 7–12 days. The A5 learning curves will show where improvement flattens.
2. **What counts as "working".** Proposal: a score of at least 0.60 against XMage's AI at skill 3, with Winota against Kinnan, over 400 greedy games, with the 95% interval above 0.5. Also report skill 1 and the highest skill that finishes in reasonable time.
3. **GPU sharing.** Another project (`ablations/campaign.py`) has used the RTX 5070. The final run should have it to itself, or its timings will be off.

## Phase 0: running now (no action needed)

Done. The A5 scripts and comparison configurations were removed afterwards (git history, commit `0f2acb7`); arm (d) never ran. Results: [docs/warm-start.md](../docs/warm-start.md#results-the-a5-comparison).

`scripts/a5-comparison.sh` (log `artifacts/a5-chain.log`) runs these steps one after another:

1. Clone (`artifacts/a5/clone.log`): reads all 3,987 teacher trajectories once, then trains for 10 epochs, about 1.5–2 h in all. Output `artifacts/commander-cloned.pt`.
2. Arms (a), (b), (c), each 25 generations × 64 games, then 128 evaluation games against XMage's AI at skill 3 (`artifacts/a5/{a,b,c}.log`, `evaluations.jsonl` in each arm folder).

Optional: `bash scripts/a5-extra-arms.sh > artifacts/a5-extra.log 2>&1` queues arm (d) (learning rate 1e-4), which starts after the chain.

**Check once the clone finishes, before the arms spend hours:**
- Held-out accuracy overall and in `accuracy_by_type`. Watch two things: priority decisions predicted as "pass" almost always; and `amount`, which is rare and noisy.
- `hand_nll_per_card` and `public_value_mse` decreasing over epochs.
- The ablation on 600 trajectories reached held-out NLL 0.39 and accuracy 0.85. The full clone should do at least as well.

## Phase 1: decide from A5 (one session)

Compare arms on the evaluation score with Wilson intervals. With 128 games an interval is about ±0.09, so only large differences are decisive. Also check the training-game win rate against Kinnan per generation, `failure_fraction`, `truncation_fraction`, `entropy` / `normalized_entropy`, `approx_kl`, `early_stopped`, and `reference_kl`.

| Result | Decision |
|---|---|
| (b) clearly above (a) | Warm start with the new settings. Use (b), or (d) if it is better. |
| (c) ≈ (b), both above (a) | The gain is the clone. Keep (b)'s settings unless (c) is better; they are cheap either way. |
| (b) ≈ (a) | The warm start does not help at this budget. Look at `reference_kl`: if the KL pull is holding the policy at the teacher, try KL 0.01 or a faster decay. Check whether shaping 0.5 dominates the value targets. Run one more short arm before the final run. |
| Any arm with failures above 5% or truncations above 5% | Fix that first (engine patch or `max_turns`), whatever the scores. |

If the arms cannot be separated, run the confirmation evaluation (T1 below) on their final checkpoints with 400 games each before deciding.

Optional follow-up arms, if time allows: `belief_mode: learned` (finding 5; in cloning it predicted hands 10% better but did not change imitation), or a second seed of the winning arm.

## Phase 2: code to finish before the final run (one or two sessions)

### T1. `mz evaluate`: standalone evaluation (needed)

There is no command that evaluates a checkpoint without training. `_evaluate` exists only inside `selfplay`, and `scripts/evaluate_policy_games.py` is the legacy MCTS evaluator.
- Add `mz evaluate --checkpoint X [--checkpoint Y ...] --game configs/commander-mirror.yml --opponent-deck xmage/decks/kinnan.dck --ai-skill 1 3 ... --games N --greedy/--sample --engine-threads 16 --output dir`. Reuse `runner._evaluate` and the persistent engine and policy services.
- Alternate seats and starting player. Report the score with a Wilson interval, wins/draws/losses/truncations/errors, average turns, and games/hour per skill. Write JSON and a short Markdown summary.
- Tests: runner-style test with the simulated engine, as in `tests/test_long_run.py`.
- Used for: A5 tie-breaks, the final acceptance evaluation, and choosing among kept checkpoints (`best.pt` is biased upward by selection).

### T2. Java rebuild window (when no run uses `xmage/lib`)

Back up the jars first (`artifacts/jar-backup-<date>/`) and follow the rebuild recipe in `next-session-plan.md`. One rebuild covers all three changes.
- **Finding 5 bug (done 14 October):** opponent hand cards shown through `visibleChoices` must update `PolicyHistory.knownHand`, so public lower bounds include them. It cannot happen with these two decks, but it is cheap and keeps the observation correct for any later deck.
- **Failed activations (0.23% of decisions; done 14 October, as `rejected_steps` on `/finish`, also masking the choices made during the refused activation):** when `activateAbility` fails in `PolicyPlayer.priority`, report the refused step to the service. Make it a new optional step field `rejected: true`, set through a small endpoint or the next request, and never part of the observation. `validate_episode` accepts the field, and `ppo_update` sets `choice = 0` for those steps, so there is no policy, entropy or KL term. The values still train. Teacher data has no such steps and is unaffected.
- **Optional:** record generic reveals (e.g. Vineglimmer Snarl revealing a land from hand) as public knowledge. This would change observations; only do it before collecting new data.
- Verify with the CPU smoke in `next-session-plan.md` (4 games, zero errors), plus one gauntlet batch. Run all tests.

Teacher data does not need re-collection for any of these, because they change no observation field used by the existing data. Check this in the diff.

### T3. Unattended-run supervisor (needed for a multi-day run)

- A wrapper script (`scripts/final-run.sh` or `.ps1`). It runs `mz selfplay ... --resume` in a loop. It restarts after a crash with a bounded retry count and a pause, and stops on a clean finish or on the failure guard (`--max-failure-fraction`). It logs every restart.
- Keep the PC awake for the duration: Windows power settings, or the app's keep-awake. Document how to check.
- A disk projection from A5 numbers: about 20 MB per generation without kept checkpoints, plus about 52 MB per kept checkpoint. Set `--keep-every` and `--min-free-gb` from it.
- Optional: a daily summary line (generation, best score, games/hour, failures) appended to a file the user can glance at.

### T4. Run report

Extend `scripts/report_cpu_training.py` / `plot_cpu_training.py` (written for the CPU pilot) to the Commander run folder. It should plot:
- the evaluation score with intervals per generation;
- the training-game win rate against Kinnan;
- `failure_fraction` / `truncation_fraction`;
- entropy, normalized entropy, KL, reference KL and its coefficient, value losses;
- games/hour.

Use it on A5 first, then on the final run.

### T5. Freeze the final configuration

- Write `configs/final.yml` from the A5 winner: learning rate, λ, shaping, KL weight and decay, `entropy_coefficient`. The entropy bonus is now normalized by log(number of candidates), so 0.01 means more exploration on binary choices than before. Check entropy in A5 before keeping it.
- Settle the generation mix: `--gauntlet-fraction`, `--gauntlet-ai-skill` (1 is fast; consider raising it as the policy improves), `--eval-every`, `--eval-games` (at least 64, greedy), `--snapshot-every`.
- Update the command in `docs/long-run.md` with the final numbers.
- Commit, and tag the commit used for the run (for example `final-run-1`). Record the SHA-256 of the three patched jars and the catalog digest in the run folder, so the agent can be reproduced.
- Merge `policy-redesign` into `main` (or open a PR) before or after the run, as the user prefers.

### T6. Human-play check with a format-4 checkpoint

The deliverable is playing against the agent, so test that path before spending days on training. Run `mz play --checkpoint artifacts/commander-cloned.pt --greedy`, then play one Commander Two Player Duel in the XMage client: the user's Kinnan against the agent's Winota. Confirm that it loads, responds in reasonable time on CPU, and handles mulligans, combat and Winota triggers. Fix any server-path issue then, not after the run.

## Phase 3: launch

1. Short launch check (1–2 h) with `configs/final.yml` and the supervisor, unless an A5 arm already ran the identical configuration. Confirm games/hour, failures below 2%, truncations below 2%, the learner under about 10% of the time, and disk use as projected.
2. Start the final run from `commander-cloned.pt` with `--reference commander-cloned.pt` (if KL is on), through the supervisor.
3. While it runs, every day or two: the T4 report. Stop early if the evaluation score has been flat for a long stretch (for example 300 generations), or if failures rise.

## Phase 4: after the run

1. Run `mz evaluate` (T1) on `best.pt` and the last few kept checkpoints: 400 greedy games each against XMage's AI at skill 3, plus skill 1 and a higher skill, with alternating seats and starts. Choose the final agent from this fresh evaluation, not from `best.pt`'s own score.
2. Copy the chosen checkpoint to `artifacts/winota-agent.pt` without optimizer state (`_strip_optimizer`). Write `docs/playing-the-agent.md`: the `mz play` command, the `--greedy` recommendation, the deck rules (catalog cards only, and the three substitutions), and the measured strength.
3. Play it (T6 path) and note weaknesses for any later work.

## Later / optional (not needed for a working agent)

- A PPO arm with `belief_mode: learned` (finding 5).
- Task E: imitation from the user's own games (`next-session-plan.md`).
- Warm-start limitations in `review-handoff.md` section B: unrecorded teacher decisions, deferred combat declarations, a deterministic teacher.
- Rules-text embeddings: only if the card pool grows (needs new teacher data).
- Multiple training seeds for a robust strength estimate.

## Order and rough effort

| Step | When | Effort |
|---|---|---|
| Phase 0 checks | as the clone and arms finish | minutes |
| Phase 1 decision | after A5 | ½ session |
| T1 `mz evaluate` | can start now (Python only) | ½–1 session |
| T4 report | can start now | ½ session |
| T3 supervisor | can start now | ½ session |
| T2 Java rebuild | after A5, nothing running | ½–1 session |
| T5 freeze, T6 human-play check | after Phase 1 and T2 | ½ session |
| Phase 3 launch | after T1–T6 | 1–2 h check, then days |
| Phase 4 | after the run | ½–1 session |

T1, T3 and T4 do not touch the engine or the running jobs, so they can be done while A5 runs. That is safe only because each step of the running A5 script imports Python code when it starts. Avoid editing `model.py`, `learner.py`, `runner.py` or `teacher.py` in ways that change behaviour until A5 is done: a later arm would import the changed code and the comparison would no longer be equal.
