#!/usr/bin/env bash
# Acceptance evaluation of the final run (docs/playing-the-agent.md#how-strong-it-is):
# 400 greedy games per checkpoint against XMage's AI at skill 3, piloting Winota
# (mirror) and Kinnan, seats and starts alternating. On Windows, run it from Git Bash.
#
#   bash scripts/final-evaluate.sh
#   # results: artifacts/final/evaluation/evaluations.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-artifacts/final/run}
if [ -x .venv/Scripts/mz.exe ]; then MZ=.venv/Scripts/mz.exe
elif [ -x .venv/bin/mz ]; then MZ=.venv/bin/mz
else MZ=mz; fi
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ENGINE=xmage/mz-policy.bat ;; *) ENGINE=xmage/mz-policy.sh ;; esac

"$MZ" evaluate --checkpoint \
    "$RUN/best.pt" "$RUN/generation_0500/policy.pt" "$RUN/generation_0600/policy.pt" \
    "$RUN/generation_0850/policy.pt" "$RUN/generation_0999/policy.pt" \
    --output artifacts/final/evaluation --game configs/commander-mirror.yml \
    --gauntlet xmage/decks/kinnan.dck --engine-script "$ENGINE" \
    --games 400 --ai-skill 3 --engine-threads 16 --greedy --device cpu
