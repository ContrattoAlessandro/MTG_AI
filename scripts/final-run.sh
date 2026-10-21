#!/usr/bin/env bash
# The final run that produced the published agent: Winota learner, Kinnan gauntlet,
# from the 20-epoch behaviour clone, under the crash-restarting supervisor
# (scripts/final_run.py). Rerun this script unchanged to continue after an
# interruption; the supervisor adds --resume. On Windows, run it from Git Bash.
#
#   bash scripts/final-run.sh            # foreground
#   tail -f artifacts/final/run/status.log
#
# CLONE is the output of `mz clone --config configs/policy.yml` (README, step 4).
# ANCHOR, if present, is an extra frozen benchmark (the published run used the best
# 25-generation checkpoint of the warm-start comparison); it does not affect training.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-artifacts/final/run}
CLONE=${CLONE:-artifacts/commander-cloned-20.pt}
ANCHOR=${ANCHOR:-artifacts/anchors/a5-b-gen24.pt}
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ENGINE=xmage/mz-policy.bat ;; *) ENGINE=xmage/mz-policy.sh ;; esac
EXTRA=()
if [ -f "$ANCHOR" ]; then EXTRA+=(--eval-anchor "$ANCHOR"); fi
mkdir -p "$OUT"

if [ ! -f "$OUT/provenance.txt" ]; then
    {
        echo "commit $(git rev-parse HEAD) $(git describe --tags --always)"
        sha256sum "$CLONE" configs/policy.yml configs/commander-mirror.yml xmage/decks/winota.dck \
            xmage/decks/kinnan.dck xmage/mz/*.jar xmage/mz/build.json
    } > "$OUT/provenance.txt"
fi

"$PY" scripts/final_run.py --output "$OUT" -- \
    --checkpoint "$CLONE" --reference "$CLONE" \
    --game configs/commander-mirror.yml --config configs/policy.yml --engine-script "$ENGINE" \
    --iterations 1000 --games 64 \
    --gauntlet xmage/decks/kinnan.dck --gauntlet-fraction 0.5 --gauntlet-ai-skill 1 \
    --engine-threads 16 --device cuda \
    --eval-every 25 --eval-games 64 --eval-ai-skill 3 --eval-greedy \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --keep-every 50 --min-free-gb 30
