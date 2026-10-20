#!/usr/bin/env bash
# The final run: Winota learner, Kinnan gauntlet, from the 20-epoch behaviour clone,
# under the crash-restarting supervisor (scripts/final_run.py). Rerun this script
# unchanged to continue after an interruption; the supervisor adds --resume.
#
#   bash scripts/final-run.sh            # foreground
#   tail -f artifacts/final/run/status.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=artifacts/final/run
CLONE=artifacts/commander-cloned-20.pt
mkdir -p "$OUT"

if [ ! -f "$OUT/provenance.txt" ]; then
    {
        echo "commit $(git rev-parse HEAD) $(git describe --tags --always)"
        sha256sum "$CLONE" configs/final.yml configs/commander-mirror.yml xmage/decks/winota.dck \
            xmage/decks/kinnan.dck xmage/mz/*.jar xmage/mz/build.json
    } > "$OUT/provenance.txt"
fi

"$PY" scripts/final_run.py --output "$OUT" -- \
    --checkpoint "$CLONE" --reference "$CLONE" \
    --game configs/commander-mirror.yml --config configs/final.yml --engine-script xmage/mz-policy.bat \
    --iterations 1000 --games 64 \
    --gauntlet xmage/decks/kinnan.dck --gauntlet-fraction 0.5 --gauntlet-ai-skill 1 \
    --engine-threads 16 --device cuda \
    --eval-every 25 --eval-games 64 --eval-ai-skill 3 --eval-greedy \
    --eval-anchor artifacts/anchors/a5-b-gen24.pt \
    --keep-every 50 --min-free-gb 30
