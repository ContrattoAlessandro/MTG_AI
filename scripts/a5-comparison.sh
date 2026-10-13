#!/usr/bin/env bash
# A5 of docs/review-handoff.md: wait for the teacher collection, clone, then three
# PPO arms at equal budget (25 generations of 64 games), each evaluated at the end
# against XMage's AI (skill 3, 128 games per deck). Rerunning resumes finished steps.
#   (a) commander-initial.pt, control settings (lambda 1, no shaping, no KL)
#   (b) commander-cloned.pt, policy-final.yml (lambda 0.95, shaping 0.5, KL 0.05)
#   (c) commander-cloned.pt, control settings
set -euo pipefail
cd "$(dirname "$0")/.."
MZ=.venv/Scripts/mz.exe
OUT=artifacts/a5
mkdir -p "$OUT"

until .venv/Scripts/python.exe -c "import json,sys; sys.exit(json.load(open('artifacts/teacher/teacher-summary.json'))['games_played'] < 2000)" 2>/dev/null; do
    if grep -qE "Traceback" artifacts/teacher-collect.log; then echo "teacher collection failed" >&2; exit 1; fi
    sleep 60
done
sleep 60  # let the collector stop its engine and service

if [ ! -f artifacts/commander-cloned.pt ]; then
    echo "== clone $(date)"
    "$MZ" clone --checkpoint artifacts/commander-initial.pt --episodes artifacts/teacher/episodes \
        --output artifacts/commander-cloned.pt --config configs/policy-final.yml --device cuda 2>&1 | tee "$OUT/clone.log"
fi

arm() {  # name checkpoint config [extra args]
    local name=$1 checkpoint=$2 config=$3; shift 3
    local resume=()
    [ -f "$OUT/$name/run-state.json" ] && resume=(--resume)
    echo "== arm $name $(date)"
    "$MZ" selfplay --checkpoint "$checkpoint" --game configs/commander-mirror.yml --config "$config" \
        --engine-script xmage/mz-policy.bat --output "$OUT/$name" --iterations 25 --games 64 \
        --gauntlet xmage/decks/kinnan.dck --gauntlet-fraction 0.5 --gauntlet-ai-skill 1 \
        --engine-threads 16 --device cuda --eval-every 25 --eval-games 128 --eval-ai-skill 3 \
        --keep-every 5 --min-free-gb 30 "${resume[@]}" "$@" >> "$OUT/$name.log" 2>&1
}

arm a artifacts/commander-initial.pt configs/policy-compare-control.yml
arm b artifacts/commander-cloned.pt configs/policy-final.yml --reference artifacts/commander-cloned.pt
arm c artifacts/commander-cloned.pt configs/policy-compare-control.yml
echo "== done $(date)"
