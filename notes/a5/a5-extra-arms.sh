#!/usr/bin/env bash
# Extra A5 arms, run after notes/a5/a5-comparison.sh finishes (same budget and
# evaluation as arms a-c, so all arms are comparable). Rerunning resumes.
#   (d) commander-cloned.pt, policy-final.yml with learning rate 1e-4 (fine-tuning rate)
set -euo pipefail
cd "$(dirname "$0")/../.."
MZ=.venv/Scripts/mz.exe
OUT=artifacts/a5

until grep -q "^== done" artifacts/a5-chain.log 2>/dev/null; do
    if grep -qE "Traceback|failed" artifacts/a5-chain.log 2>/dev/null; then echo "A5 chain failed" >&2; exit 1; fi
    sleep 300
done

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

arm d artifacts/commander-cloned.pt configs/experiments/policy-final-lr1e4.yml --reference artifacts/commander-cloned.pt
echo "== done $(date)"
