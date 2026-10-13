#!/usr/bin/env bash
# A5 follow-up: score the clone with no PPO and re-score the arms' final checkpoints
# over 256 fresh games per deck, then a second run of arms b and c. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
MZ=.venv/Scripts/mz.exe
OUT=artifacts/a5

if [ ! -f "$OUT/rescore/evaluations.jsonl" ]; then
    echo "== rescore $(date)"
    .venv/Scripts/python.exe scripts/evaluate.py "$OUT/rescore" artifacts/commander-cloned.pt \
        "$OUT/a/generation_0024/policy.pt" "$OUT/b/generation_0024/policy.pt" "$OUT/c/generation_0024/policy.pt" \
        --games 256 >> "$OUT/rescore.log" 2>&1
fi

arm() {  # name checkpoint config [extra args]
    local name=$1 checkpoint=$2 config=$3; shift 3
    local resume=()
    [ -f "$OUT/$name/run-state.json" ] && resume=(--resume)
    echo "== arm $name $(date)"
    "$MZ" selfplay --checkpoint "$checkpoint" --game configs/commander-mirror.yml --config "$config" \
        --engine-script xmage/mz-policy.bat --output "$OUT/$name" --iterations 25 --games 64 \
        --gauntlet xmage/decks/kinnan.dck --gauntlet-fraction 0.5 --gauntlet-ai-skill 1 \
        --engine-threads 16 --device cuda --eval-every 25 --eval-games 256 --eval-ai-skill 3 \
        --keep-every 5 --min-free-gb 30 "${resume[@]}" "$@" >> "$OUT/$name.log" 2>&1
}

arm b2 artifacts/commander-cloned.pt configs/policy-final.yml --reference artifacts/commander-cloned.pt
arm c2 artifacts/commander-cloned.pt configs/policy-compare-control.yml
echo "== done $(date)"
