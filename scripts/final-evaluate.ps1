# Final acceptance evaluation of the final run (detached, survives the launching session):
# 400 greedy games per checkpoint against XMage's AI at skill 3, piloting Winota (mirror)
# and Kinnan, seats and starts alternating. Results: artifacts/final/evaluation/evaluations.jsonl
#   powershell -ExecutionPolicy Bypass -File scripts/final-evaluate.ps1
$root = Split-Path -Parent $PSScriptRoot
$run = 'artifacts/final/run'
$arguments = @(
    'evaluate', '--checkpoint',
    "$run/best.pt", "$run/generation_0500/policy.pt", "$run/generation_0600/policy.pt",
    "$run/generation_0850/policy.pt", "$run/generation_0999/policy.pt",
    '--output', 'artifacts/final/evaluation', '--game', 'configs/commander-mirror.yml',
    '--gauntlet', 'xmage/decks/kinnan.dck', '--engine-script', 'xmage/mz-policy.bat',
    '--games', '400', '--ai-skill', '3', '--engine-threads', '16', '--greedy', '--device', 'cpu'
)
$process = Start-Process -FilePath "$root\.venv\Scripts\mz.exe" -ArgumentList $arguments `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$root\artifacts\final\evaluation.out" -RedirectStandardError "$root\artifacts\final\evaluation.err"
"Evaluation started: PID $($process.Id). Log: artifacts/final/evaluation.out"
