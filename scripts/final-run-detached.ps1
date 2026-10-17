# Start (or resume) the final run as a detached Windows process, so it survives
# the terminal or Claude session that launched it. Same command as final-run.sh.
#   powershell -ExecutionPolicy Bypass -File scripts/final-run-detached.ps1
$root = Split-Path -Parent $PSScriptRoot
$arguments = @(
    'scripts/final_run.py', '--output', 'artifacts/final/run', '--',
    '--checkpoint', 'artifacts/commander-cloned-20.pt', '--reference', 'artifacts/commander-cloned-20.pt',
    '--game', 'configs/commander-mirror.yml', '--config', 'configs/final.yml', '--engine-script', 'xmage/mz-policy.bat',
    '--iterations', '1000', '--games', '64',
    '--gauntlet', 'xmage/decks/kinnan.dck', '--gauntlet-fraction', '0.5', '--gauntlet-ai-skill', '1',
    '--engine-threads', '16', '--device', 'cuda',
    '--eval-every', '25', '--eval-games', '64', '--eval-ai-skill', '3', '--eval-greedy',
    '--eval-anchor', 'artifacts/anchors/a5-b-gen24.pt',
    '--keep-every', '50', '--min-free-gb', '30'
)
$process = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList $arguments `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$root\artifacts\final\detached.out" -RedirectStandardError "$root\artifacts\final\detached.err"
"Supervisor started: PID $($process.Id). Progress: artifacts/final/run/status.log"
