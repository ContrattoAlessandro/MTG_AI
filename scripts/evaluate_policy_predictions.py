import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from magezero.diagnostics import evaluate_predictions, trace_decision

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", nargs="+", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--trace")
args = parser.parse_args()
paths = [file for item in args.episodes for file in (sorted(Path(item).glob("*.jsonl")) if Path(item).is_dir() else [Path(item)])]
report = evaluate_predictions(args.checkpoint, paths)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
if args.trace: trace_decision(args.checkpoint, paths[0], args.trace)
print(json.dumps(report, indent=2))
