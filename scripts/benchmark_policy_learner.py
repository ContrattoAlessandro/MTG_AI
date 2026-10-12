"""CPU/CUDA learner timing on the same complete recorded episode (no main-policy update)."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import torch
from magezero.policy.learner import PPOConfig, train_from_files
parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",required=True)
parser.add_argument("--episodes",required=True)
parser.add_argument("--output",required=True)
args=parser.parse_args()
root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
paths=list(Path(args.episodes).glob("*.jsonl"))
episode=min(paths,key=lambda p:len(json.loads(p.read_text())["steps"]))
states=len(json.loads(episode.read_text())["steps"])
torch.set_num_threads(1)
reports=[]
for device in ["cpu"]+(["cuda"] if torch.cuda.is_available() else []):
    torch.manual_seed(17)
    if device=="cuda":
        torch.cuda.init(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    metrics=train_from_files(args.checkpoint,[episode],root/f"profile-{device}.pt",
        PPOConfig(epochs=1,episodes_per_batch=1,sequence_length=32),device)
    if device=="cuda":torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    reports.append({"device":device,"seconds":seconds,"decision_passes_per_second":states/seconds,
        "peak_cuda_mb":torch.cuda.max_memory_allocated()/1024**2 if device=="cuda" else None,"metrics":metrics})
report={"episode":str(episode),"states":states,"torch_cpu_threads":1,"results":reports,
    "scope":"same complete episode, same initial checkpoint, one PPO epoch; includes checkpoint loading/saving, target replay, backward; may include concurrent evaluation load"}
(root/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
