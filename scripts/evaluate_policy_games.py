"""Evaluate fixed policies or the pinned upstream offline MCTS reference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from magezero.checkpoint import load_checkpoint
from magezero.runner import _start_server, _stop_server

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",required=True)
parser.add_argument("--opponent-checkpoint")
parser.add_argument("--opponent",choices=["policy","mcts-public","mcts-raw"],default="policy")
parser.add_argument("--games",type=int,default=8)
parser.add_argument("--budget",type=int,default=300)
parser.add_argument("--timeout",type=float,default=1.)
parser.add_argument("--output",required=True)
parser.add_argument("--seed",type=int,default=20261012)
parser.add_argument("--threads",type=int,default=2)
parser.add_argument("--record-episodes",action="store_true",help="save evaluation trajectories for fixed-history behavior diagnostics; never train on these")
args=parser.parse_args()
if args.games <= 0 or not 1 <= args.threads <= 16:
    raise ValueError("games must be positive and threads must be in 1..16")
root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
if (root/"results.json").exists(): raise ValueError("choose a fresh output directory")
jars=sorted(Path("xmage/lib").glob("*.jar"))
classpath=";".join(str(p.resolve()) for p in jars)
sources=Path("integration/xmage/src/org/mage/magezero/PolicyArena.java").resolve()
subprocess.run(["javac","--release","8","-proc:none","-Xlint:-options","-cp",classpath,"-d",str(root/"classes"),str(sources)],check=True)
_,_,_,data=load_checkpoint(args.checkpoint)
processes=[]; started=time.perf_counter()
try:
    processes.append(_start_server(Path(args.checkpoint).resolve(),50054,root/"episodes-learner" if args.record_episodes else None,"cpu",root/"learner.log",data["policy_version"]))
    if args.opponent=="policy":
        if not args.opponent_checkpoint: raise ValueError("policy opponent requires a checkpoint")
        _,_,_,other=load_checkpoint(args.opponent_checkpoint)
        processes.append(_start_server(Path(args.opponent_checkpoint).resolve(),50055,root/"episodes-opponent" if args.record_episodes else None,"cpu",root/"opponent.log",other["policy_version"]))
    command=["java","-Xmx8G","-Dh2.bindAddress=127.0.0.1","-cp",str(root/"classes")+";"+classpath,
        "org.mage.magezero.PolicyArena","--deck",str(Path("xmage/decks/simplegreen.dck").resolve()),
        "--games",str(args.games),"--threads",str(args.threads),"--seed",str(args.seed),"--opponent",args.opponent,
        "--port-a","50054","--port-b","50055",
        "--budget",str(args.budget),"--timeout",str(args.timeout),"--output",str(root/"results.json")]
    with (root/"engine.log").open("w",encoding="utf-8") as log:
        subprocess.run(command,cwd="xmage",check=True,stdout=log,stderr=log,timeout=max(600,args.games*15))
finally:
    for process in processes: _stop_server(process)
report=json.loads((root/"results.json").read_text()); games=report["games"]
terminal=[g for g in games if g.get("terminal") and "finish_error" not in g]
wins=sum(g["result"]==1 for g in terminal); losses=sum(g["result"]==-1 for g in terminal); draws=sum(g["result"]==0 for g in terminal)
n=len(terminal); rate=wins/n if n else 0; z=1.96; denom=1+z*z/max(n,1)
center=(rate+z*z/(2*max(n,1)))/denom
half=z*((rate*(1-rate)/max(n,1)+z*z/(4*max(n,1)**2))**.5)/denom
summary={"checkpoint":str(Path(args.checkpoint).resolve()),"opponent":args.opponent,"opponent_checkpoint":args.opponent_checkpoint,
    "requested_games":args.games,"terminal_games":n,"wins":wins,"losses":losses,"draws":draws,"errors_or_truncations":len(games)-n,
    "win_rate":rate,"win_rate_wilson_95":[max(0,center-half),min(1,center+half)],"wall_seconds":time.perf_counter()-started,
    "mean_game_seconds":sum(g["seconds"] for g in games)/len(games),"mcts_visit_cap":args.budget,"mcts_seconds_cap":args.timeout,
    "seed":args.seed,"threads":args.threads,"recorded_evaluation_episodes":args.record_episodes,
    "scope":"fixed-opponent evaluation; paired seeds/seats do not ensure identical UUID ordering or stochastic action streams"}
(root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
