"""Assemble measured pilot results and export learning/speed figures."""
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path("artifacts/pilot-2026-10-12")
extension=Path("artifacts/pilot-2026-10-12-extension")
metrics_paths=sorted(root.glob("generation_*/metrics.json"))+sorted(extension.glob("generation_*/metrics.json"))
metrics=[json.loads(p.read_text()) for p in metrics_paths]
evaluations=[json.loads((root/name).read_text()) for name in ["evaluation-initial.json","evaluation-generation-0.json",
    "evaluation-generation-1.json","evaluation-generation-2.json","evaluation-generation-3.json","evaluation-final.json"]]
arenas={name:json.loads((root/folder/"summary.json").read_text()) for name,folder in {
    "initial_vs_upstream_search":"arena-initial-mcts","update3_vs_upstream_search":"arena-trained-mcts",
    "update4_vs_initial":"arena-trained-initial","final_vs_initial":"arena-final-initial"}.items()}
report={"date":"2026-10-12","timezone":"Europe/Rome","repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "engine_commit":subprocess.check_output(["git","-C",".mz_engine","rev-parse","HEAD"],text=True).strip(),
    "matchup":"simplegreen mirror","model_initialization_seed":20261012,
    "training_game_seeds":"stochastic, not fixed by current collector","generations":metrics,
    "total_training_games":sum(m["completed_games"] for m in metrics),
    "unique_recorded_decisions":sum(m["collected_decisions"] for m in metrics),
    "loss_bearing_decision_passes":sum(m["decisions"] for m in metrics),
    "summed_generation_seconds":sum(m["generation_seconds"] for m in metrics),
    "optimizer_steps":sum(m["epochs"]*((m["episodes"]+3)//4) for m in metrics),
    "heldout":evaluations,"arena":arenas,
    "final_checkpoint":str((extension/"generation_0002/policy.pt").resolve()),
    "learner_microbenchmark":json.loads((root/"learner-profile/report.json").read_text()) if (root/"learner-profile/report.json").exists() else None,
    "limitations":["Only one deck mirror and five small updates; 20 games are not a strength benchmark.",
        "Held-out prediction data contains four observer trajectories from two games, hence highly correlated states.",
        "Upstream comparison uses offline MCTS with no trained neural weights, 300-visit and 1-second caps.",
        "Public-information wrapper adds the upstream existing hidden-hand sampler at the root; this differs from raw upstream.",
        "Upstream search errors bias completed-game win rates; do not infer strength from these rates.",
        "Later training timings include concurrent baseline/evaluation work.",
        "Arena seeds/seats are balanced but random UUID ordering and stochastic action streams prevent exact paired trajectories."]}
Path("docs/pilot-2026-10-12.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
x=list(range(len(evaluations)))
axes[0].plot(x,[v["joint_nll"] for v in evaluations],marker="o",label="Learned belief")
axes[0].axhline(evaluations[0]["prior_joint_nll"],color="gray",linestyle="--",label="Known-deck prior")
axes[0].set(xlabel="Completed PPO generations",ylabel="Held-out joint NLL (nats)",title="Hand prediction · lower is better",xticks=x)
axes[0].legend(); axes[0].grid(alpha=.2)
axes[1].plot(x,[v["expected_count_mae"] for v in evaluations],marker="o",label="Learned belief")
axes[1].axhline(evaluations[0]["prior_expected_count_mae"],color="gray",linestyle="--",label="Known-deck prior")
axes[1].set(xlabel="Completed PPO generations",ylabel="Mean absolute error (copies/card)",title="Expected card counts",xticks=x)
axes[1].legend(); axes[1].grid(alpha=.2)
fig.savefig("docs/pilot-learning.png",dpi=160); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.6),layout="constrained")
ax.bar(range(1,len(metrics)+1),[m["collection_seconds"] for m in metrics],label="Game collection")
ax.bar(range(1,len(metrics)+1),[m["training_seconds"] for m in metrics],bottom=[m["collection_seconds"] for m in metrics],label="PPO learner")
ax.set(xlabel="Generation (4 games each)",ylabel="Wall time (seconds)",title="Measured runtime · later runs overlap evaluation",xticks=range(1,len(metrics)+1)); ax.legend()
fig.savefig("docs/pilot-speed.png",dpi=160); plt.close(fig)
print(json.dumps({"games":report["total_training_games"],"decisions":report["unique_recorded_decisions"],
    "optimizer_steps":report["optimizer_steps"],"minutes":report["summed_generation_seconds"]/60},indent=2))
