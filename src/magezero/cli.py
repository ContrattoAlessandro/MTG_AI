"""Command-line interface: the `mz` entry point.

    mz init            create a checkpoint for a card catalog
    mz teacher-collect record XMage's AI as behaviour-cloning trajectories
    mz clone           behaviour-clone a checkpoint (warm start)
    mz selfplay        self-play PPO training runs
    mz evaluate        score checkpoints against XMage's AI without training
    mz train           one PPO update from trajectory files
    mz serve           the policy inference service
    mz play            play against a checkpoint in the XMage client
    mz web             play against a checkpoint in the browser

Heavy imports happen inside each command, so `mz --help` stays fast.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _config_section(path: str | None, section: str) -> dict:
    """One top-level section of a YAML config file ({} without a file)."""
    import yaml

    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")).get(section, {})


def cmd_init(args: argparse.Namespace) -> None:
    import torch

    from .catalog import CardCatalog
    from .checkpoint import save_checkpoint
    from .model import EntityBeliefPolicy, ModelConfig, PrivilegedCritic

    catalog = CardCatalog.load(args.catalog)
    torch.manual_seed(args.seed)
    config = ModelConfig(
        **_config_section(args.config, "model"), num_cards=catalog.num_cards, catalog_digest=catalog.digest
    )
    roles = catalog.role_tensor() if config.belief_mode == "statistical_roles" else None
    model = EntityBeliefPolicy(config, catalog.tensor(), roles)
    critic = PrivilegedCritic(config)
    version = save_checkpoint(args.output, model, critic, catalog)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Initialized {args.output}; policy={version}; actor parameters={parameters:,}")


def cmd_teacher_collect(args: argparse.Namespace) -> None:
    from .teacher import collect_teacher_games

    summary = collect_teacher_games(
        args.checkpoint, args.game, args.engine_script, args.output, args.games, batch_games=args.batch_games,
        engine_threads=args.engine_threads, ai_skill=args.ai_skill, ai_think=args.ai_think, seats=args.seats,
        port=args.port, max_retries=args.max_retries,
    )  # fmt: skip
    print(json.dumps({k: v for k, v in summary.items() if k != "batches"}, indent=2))


def cmd_clone(args: argparse.Namespace) -> None:
    from .teacher import CloneConfig, behavior_cloning

    config = CloneConfig(**_config_section(args.config, "clone"))
    metrics = behavior_cloning(args.checkpoint, args.episodes, args.output, config, args.device)
    final = metrics["behavior_cloning"]
    best = final["history"][final["best_epoch"]]
    print(json.dumps({"output": args.output, "policy_version": metrics["policy_version"],
                      "best_epoch": final["best_epoch"], "train": best["train"],
                      "validation": best["validation"]}, indent=2))  # fmt: skip


def cmd_selfplay(args: argparse.Namespace) -> None:
    from .runner import run_selfplay

    run_selfplay(
        args.checkpoint, args.game, args.config, args.engine_script, args.output,
        iterations=args.iterations, games=args.games, device=args.device, league_path=args.league,
        actor_device=args.actor_device, max_seconds=args.minutes * 60 if args.minutes is not None else None,
        engine_threads=args.engine_threads, snapshot_every=args.snapshot_every, actor_threads=args.actor_threads,
        actor_shards=args.actor_shards, resume=args.resume, eval_every=args.eval_every, eval_games=args.eval_games,
        eval_ai_skill=args.eval_ai_skill, eval_anchors=args.eval_anchor or (), keep_episodes=args.keep_episodes,
        keep_every=args.keep_every, min_free_gb=args.min_free_gb, max_retries=args.max_retries,
        max_failure_fraction=args.max_failure_fraction, gauntlet=args.gauntlet or (),
        gauntlet_fraction=args.gauntlet_fraction, gauntlet_ai_skill=args.gauntlet_ai_skill,
        gauntlet_ai_think=args.gauntlet_ai_think, eval_greedy=args.eval_greedy, reference=args.reference,
    )  # fmt: skip


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .runner import evaluate_checkpoints

    entries = evaluate_checkpoints(
        args.checkpoint, args.game, args.engine_script, args.output, gauntlet=args.gauntlet or (),
        games=args.games, ai_skill=args.ai_skill, engine_threads=args.engine_threads, greedy=args.greedy,
        actor_device=args.device, actor_threads=args.torch_threads,
    )  # fmt: skip
    for entry in entries:
        scores = {name: result.get("score") for name, result in entry["report"].items()}
        print(f"{entry['checkpoint']}: mean score {entry['score']}; {json.dumps(scores)}")


def cmd_train(args: argparse.Namespace) -> None:
    from .learner import PPOConfig, train_from_files

    config = PPOConfig(**_config_section(args.config, "training"))
    metrics = train_from_files(
        args.checkpoint, args.episodes, args.output, config, args.device, reference=args.reference
    )
    print(json.dumps(metrics, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    from .service import serve

    serve(
        args.checkpoint,
        args.host,
        args.port,
        args.episodes,
        args.device,
        args.max_batch,
        not args.no_belief,
        args.torch_threads,
        args.allow_reload,
    )


def cmd_play(args: argparse.Namespace) -> None:
    """A local XMage server whose "Computer - monte carlo" opponent is the policy."""
    from .checkpoint import read_policy_version
    from .service_client import ServiceProcess

    xmage = Path(args.xmage).resolve()
    config = Path(args.server_config).resolve() if args.server_config else xmage / "config" / "magezero-server.xml"
    for path in (xmage / "mz" / "build.json", config):
        if not path.exists():
            sys.exit(f"not found: {path} (build the distribution with python integration/xmage/build.py)")
    log = xmage / f"policy-play-{args.port}.log"
    print(f"[play] starting policy service on port {args.port} (log: {log})")
    # No episode directory: games against people are never recorded for training.
    service = ServiceProcess(Path(args.checkpoint).resolve(), args.port, None, args.device, log, args.torch_threads)
    try:
        service.wait_ready(read_policy_version(args.checkpoint))
        print("[play] starting XMage server on localhost:17171. Connect with an XMage client, create a"
              " Two Player Duel, and add a \"Computer - monte carlo\" opponent (the policy) using a deck"
              " from the policy's card pool. Ctrl+C stops both.")  # fmt: skip
        subprocess.run(
            ["java", f"-Xmx{args.heap}", "-Dh2.bindAddress=127.0.0.1", f"-Dxmage.config.path={config}",
             f"-Dmagezero.policy.endpoint=http://127.0.0.1:{args.port}",
             f"-Dmagezero.policy.sample={'false' if args.greedy else 'true'}",
             "-cp", os.pathsep.join(("mz/*", "lib/*")), "mage.server.Main"],
            cwd=xmage, check=False,
        )  # fmt: skip
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()


def cmd_web(args: argparse.Namespace) -> None:
    """Play against the policy in a browser."""
    from .webplay import run

    run(args.checkpoint, xmage=args.xmage, port=args.port, policy_port=args.policy_port, device=args.device,
        heap=args.heap, torch_threads=args.torch_threads, open_browser=not args.no_browser)  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mz", description="Search-free MageZero policy: training and play.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="initialize a policy checkpoint from an immutable card catalog")
    p.add_argument("--catalog", required=True)
    p.add_argument("--config", default="configs/policy.yml", help="YAML with a model: section")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("teacher-collect", help="record XMage's AI decisions as behaviour-cloning trajectories")
    p.add_argument(
        "--checkpoint", required=True, help="any checkpoint of the run's card catalog (never chooses actions)"
    )
    p.add_argument("--game", required=True, help="game YAML with the decks (e.g. configs/commander-mirror.yml)")
    p.add_argument("--engine-script", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--games", type=int, required=True, help="total games; reruns continue a partial output")
    p.add_argument("--batch-games", type=int, default=32)
    p.add_argument("--engine-threads", type=int, help="parallel XMage games (1-16)")
    p.add_argument("--ai-skill", type=int, default=1, help="XMage AI skill (1-10) of the teachers")
    p.add_argument("--ai-think", type=int, default=0, help="cap on the AI's seconds per decision (0 = XMage default)")
    p.add_argument(
        "--seats",
        choices=("both", "a"),
        default="both",
        help="record both decks, or only player_a's deck against a plain XMage AI",
    )
    p.add_argument("--port", type=int, default=50060, help="recording service port")
    p.add_argument("--max-retries", type=int, default=3)
    p.set_defaults(func=cmd_teacher_collect)

    p = sub.add_parser("clone", help="behaviour-clone a checkpoint on teacher trajectories (warm start)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", nargs="+", required=True, help="teacher trajectory files or directories")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/policy-final.yml", help="YAML with a clone: section")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_clone)

    p = sub.add_parser("selfplay", help="self-play PPO generations through the XMage policy adapter")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--game", default="configs/commander-mirror.yml")
    p.add_argument("--config", default="configs/policy.yml", help="YAML with a training: section")
    p.add_argument("--engine-script", required=True)
    p.add_argument("--output", default="artifacts/selfplay")
    p.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="total generations for the run; --resume continues until this many exist",
    )
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--minutes", type=float, help="approximate wall-time budget; stop between complete generations")
    p.add_argument("--device", default="auto", help="learner device; auto uses CUDA when present")
    p.add_argument("--actor-device", default="cpu", help="inference device; --device selects the learner device")
    p.add_argument("--league", help="persistent historical opponent league JSON")
    p.add_argument(
        "--engine-threads", type=int, help="parallel XMage games (1-16); overrides training.threads in --game"
    )
    p.add_argument("--snapshot-every", type=int, default=5, help="add the learner to the league every N generations")
    p.add_argument("--actor-threads", type=int, default=4, help="torch intra-op threads per policy service")
    p.add_argument(
        "--actor-shards",
        type=int,
        default=1,
        help="policy service processes per checkpoint (1-8); games are spread across them",
    )
    p.add_argument("--resume", action="store_true", help="continue the run in --output from its run-state.json")
    p.add_argument(
        "--eval-every", type=int, default=0, help="benchmark every N generations (0 = never) and keep best.pt"
    )
    p.add_argument("--eval-games", type=int, default=64, help="games per benchmark opponent, seats/decks alternating")
    p.add_argument(
        "--eval-ai-skill", type=int, default=6, help="XMage built-in AI (ComputerPlayer7) skill 1-10; 0 disables it"
    )
    p.add_argument("--eval-anchor", action="append", help="frozen checkpoint benchmark (repeatable)")
    p.add_argument(
        "--eval-greedy", action="store_true", help="evaluate with the most likely action instead of sampling"
    )
    p.add_argument(
        "--keep-episodes", action="store_true", help="keep trajectory files after training (about 8.5 KB per decision)"
    )
    p.add_argument("--keep-every", type=int, default=50, help="keep every Nth generation checkpoint; others are pruned")
    p.add_argument(
        "--min-free-gb", type=float, default=20.0, help="stop cleanly between generations below this free disk space"
    )
    p.add_argument("--max-retries", type=int, default=3, help="engine restarts per generation before the run stops")
    p.add_argument(
        "--max-failure-fraction",
        type=float,
        default=0.25,
        help="stop when more than this fraction of a generation's games error or time out",
    )
    p.add_argument(
        "--gauntlet",
        nargs="+",
        metavar="DECK",
        help="opponent decks piloted by XMage's AI; the learner keeps player_a's deck",
    )
    p.add_argument(
        "--gauntlet-fraction",
        type=float,
        default=0.5,
        help="share of each generation's games played against the gauntlet",
    )
    p.add_argument("--gauntlet-ai-skill", type=int, default=3, help="XMage AI skill (1-10) piloting gauntlet decks")
    p.add_argument(
        "--gauntlet-ai-think",
        type=int,
        default=0,
        help="cap on the gauntlet AI's search seconds per decision (0 = XMage default, 3 s per skill)",
    )
    p.add_argument(
        "--reference", help="frozen behaviour-cloned checkpoint for the KL term (training.reference_kl_coefficient)"
    )
    p.set_defaults(func=cmd_selfplay)

    p = sub.add_parser("evaluate", help="score checkpoints against XMage's AI without training them")
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--output", required=True, help="folder for evaluations.jsonl, batch configs and logs")
    p.add_argument(
        "--game",
        default="configs/commander-mirror.yml",
        help="game YAML; the checkpoint plays player_a's deck, XMage's AI player_b's deck",
    )
    p.add_argument("--gauntlet", nargs="+", metavar="DECK", help="further decks for XMage's AI")
    p.add_argument("--engine-script", required=True)
    p.add_argument("--games", type=int, default=256, help="games per opponent deck, seats and starts alternating")
    p.add_argument("--ai-skill", type=int, default=3, help="XMage AI skill (1-10)")
    p.add_argument("--engine-threads", type=int, help="parallel XMage games (1-16); overrides training.threads")
    p.add_argument("--greedy", action="store_true", help="play the most likely action instead of sampling")
    p.add_argument("--device", default="cpu", help="inference device")
    p.add_argument("--torch-threads", type=int, default=1, help="torch intra-op threads of the policy service")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="one recurrent PPO update on fresh on-policy trajectories")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/policy.yml", help="YAML with a training: section")
    p.add_argument("--device", default="auto", help="learner device; auto uses CUDA when present")
    p.add_argument(
        "--reference", help="frozen behaviour-cloned checkpoint for the KL term (training.reference_kl_coefficient)"
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve", help="serve recurrent legal-action inference without search")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", help="directory for completed on-policy episodes")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=50052)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-batch", type=int, default=32, help="most concurrent decisions per forward pass")
    p.add_argument("--no-belief", action="store_true", help="omit diagnostic belief marginals from /act responses")
    p.add_argument("--torch-threads", type=int, default=1, help="intra-op threads for the inference worker")
    p.add_argument("--allow-reload", action="store_true", help="enable POST /reload for persistent collection runners")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("play", help="host a local XMage server where you can play against a policy checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=50052, help="policy service port")
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--greedy", action="store_true", help="play the most likely action instead of sampling")
    p.add_argument("--xmage", default="xmage", help="XMage distribution built by integration/xmage/build.py")
    p.add_argument("--server-config", help="XMage server config (default: config/magezero-server.xml in --xmage)")
    p.add_argument("--heap", default="4G", help="XMage server Java heap")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("web", help="play against a policy checkpoint in the browser")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8765, help="web server port (localhost only)")
    p.add_argument("--policy-port", type=int, default=50071, help="the bot's policy service port")
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--xmage", default="xmage", help="XMage distribution built by integration/xmage/build.py")
    p.add_argument("--heap", default="4G", help="engine Java heap")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.set_defaults(func=cmd_web)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
