"""Search-free entity-policy CLI; the original pipeline uses legacy-* commands."""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

# ─── train ───────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    from magezero.policy.learner import PPOConfig, train_from_files
    import yaml
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    metrics = train_from_files(args.checkpoint, args.episodes, args.output,
                               PPOConfig(**raw.get("training", {})), args.device)
    print(json.dumps(metrics, indent=2))


def cmd_init(args: argparse.Namespace) -> None:
    import torch
    import yaml
    from magezero.policy.catalog import CardCatalog
    from magezero.policy.checkpoint import save_checkpoint
    from magezero.policy.model import EntityBeliefPolicy, ModelConfig, PrivilegedCritic
    catalog = CardCatalog.load(args.catalog)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    torch.manual_seed(args.seed)
    config = ModelConfig(**raw.get("model", {}), num_cards=catalog.num_cards, catalog_digest=catalog.digest)
    model = EntityBeliefPolicy(config, catalog.tensor(),
        catalog.role_tensor() if config.belief_mode == "statistical_roles" else None)
    critic = PrivilegedCritic(config)
    version = save_checkpoint(args.output, model, critic, catalog)
    print(f"Initialized {args.output}; policy={version}; actor parameters={sum(p.numel() for p in model.parameters()):,}")


def cmd_serve(args: argparse.Namespace) -> None:
    from magezero.policy.service import serve
    serve(args.checkpoint, args.host, args.port, args.episodes, args.device, args.max_batch,
          not args.no_belief, args.torch_threads, args.allow_reload)


def cmd_selfplay(args: argparse.Namespace) -> None:
    from magezero.policy.runner import run_selfplay
    run_selfplay(args.checkpoint, args.game, args.config, args.engine_script,
                 args.output, args.iterations, args.games, args.device, args.league, args.actor_device,
                 args.minutes * 60 if args.minutes is not None else None,
                 args.engine_threads, args.snapshot_every, args.actor_threads, args.actor_shards,
                 resume=args.resume, eval_every=args.eval_every, eval_games=args.eval_games,
                 eval_ai_skill=args.eval_ai_skill, eval_anchors=args.eval_anchor or (),
                 keep_episodes=args.keep_episodes, keep_every=args.keep_every,
                 min_free_gb=args.min_free_gb, max_retries=args.max_retries,
                 max_failure_fraction=args.max_failure_fraction)


def cmd_play_policy(args: argparse.Namespace) -> None:
    """Local XMage server whose "Computer - monte carlo" opponent is the policy."""
    import subprocess
    from magezero.policy.runner import _await_server, _launch_server, _policy_version, _stop_server
    xmage = Path(args.xmage).resolve()
    config = Path(args.server_config).resolve()
    for path in (xmage / "lib", config):
        if not path.exists():
            sys.exit(f"not found: {path}")
    log = xmage / f"policy-play-{args.port}.log"
    print(f"[play] starting policy service on port {args.port} (log: {log})")
    # No episode directory: games against people are never recorded for training.
    service = _launch_server(Path(args.checkpoint).resolve(), args.port, None, args.device, log, args.torch_threads)
    try:
        _await_server(service, _policy_version(args.checkpoint))
        print("[play] starting XMage server on localhost:17171. Connect with an XMage client, create a"
              " Two Player Duel, and add a \"Computer - monte carlo\" opponent (the policy) using a deck"
              " from the policy's card pool. Ctrl+C stops both.")
        subprocess.run(["java", f"-Xmx{args.heap}", "-Dh2.bindAddress=127.0.0.1",
                        f"-Dxmage.config.path={config}",
                        f"-Dmagezero.policy.endpoint=http://127.0.0.1:{args.port}",
                        f"-Dmagezero.policy.sample={'false' if args.greedy else 'true'}",
                        "-cp", "lib/*", "mage.server.Main"], cwd=xmage, check=False)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_server(service)


def cmd_legacy_train(args: argparse.Namespace) -> None:
    from magezero.util.config import load_all
    from magezero import runner
    run_cfg, cur_cfg = load_all(args.run)
    runner.run_pipeline(run_cfg, cur_cfg, base_game_yml=args.game, resume=args.resume)


# ─── batch ───────────────────────────────────────────────────

def cmd_batch(args: argparse.Namespace) -> None:
    from magezero import runner
    runner.launch_jvm(args.config)


# ─── play ────────────────────────────────────────────────────

def cmd_play(args: argparse.Namespace) -> None:
    from magezero import runner
    import subprocess
    from pathlib import Path

    config = Path(args.config).resolve()
    if not config.exists():
        sys.exit(f"config not found: {config}")

    deck = args.deck
    version = args.version
    if version is None:
        version = runner.latest_version(deck)
    server = None
    if version is None or not runner.has_checkpoint(deck, version):
    #if not runner.has_checkpoint(deck, version):
        print(f"model for {deck} is not found, falling back to offline MCTS")
        #sys.exit(f"no checkpoint at models/{deck}/ver{version}/model.pt.gz")
    else:
        # start inference server
        print(f"[play] starting inference server for {deck} v{version}")
        server = runner.start_server(deck, version, runner.PRIMARY_PORT, Path("."))

    try:
        # launch XMage server
        script = "xmage\\mz-xmage-play.bat" if sys.platform == "win32" else "xmage/mz-xmage-play.sh"
        cmd = ["cmd", "/c", script, str(config)] if sys.platform == "win32" else [script, str(config)]
        subprocess.run(cmd, check=True)
    finally:
        if server is not None:
            runner.stop_server(server)


# ─── import ──────────────────────────────────────────────────

def cmd_import(args: argparse.Namespace) -> None:
    from magezero import runner
    src = Path(args.file)
    if not src.exists():
        sys.exit(f"file not found: {src}")

    suffix = src.suffix.lower()
    if suffix == ".dck":
        dst = runner.DECKS_DIR / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"✓ imported deck → {dst}")

    elif suffix == ".mz":
        with zipfile.ZipFile(src) as zf:
            meta = json.loads(zf.read("metadata.json"))
            deck = meta["deck"]
            version = meta["version"]
            dst = Path("models") / deck / f"ver{version}"
            dst.mkdir(parents=True, exist_ok=True)
            zf.extract("model.pt.gz", dst)
        print(f"✓ imported model → {dst}")

    elif suffix == ".txt":
        dst = Path("xmage/decks") / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"✓ imported deck → {dst}")

    else:
        sys.exit(f"unknown file type: {suffix} (expected .dck, .mz, or .txt)")


# ─── export ──────────────────────────────────────────────────

def cmd_export(args: argparse.Namespace) -> None:
    src = Path("models") / args.deck / f"ver{args.version}"
    if not src.exists():
        sys.exit(f"model not found: {src}")

    model_file = src / "model.pt.gz"
    if not model_file.exists():
        sys.exit(f"missing model.pt.gz in {src}")

    out_dir = Path("exports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.deck}_v{args.version}.mz"

    metadata = {"deck": args.deck, "version": args.version}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_file, "model.pt.gz")
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    print(f"✓ exported → {out_path}")


# ─── main ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="mz")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize an entity/belief policy from an immutable card catalog")
    p_init.add_argument("--catalog", required=True)
    p_init.add_argument("--config", default="configs/policy.yml")
    p_init.add_argument("--output", required=True)
    p_init.add_argument("--seed", type=int, default=0)
    p_init.set_defaults(func=cmd_init)

    p_train = sub.add_parser("train", help="recurrent PPO on fresh structured policy trajectories")
    p_train.add_argument("--checkpoint", required=True)
    p_train.add_argument("--episodes", nargs="+", required=True)
    p_train.add_argument("--output", required=True)
    p_train.add_argument("--config", default="configs/policy.yml")
    p_train.add_argument("--device", default="auto", help="learner device; auto uses CUDA when present (faster for the batched learner)")
    p_train.set_defaults(func=cmd_train)

    p_serve = sub.add_parser("serve", help="serve recurrent legal-action inference without search")
    p_serve.add_argument("--checkpoint", required=True)
    p_serve.add_argument("--episodes", help="directory for completed on-policy episodes")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=50052)
    p_serve.add_argument("--device", default="cpu")
    p_serve.add_argument("--max-batch", type=int, default=32, help="most concurrent decisions per forward pass")
    p_serve.add_argument("--no-belief", action="store_true", help="omit diagnostic belief marginals from /act responses")
    p_serve.add_argument("--torch-threads", type=int, default=1, help="intra-op threads for the inference worker")
    p_serve.add_argument("--allow-reload", action="store_true", help="enable POST /reload for persistent collection runners")
    p_serve.set_defaults(func=cmd_serve)

    p_play_policy = sub.add_parser("play", help="host a local XMage server where you can play against a policy checkpoint")
    p_play_policy.add_argument("--checkpoint", required=True)
    p_play_policy.add_argument("--port", type=int, default=50052, help="policy service port")
    p_play_policy.add_argument("--device", default="cpu")
    p_play_policy.add_argument("--torch-threads", type=int, default=2)
    p_play_policy.add_argument("--greedy", action="store_true", help="play the most likely action instead of sampling")
    p_play_policy.add_argument("--xmage", default="xmage", help="XMage distribution with the patched lib/ jars")
    p_play_policy.add_argument("--server-config", default="integration/xmage/server-config.xml")
    p_play_policy.add_argument("--heap", default="4G", help="XMage server Java heap")
    p_play_policy.set_defaults(func=cmd_play_policy)

    p_selfplay = sub.add_parser("selfplay", help="collect and learn using the v2 XMage policy adapter")
    p_selfplay.add_argument("--checkpoint", required=True)
    p_selfplay.add_argument("--game", default="configs/policy-game.yml")
    p_selfplay.add_argument("--config", default="configs/policy.yml")
    p_selfplay.add_argument("--engine-script", required=True)
    p_selfplay.add_argument("--output", default="artifacts/selfplay")
    p_selfplay.add_argument("--iterations", type=int, default=10,
                            help="total generations for the run; --resume continues until this many exist")
    p_selfplay.add_argument("--games", type=int, default=32)
    p_selfplay.add_argument("--minutes", type=float, help="approximate wall-time budget; stop between complete generations")
    p_selfplay.add_argument("--device", default="auto", help="learner device; auto uses CUDA when present (faster for the batched learner)")
    p_selfplay.add_argument("--actor-device", default="cpu", help="inference device; --device selects the learner device")
    p_selfplay.add_argument("--league", help="persistent historical opponent league JSON")
    p_selfplay.add_argument("--engine-threads", type=int, help="parallel XMage games (1-16); overrides training.threads in --game")
    p_selfplay.add_argument("--snapshot-every", type=int, default=5, help="add the learner to the league every N generations")
    p_selfplay.add_argument("--actor-threads", type=int, default=4, help="torch intra-op threads per policy service")
    p_selfplay.add_argument("--actor-shards", type=int, default=1, help="policy service processes per checkpoint (1-8); games are spread across them")
    p_selfplay.add_argument("--resume", action="store_true", help="continue the run in --output from its run-state.json")
    p_selfplay.add_argument("--eval-every", type=int, default=0, help="benchmark every N generations (0 = never) and keep best.pt")
    p_selfplay.add_argument("--eval-games", type=int, default=64, help="games per benchmark opponent, seats/decks alternating")
    p_selfplay.add_argument("--eval-ai-skill", type=int, default=6, help="XMage built-in AI (ComputerPlayer7) skill 1-10; 0 disables it")
    p_selfplay.add_argument("--eval-anchor", action="append", help="frozen checkpoint benchmark (repeatable)")
    p_selfplay.add_argument("--keep-episodes", action="store_true", help="keep trajectory files after training (about 8.5 KB per decision)")
    p_selfplay.add_argument("--keep-every", type=int, default=50, help="keep every Nth generation checkpoint; others are pruned")
    p_selfplay.add_argument("--min-free-gb", type=float, default=20.0, help="stop cleanly between generations below this free disk space")
    p_selfplay.add_argument("--max-retries", type=int, default=3, help="engine restarts per generation before the run stops")
    p_selfplay.add_argument("--max-failure-fraction", type=float, default=0.25,
                            help="stop when more than this fraction of a generation's games error or time out")
    p_selfplay.set_defaults(func=cmd_selfplay)

    p_old_train = sub.add_parser("legacy-train", help="original MCTS pipeline (explicit opt-in)")
    p_old_train.add_argument("--run", default="configs/run.yml")
    p_old_train.add_argument("--game", default="configs/game.yml")
    p_old_train.add_argument("--resume", action="store_true")
    p_old_train.set_defaults(func=cmd_legacy_train)

    p_batch = sub.add_parser("legacy-batch", help="original JVM batch launcher")
    p_batch.add_argument("--config", default="configs/game.yml")
    p_batch.set_defaults(func=cmd_batch)

    p_play = sub.add_parser("legacy-play", help="original XMage player launcher")
    p_play.add_argument("--deck", required=True)
    p_play.add_argument("--version", type=int, default=None)
    p_play.add_argument("--config", default="configs/game.yml")
    p_play.set_defaults(func=cmd_play)

    p_import = sub.add_parser("legacy-import", help="import original .dck or .mz file")
    p_import.add_argument("file")
    p_import.set_defaults(func=cmd_import)

    p_export = sub.add_parser("legacy-export", help="export original model as .mz bundle")
    p_export.add_argument("--deck", required=True)
    p_export.add_argument("--version", type=int, required=True)
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
