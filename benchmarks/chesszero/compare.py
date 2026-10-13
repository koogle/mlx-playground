"""Run identical benchmark workloads in isolated processes on an MLX Metal or CUDA GPU.

The script lives on the integration branch but imports each target checkout.
It does not modify either implementation or replace its chess/search/training code.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def worker(args):
    sys.path.insert(0, str(Path(args.repo).resolve() / "chesszero"))
    import gc
    import resource
    import tempfile
    from unittest.mock import patch

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from chess_engine.bitboard import BitBoard
    from config.model_config import ModelConfig
    from model.mcts import MCTS
    from model.network import ChessNet
    import model.training as training

    if args.backend == "metal" and (
        platform.system() != "Darwin" or not mx.metal.is_available()
    ):
        raise RuntimeError("Metal comparison requires an Apple Silicon Mac.")
    if args.backend == "cuda" and platform.system() != "Linux":
        raise RuntimeError("CUDA comparison requires Linux and an MLX CUDA build.")
    if hasattr(mx, "is_available") and not mx.is_available(mx.gpu):
        raise RuntimeError("MLX GPU is unavailable; CPU fallback is disabled.")
    mx.set_default_device(mx.gpu)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    overrides = dict(
        n_simulations=args.simulations,
        n_games_per_iteration=args.games,
        n_epochs=args.generations,
        eval_interval_epochs=args.generations + 1,
        batch_size=args.batch_size,
    )
    if args.profile == "smoke":
        overrides.update(n_residual_blocks=2, n_filters=64)
    config = ModelConfig(**overrides)
    import psutil

    revision_file = Path(args.repo) / "REVISION"
    revision = (
        revision_file.read_text().strip()
        if revision_file.exists()
        else subprocess.check_output(
            ["git", "-C", args.repo, "rev-parse", "HEAD"], text=True
        ).strip()
    )
    chip = (
        subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if args.backend == "metal"
        else subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    )
    result = {
        "commit": revision,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": chip,
        "backend": args.backend,
        "host_memory_bytes": psutil.virtual_memory().total,
        "python": sys.version,
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
        "config": dataclasses.asdict(config),
        "seed": args.seed,
        "task": args.task,
        "workers": args.workers,
        "status": "running",
    }
    output = Path(args.output)
    write_json(output, result)

    if args.task == "copy":
        board = BitBoard()
        # Fixed warm-up followed by independent repeats; copies are consumed.
        for _ in range(100):
            board.copy()
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            for _ in range(5000):
                clone = board.copy()
            samples.append((time.perf_counter() - start) / 5000)
            assert np.array_equal(clone.state, board.state)
        result["seconds_per_copy"] = samples
    elif args.task == "search":
        model = ChessNet(config)
        mx.eval(model.parameters())
        result["parameters"] = sum(
            x.size for _, x in tree_flatten(model.trainable_parameters())
        )

        class MeasuredModel:
            def __init__(self, model):
                self.model = model
                self.calls = 0
                self.positions = 0

            @property
            def training(self):
                return self.model.training

            def train(self, mode=True):
                self.model.train(mode)

            def eval(self):
                self.model.eval()

            def __call__(self, states):
                self.calls += 1
                self.positions += len(states)
                return self.model(states)

        measured = MeasuredModel(model)
        # Warm up Metal kernels without changing BatchNorm statistics.
        model.eval()
        mx.eval(model(mx.array(BitBoard().state)[None]))
        rows = []
        for repeat in range(3):
            for moves in (
                (),
                (((1, 4), (3, 4)), ((6, 4), (4, 4))),
                (((1, 3), (3, 3)), ((6, 3), (4, 3)), ((0, 1), (2, 2))),
            ):
                board = BitBoard()
                for move in moves:
                    assert board.make_move(*move)
                # Cold rule/tree caches give the same starting condition. Each
                # search then uses its implementation's caching behavior.
                for cls in (BitBoard, MCTS):
                    for name, cache in vars(cls).items():
                        if (
                            name.endswith("_cache")
                            and name not in ("_attack_pattern_cache", "_ray_cache")
                            and hasattr(cache, "clear")
                        ):
                            cache.clear()
                measured.calls = measured.positions = 0
                search = MCTS(measured, config)
                model.train()  # Match the main training worker's starting mode.
                start = time.perf_counter()
                move = search.get_move(board, temperature=0.0)
                elapsed = time.perf_counter() - start
                stack = [search.root_node]
                nodes = materialized = 0
                seen = set()
                while stack:
                    node = stack.pop()
                    if id(node) in seen:
                        continue
                    seen.add(id(node))
                    nodes += 1
                    materialized += (
                        node._board is not None if hasattr(node, "_board") else 1
                    )
                    stack.extend(node.children.values())
                rows.append(
                    {
                        "repeat": repeat,
                        "opening_plies": len(moves),
                        "seconds": elapsed,
                        "root_visits": search.root_node.visit_count,
                        "child_visits": sum(
                            c.visit_count for c in search.root_node.children.values()
                        ),
                        "network_calls": measured.calls,
                        "network_positions": measured.positions,
                        "nodes": nodes,
                        "materialized_boards": materialized,
                        "move": move,
                    }
                )
                del search
                gc.collect()
        result["searches"] = rows
    else:
        # Exercise the real Trainer loop, real complete self-play games and its
        # actual optimizer update frequency. No capped games receive draw labels.
        with tempfile.TemporaryDirectory(
            prefix="chesszero-benchmark-"
        ) as checkpoint_dir:
            trainer = training.Trainer(config, checkpoint_dir=checkpoint_dir)
            mx.eval(trainer.model.parameters())
            result["parameters"] = sum(
                x.size for _, x in tree_flatten(trainer.model.trainable_parameters())
            )
            generations = []
            original_generate = training.generate_games
            original_step = (
                trainer.train_on_batches
                if hasattr(trainer, "train_on_batches")
                else trainer.train_on_batch
            )

            def generate(*values, **kwargs):
                start = time.perf_counter()
                games = original_generate(*values, **kwargs)
                generations.append(
                    {
                        "generation": len(generations) + 1,
                        "self_play_seconds": time.perf_counter() - start,
                        "game_plies": [len(history) for history, _ in games],
                        "outcomes_white": [float(outcome) for _, outcome in games],
                        "positions": sum(len(history) for history, _ in games),
                        "training_seconds": 0.0,
                        "optimizer_updates": 0,
                        "losses": [],
                    }
                )
                result["generations"] = generations
                write_json(output, result)
                return games

            def step(*values, **kwargs):
                before = int(trainer.optimizer.state["step"])
                start = time.perf_counter()
                losses = original_step(*values, **kwargs)
                generations[-1]["training_seconds"] += time.perf_counter() - start
                generations[-1]["optimizer_updates"] += (
                    int(trainer.optimizer.state["step"]) - before
                )
                generations[-1]["losses"].append([float(value) for value in losses])
                write_json(output, result)
                return losses

            if hasattr(trainer, "train_on_batches"):
                trainer.train_on_batches = step
            else:
                trainer.train_on_batch = step
            start = time.perf_counter()
            with patch.object(training, "generate_games", generate):
                trainer.train(n_epochs=args.generations, n_workers=args.workers)
            result["total_training_wall_seconds"] = time.perf_counter() - start
            result["optimizer_step"] = int(trainer.optimizer.state["step"])
            result["all_parameters_finite"] = all(
                bool(mx.all(mx.isfinite(value)).item())
                for _, value in tree_flatten(trainer.model.parameters())
            )
            trainer.model.eval()
            policy, value = trainer.model(mx.array(BitBoard().state)[None])
            result["initial_position_value_after_training"] = float(value.item())
            result["initial_position_policy_entropy_after_training"] = float(
                (-mx.sum(policy * mx.log(mx.maximum(policy, 1e-12)))).item()
            )
    result["parent_peak_rss_bytes"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    result["parent_gpu_peak_bytes"] = (
        mx.get_peak_memory()
        if hasattr(mx, "get_peak_memory")
        else mx.metal.get_peak_memory()
    )
    result["status"] = "complete"
    write_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["metal", "cuda"], default="metal")
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=["smoke", "default"], default="smoke")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--timeout", type=int, default=1200, help="Seconds per isolated workload"
    )
    parser.add_argument("--task", choices=["copy", "search", "train"])
    parser.add_argument("--repo", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repo:
        worker(args)
        return
    if args.backend == "metal" and (
        platform.system() != "Darwin" or platform.machine() != "arm64"
    ):
        parser.error(
            "Run this comparison on an Apple Silicon Mac; Use --backend cuda explicitly for a CUDA GPU comparison."
        )
    if not args.baseline or not args.candidate:
        parser.error("--baseline and --candidate are required")
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    # Serial runs avoid resource contention; separate processes isolate caches,
    # allocator high-water marks, model mode and Python module imports.
    for task in [args.task] if args.task else ["copy", "search", "train"]:
        for label, repo in [("baseline", args.baseline), ("candidate", args.candidate)]:
            output = directory / f"{label}-{task}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--repo",
                str(Path(repo).resolve()),
                "--task",
                task,
                "--output",
                str(output),
                "--profile",
                args.profile,
                "--backend",
                args.backend,
            ]
            for name in (
                "simulations",
                "games",
                "generations",
                "workers",
                "batch_size",
                "seed",
            ):
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            print(f"Running {label} {task}", flush=True)
            env = dict(os.environ, PYTHONHASHSEED=str(args.seed), PYTHONUNBUFFERED="1")
            with (directory / f"{label}-{task}.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=args.timeout)
                    status = "complete" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    status = "timed_out"
            manifest.append(
                {"label": label, "task": task, "status": status, "result": output.name}
            )
            write_json(directory / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    if any(row["status"] != "complete" for row in manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
