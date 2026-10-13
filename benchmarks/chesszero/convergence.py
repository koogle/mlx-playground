"""Bounded fresh-self-play learning experiment; evaluation games never train it."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
import shutil
from pathlib import Path
import sys
import time

import numpy as np

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6",
    "e2e4 c7c5 g1f3 d7d6",
    "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5",
    "d2d4 d7d5 c2c4 e7e6",
    "d2d4 g8f6 c2c4 g7g6",
    "d2d4 g8f6 c2c4 e7e6",
    "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 g2g3 g8f6",
    "b2b3 e7e5 c1b2 b8c6",
    "e2e4 d7d5 e4d5 d8d5",
    "e2e4 g8f6 e4e5 f6d5",
    "d2d4 f7f5 g2g3 g8f6",
    "c2c4 c7c5 g1f3 g8f6",
    "g2g3 d7d5 f1g2 e7e5",
    "d2d4 d7d5 g1f3 g8f6",
]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def summarize(rows):
    """Bootstrap opening pairs, not correlated individual color-swapped games."""
    complete = [r for r in rows if r["score"] is not None]
    pairs = []
    for opening in sorted({r["opening_id"] for r in rows}):
        pair = [r["score"] for r in complete if r["opening_id"] == opening]
        if len(pair) == 2:
            pairs.append(float(np.mean(pair)))
    interval = None
    if pairs:
        rng = np.random.default_rng(771)
        sampled = rng.choice(pairs, (10000, len(pairs)), replace=True).mean(axis=1)
        interval = np.quantile(sampled, [0.025, 0.975]).tolist()
    total = len(rows)
    points = sum(r["score"] for r in complete)
    return {
        "games": total,
        "completed": len(complete),
        "censored": total - len(complete),
        "wins": sum(r["score"] == 1 for r in complete),
        "draws": sum(r["score"] == 0.5 for r in complete),
        "losses": sum(r["score"] == 0 for r in complete),
        "score": points / len(complete) if complete else None,
        "complete_pairs": len(pairs),
        "pair_bootstrap_95_percent": interval,
        "censored_score_bounds": [
            points / total,
            (points + total - len(complete)) / total,
        ]
        if total
        else None,
    }


def evaluation_init(repo, config, candidate_path, initial_path):
    sys.path.insert(0, str(Path(repo) / "chesszero"))
    import mlx.core as mx
    from config.model_config import ModelConfig
    from model.network import ChessNet

    global EVAL_MODELS, EVAL_CONFIG
    mx.set_default_device(mx.gpu)
    EVAL_CONFIG = ModelConfig(**config)
    EVAL_MODELS = []
    for path in (candidate_path, initial_path):
        model = ChessNet(EVAL_CONFIG)
        model.load_weights(path)
        model.eval()
        mx.eval(model.parameters())
        EVAL_MODELS.append(model)


def evaluation_game(task):
    import chess  # Independent experiment validator; never selects a move.
    from chess_engine.bitboard import BitBoard
    from model.mcts import MCTS

    opening_id, candidate_color, opponent = task
    seed = 9000 + opening_id
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    board = BitBoard()
    oracle = chess.Board()
    moves = []
    for uci in OPENINGS[opening_id].split():
        source = (int(uci[1]) - 1, ord(uci[0]) - 97)
        target = (int(uci[3]) - 1, ord(uci[2]) - 97)
        if not board.make_move(source, target):
            raise RuntimeError(f"Illegal opening move: {uci}")
        oracle.push_uci(uci)
        moves.append([source, target])
    start = time.perf_counter()
    for _ in range(512):
        if board.is_game_over():
            break
        our_turn = board.get_current_turn() == candidate_color
        if not our_turn and opponent == "random":
            legal = [
                (source, dest)
                for source, _ in board.get_all_pieces(board.get_current_turn())
                for dest in sorted(board.get_valid_moves(source))
            ]
            move = legal[int(rng.integers(len(legal)))]
        else:
            model = EVAL_MODELS[0 if our_turn else 1]
            move = MCTS(model, EVAL_CONFIG, exploration_noise=False).get_move(
                board, temperature=0
            )
        if move is not None:
            source, target = move
            reference_move = chess.Move(
                chess.square(source[1], source[0]),
                chess.square(target[1], target[0]),
                promotion=target[2] + 1 if len(target) == 3 else None,
            )
            if reference_move not in oracle.legal_moves:
                cached_moves = sorted(board.get_valid_moves(source))
                BitBoard._valid_moves_cache.clear()
                fresh_moves = sorted(board.get_valid_moves(source))
                raise RuntimeError(
                    json.dumps(
                        {
                            "illegal_move": reference_move.uci(),
                            "fen": oracle.fen(),
                            "king_positions": board.king_positions,
                            "cached_moves": cached_moves,
                            "fresh_moves": fresh_moves,
                            "native_pieces": {
                                str(color): board.get_all_pieces(color)
                                for color in (0, 1)
                            },
                            "moves": moves,
                            "opening_id": opening_id,
                            "candidate_color": candidate_color,
                        }
                    )
                )
            oracle.push(reference_move)
        if move is None or not board.make_move(*move):
            raise RuntimeError(
                "Search returned no legal move in a nonterminal position"
            )
        moves.append(move)
    outcome = board.get_game_result(perspective_color=candidate_color)
    if outcome is not None:
        reference = oracle.outcome(claim_draw=True)
        expected = (
            None
            if reference is None
            else 0.0
            if reference.winner is None
            else 1.0
            if reference.winner == (candidate_color == 0)
            else -1.0
        )
        if outcome != expected:
            raise RuntimeError(
                f"Result mismatch: native={outcome}, oracle={expected}, {oracle.fen()}"
            )
    return {
        "opening_id": opening_id,
        "candidate_color": candidate_color,
        "opponent": opponent,
        "score": (outcome + 1) / 2 if outcome is not None else None,
        "plies": len(moves),
        "seconds": time.perf_counter() - start,
        "moves": moves,
    }


def evaluate(
    repo, config, out, generation, initial_path, candidate_path, pairs, workers
):
    summaries = {}
    for opponent in ["random"] if generation == 0 else ["initial", "random"]:
        tasks = [(i, color, opponent) for i in range(pairs) for color in (0, 1)]
        rows = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=evaluation_init,
            initargs=(repo, asdict(config), str(candidate_path), str(initial_path)),
        ) as pool:
            for row in pool.map(evaluation_game, tasks):
                rows.append(row)
                write_json(out / f"evaluation-{generation:04d}-{opponent}.json", rows)
        summaries[opponent] = summarize(rows)
        print(
            json.dumps(
                {
                    "evaluation_generation": generation,
                    "opponent": opponent,
                    **summaries[opponent],
                }
            ),
            flush=True,
        )
    write_json(out / f"scores-{generation:04d}.json", summaries)
    return summaries


def training_games(games):
    """Convert CPU-only IPC records to production Trainer input arrays."""
    import mlx.core as mx

    return [
        (
            [
                (mx.array(state), mx.array(policy), value)
                for state, policy, value in history
            ],
            outcome,
        )
        for history, outcome in games
    ]


def value_mode_probe(model, games):
    """Measure BatchNorm mode disagreement without updating weights or statistics."""
    import mlx.core as mx

    samples = [
        (state, result if state[12, 0, 0] == 1 else -result)
        for history, result in games
        for state, _, _ in history
    ]
    chosen = np.linspace(0, len(samples) - 1, min(64, len(samples)), dtype=int)
    states = mx.array(np.stack([samples[i][0] for i in chosen]))
    targets = mx.array([samples[i][1] for i in chosen])
    before = model.parameters()
    metrics = {}
    try:
        for mode in ("eval", "train"):
            model.eval() if mode == "eval" else model.train()
            _, values = model(states)
            mx.eval(values, model.parameters())
            metrics[mode + "_value_mse"] = float(
                mx.mean((values.reshape(-1) - targets) ** 2)
            )
    finally:
        model.update(before)
        model.eval()
        mx.eval(model.parameters())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games-per-generation", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--training-deadline-seconds", type=int, default=4500)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--resume-generation", type=int, default=10)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.repo) / "chesszero"))
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from config.model_config import ModelConfig
    from model.self_play import generate_games, create_batches
    from model.training import Trainer

    mx.set_default_device(mx.gpu)
    mx.random.seed(2026)
    np.random.seed(2026)
    config = ModelConfig(
        n_residual_blocks=19,
        n_filters=256,
        n_simulations=args.simulations,
        n_games_per_iteration=args.games_per_generation,
        batch_size=64,
        learning_rate=0.001,
        eval_interval_epochs=args.generations + 1,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    generations = []
    if args.resume_run:
        checkpoint_dir = out / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        source_dir = args.resume_run / "checkpoints"
        manifest_name = f"state_epoch_{args.resume_generation}.json"
        manifest = json.loads((source_dir / manifest_name).read_text())
        # Older experimental runs stored a removed MCTS tuning option.
        manifest["config"].pop("mcts_batch_size", None)
        write_json(checkpoint_dir / manifest_name, manifest)
        for key in (
            "model_file",
            "optimizer_file",
            "best_model_file",
            "best_optimizer_file",
        ):
            if manifest.get(key):
                shutil.copy2(source_dir / manifest[key], checkpoint_dir / manifest[key])
        trainer = Trainer(
            config,
            checkpoint_dir=out / "checkpoints",
            resume_epoch=args.resume_generation,
        )
        config = trainer.config
        generations = [
            row
            for row in json.loads((args.resume_run / "generations.json").read_text())
            if row["generation"] <= args.resume_generation
        ]
        for pattern in ("initial.safetensors", "evaluation-*.json", "scores-*.json"):
            for path in args.resume_run.glob(pattern):
                shutil.copy2(path, out / path.name)
        write_json(out / "generations.json", generations)
    else:
        trainer = Trainer(config, checkpoint_dir=out / "checkpoints")
    mx.eval(trainer.model.parameters())
    metadata = {
        "revision": (Path(args.repo) / "REVISION").read_text().strip(),
        "config": asdict(config),
        "seed": 2026,
        "workers": args.workers,
        "requested_generations": args.generations,
        "openings": OPENINGS,
        "evaluation_ply_cap": 512,
        "capped_evaluation_policy": "censored, never draw",
        "training": "fresh games only, no replay; one Adam step per generation",
        "selection": "no checkpoint rejection or evaluation-game training",
        "worker_rng": "seeded by batch and game"
        if args.pipeline
        else "production nondeterminism",
        "pipeline": args.pipeline,
        "maximum_model_lag_updates": 1 if args.pipeline else 0,
        "resume_run": str(args.resume_run) if args.resume_run else None,
        "resume_generation": args.resume_generation if args.resume_run else None,
        "release_unused_gpu_cache_between_phases": True,
    }
    write_json(out / "experiment.json", metadata)
    initial_path = out / "initial.safetensors"
    if not args.resume_run:
        trainer.model.save_weights(str(initial_path))
    start = time.perf_counter()
    elapsed_offset = generations[-1]["elapsed_seconds"] if generations else 0
    if not args.resume_run:
        evaluate(
            args.repo, config, out, 0, initial_path, initial_path, 16, args.workers
        )
    pipeline = None
    if args.pipeline:
        from parallel_selfplay import SelfPlayPipeline

        pipeline = SelfPlayPipeline(args.repo, config, out / "snapshots", args.workers)
        pipeline.submit(trainer.model, len(generations) + 1, len(generations))
    for generation in range(len(generations) + 1, args.generations + 1):
        if time.perf_counter() - start > args.training_deadline_seconds and not (
            pipeline and pipeline.pending
        ):
            break
        trainer.model.eval()
        mx.clear_cache()
        game_start = time.perf_counter()
        if pipeline:
            games, source_generation, self_play_seconds = pipeline.collect()
            games = training_games(games)
            # Snapshot BEFORE learning: generation N+1 overlaps update N and
            # uses weights N-1. No mutable model is shared with a game worker.
            if (
                generation < args.generations
                and time.perf_counter() - start < args.training_deadline_seconds
            ):
                pipeline.submit(trainer.model, generation + 1, generation - 1)
        else:
            games = generate_games(trainer.model, config, args.workers)
            source_generation = generation - 1
            self_play_seconds = time.perf_counter() - game_start
        collection_wait_seconds = time.perf_counter() - game_start
        learning_start = time.perf_counter()
        losses = trainer.train_on_batches(create_batches(games, config.batch_size))
        learning_seconds = time.perf_counter() - learning_start
        finite = all(
            bool(mx.all(mx.isfinite(v)).item())
            for _, v in tree_flatten(trainer.model.parameters())
        )
        row = {
            "generation": generation,
            "positions": sum(len(h) for h, _ in games),
            "game_plies": [len(h) for h, _ in games],
            "outcomes_white": [float(r) for _, r in games],
            "losses": losses,
            "self_play_seconds": self_play_seconds,
            "collection_wait_seconds": collection_wait_seconds,
            "self_play_model_generation": source_generation,
            "learning_seconds": learning_seconds,
            "optimizer_step": int(trainer.optimizer.state["step"]),
            "finite_parameters": finite,
            "elapsed_seconds": elapsed_offset + time.perf_counter() - start,
        }
        if generation == 1 or generation % 10 == 0:
            row["value_mode_probe"] = value_mode_probe(trainer.model, games)
        del games
        generations.append(row)
        write_json(out / "generations.json", generations)
        print(json.dumps(row), flush=True)
        if not finite:
            raise RuntimeError("Nonfinite model parameters")
        trainer._save_checkpoint(generation)
        # Keep the newest two resumable checkpoints plus every 50th milestone.
        for manifest_path in (out / "checkpoints").glob("state_epoch_*.json"):
            epoch = int(manifest_path.stem.rsplit("_", 1)[1])
            if epoch < generation - 1 and epoch % 50 != 0:
                manifest = json.loads(manifest_path.read_text())
                for key in ("model_file", "optimizer_file"):
                    if manifest.get(key):
                        (manifest_path.parent / manifest[key]).unlink(missing_ok=True)
                manifest_path.unlink()
        if generation % args.eval_every == 0 and generation != args.generations:
            weights = out / f"evaluation-weights-{generation:04d}.safetensors"
            trainer.model.save_weights(str(weights))
            mx.clear_cache()
            evaluate(
                args.repo,
                config,
                out,
                generation,
                initial_path,
                weights,
                8,
                min(args.workers, 4),
            )
    if pipeline:
        pipeline.close()
    final_generation = len(generations)
    trainer._save_checkpoint(final_generation)
    final_path = out / "final.safetensors"
    trainer.model.save_weights(str(final_path))
    mx.clear_cache()
    scores = evaluate(
        args.repo,
        config,
        out,
        final_generation,
        initial_path,
        final_path,
        16,
        args.workers,
    )
    write_json(
        out / "completion.json",
        {
            "completed_generations": final_generation,
            "elapsed_seconds": time.perf_counter() - start,
            "final_scores": scores,
            "stopped_at_training_deadline": final_generation < args.generations,
        },
    )


if __name__ == "__main__":
    main()
