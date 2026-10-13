"""Bounded snapshot-based self-play pipeline for the integration experiment."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np


def play_snapshot(repo, config_dict, snapshot, seed):
    """Each task owns its model; only CPU arrays cross process boundaries."""
    sys.path.insert(0, str(Path(repo) / "chesszero"))
    import mlx.core as mx
    from config.model_config import ModelConfig
    from chess_engine.bitboard import BitBoard
    from model.network import ChessNet
    from model.mcts import MCTS
    from model.self_play import get_policy_distribution

    mx.set_default_device(mx.gpu)
    mx.random.seed(seed)
    np.random.seed(seed)
    config = ModelConfig(**config_dict)
    model = ChessNet(config)
    model.load_weights(str(snapshot))
    model.eval()
    mx.eval(model.parameters())
    board = BitBoard()
    history = []
    while not board.is_game_over():
        search = MCTS(model, config)
        move = search.get_move(board, temperature=1.0)
        if move is None:
            raise RuntimeError("No move from a nonterminal self-play position")
        policy = get_policy_distribution(search.root_node, config.policy_output_dim)
        history.append((np.array(board.state, copy=True), np.array(policy), None))
        if not board.make_move(*move):
            raise RuntimeError("Illegal self-play move")
    result = board.get_game_result(perspective_color=0)
    if result is None:
        raise RuntimeError("Unfinished game cannot supply an outcome target")
    del model, search, policy
    mx.clear_cache()
    return history, result


class SelfPlayPipeline:
    """One batch in flight, immutable snapshots, deterministic game ordering."""

    def __init__(self, repo, config, directory, workers):
        self.repo, self.config = repo, asdict(config)
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
        self.pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        )
        self.pending = None

    def submit(self, model, batch, model_generation):
        if self.pending is not None:
            raise RuntimeError("Only one fresh batch may be queued")
        path = self.directory / f"batch-{batch:04d}.safetensors"
        model.save_weights(str(path))
        start = time.perf_counter()
        futures = [
            self.pool.submit(
                play_snapshot,
                self.repo,
                self.config,
                str(path),
                2026 + batch * self.config["n_games_per_iteration"] + game,
            )
            for game in range(self.config["n_games_per_iteration"])
        ]
        self.pending = path, futures, start, model_generation

    def collect(self, timeout=3600):
        path, futures, start, version = self.pending
        # Worker exceptions propagate instead of being silently counted as draws
        # or leaving the parent waiting forever for a missing queue result.
        deadline = time.perf_counter() + timeout
        games = [
            f.result(timeout=max(0, deadline - time.perf_counter())) for f in futures
        ]
        self.pending = None
        path.unlink()
        return games, version, time.perf_counter() - start

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
