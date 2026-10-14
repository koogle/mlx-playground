"""Bounded PR29 self-play experiment. Evaluation data never enters training."""

import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import shutil
import hashlib
import time

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from config.model_config import ModelConfig
from model.mcts import MCTS
from model.pipe_inference import run_pipe_games
from model.self_play import play_single_game, create_batches

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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def training_worker(client, config, game_id, generation, connection, workers):
    seed = 1000000 + generation * 1000 + game_id
    np.random.seed(seed)
    random.seed(seed)
    play_single_game(client, config, game_id, connection, workers)


def evaluation_worker(
    client, config, pair_id, color, bank_seed, random_only, connection, workers
):
    """CPU-only game with all candidate predictions served by the GPU parent."""
    import chess
    from chess_engine.bitboard import BitBoard

    try:
        rng = np.random.default_rng(bank_seed + pair_id)
        board, oracle = BitBoard(), chess.Board()
        moves = []
        search = MCTS(client, config, exploration_noise=False)
        started = time.monotonic()

        def apply(move):
            source, target = move
            ref = chess.Move(
                chess.square(source[1], source[0]),
                chess.square(target[1], target[0]),
                promotion=target[2] + 1 if len(target) == 3 else None,
            )
            if ref not in oracle.legal_moves:
                raise RuntimeError(f"Illegal native move {ref.uci()} at {oracle.fen()}")
            if not board.make_move(source, target):
                raise RuntimeError("Native engine rejected selected move")
            oracle.push(ref)
            moves.append(ref.uci())
            search.advance_root(move, board)

        for uci in OPENINGS[pair_id % len(OPENINGS)].split():
            apply(
                (
                    (int(uci[1]) - 1, ord(uci[0]) - 97),
                    (int(uci[3]) - 1, ord(uci[2]) - 97),
                )
            )
        while (
            not board.is_game_over()
            and len(moves) < 1024
            and time.monotonic() - started < 600
        ):
            if random_only or board.get_current_turn() != color:
                legal = [
                    (s, d)
                    for s, _ in board.get_all_pieces(board.get_current_turn())
                    for d in sorted(board.get_valid_moves(s))
                ]
                move = legal[int(rng.integers(len(legal)))]
            else:
                move = search.get_move(board, temperature=0)
            if move is None:
                raise RuntimeError("No move in ongoing evaluation game")
            apply(move)
        outcome = board.get_game_result(perspective_color=color)
        if outcome is not None:
            ref = oracle.outcome(claim_draw=True)
            expected = (
                None
                if ref is None
                else 0.0
                if ref.winner is None
                else 1.0
                if ref.winner == (color == 0)
                else -1.0
            )
            if outcome != expected:
                raise RuntimeError(
                    json.dumps(
                        dict(
                            native=outcome,
                            oracle=expected,
                            fen=oracle.fen(),
                            moves=moves,
                            halfmoves=board.get_moves_without_progress(),
                            repetition=board.is_repetition(),
                            insufficient=board._has_insufficient_material(),
                            stalemate=board.is_stalemate(board.get_current_turn()),
                        )
                    )
                )
        connection.send(
            dict(
                pair_id=pair_id,
                color=color,
                outcome=outcome,
                score=None if outcome is None else (outcome + 1) / 2,
                moves=moves,
                seconds=time.monotonic() - started,
                oracle_verified=True,
            )
        )
    finally:
        client.close()


def summarize(rows):
    scores = [r["score"] for r in rows]
    wins, losses, draws = (sum(s == value for s in scores) for value in (1, 0, 0.5))
    decisive = wins + losses
    p_value = (
        sum(math.comb(decisive, k) for k in range(wins, decisive + 1)) / 2**decisive
        if decisive
        else 1.0
    )
    # Treat censored games as losses for the success gate and all-game lower bound.
    lower_score = sum(s or 0 for s in scores) / len(rows)
    pairs = [
        np.mean([r["score"] or 0 for r in rows if r["pair_id"] == p])
        for p in sorted({r["pair_id"] for r in rows})
    ]
    boot = (
        np.random.default_rng(777)
        .choice(pairs, (10000, len(pairs)), replace=True)
        .mean(axis=1)
    )
    interval = np.quantile(boot, [0.025, 0.975]).tolist()
    return dict(
        games=len(rows),
        wins=wins,
        losses=losses,
        draws=draws,
        censored=sum(s is None for s in scores),
        score_lower_bound=lower_score,
        score_upper_bound=lower_score + sum(s is None for s in scores) / len(rows),
        decisive_one_sided_p=p_value,
        pair_bootstrap_95=interval,
        beats_random=all(s is not None for s in scores)
        and p_value < 0.05
        and interval[0] > 0.5,
    )


def evaluate(model, config, out, name, pairs, bank_seed, random_only=False):
    model.eval()
    jobs = [
        (config, p, color, bank_seed, random_only)
        for p in range(pairs)
        for color in (0, 1)
    ]
    print("EVALUATION_START", name, len(jobs), flush=True)
    rows, failures, stats = run_pipe_games(
        model, config, evaluation_worker, jobs, min(32, len(jobs))
    )
    write_json(out / f"{name}-games.json", rows)
    if failures or len(rows) != len(jobs):
        raise RuntimeError(
            f"Incomplete evaluation {name}: {len(rows)} games, {failures} failures"
        )
    result = dict(
        name=name, bank_seed=bank_seed, summary=summarize(rows), inference=stats
    )
    write_json(out / f"{name}.json", result)
    print("EVALUATION_COMPLETE", json.dumps(result), flush=True)
    return result


def main():
    import logging
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from model.training import Trainer
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    mx.random.seed(20261014)
    np.random.seed(20261014)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    config = ModelConfig(
        n_games_per_iteration=32,
        n_simulations=64,
        batch_size=64,
        learning_rate=1e-4,
        n_epochs=60,
    )
    write_json(
        out / "experiment.json",
        dict(
            source_revision=os.environ.get(
                "CHESSZERO_SOURCE_REVISION", "recorded-by-source-hashes"
            ),
            source_hashes={
                str(p.relative_to("/opt/chesszero")): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in Path("/opt/chesszero/chesszero").rglob("*.py")
            },
            baseline_run=os.environ.get("CHESSZERO_BASELINE_RUN", ""),
            preflight_run=os.environ.get("CHESSZERO_PREFLIGHT_RUN", ""),
            config=vars(config),
            workers=32,
            seed=20261014,
            training_budget_seconds=10800,
            job_budget_seconds=14400,
            protocol="Fresh complete self-play; one accumulated Adam update per generation; no replay. "
            "Probe every 10 generations; one final independent 64-game confirmation after "
            "a probe score >=0.60 or at budget. Draws worth 0.5; censored never draws. "
            "Success requires one-sided decisive binomial p<0.05 and paired bootstrap lower bound>0.5. "
            "Random gate does not establish improvement over initial MCTS model.",
        ),
    )
    trainer = Trainer(config, checkpoint_dir=str(out / "checkpoints"))
    mx.eval(trainer.model.parameters())
    baseline_run = os.environ.get("CHESSZERO_BASELINE_RUN")
    preflight_run = os.environ.get("CHESSZERO_PREFLIGHT_RUN")
    if baseline_run:
        source = Path("/results") / baseline_run
        state = json.loads((source / "checkpoints/state_epoch_0.json").read_text())
        for field in (
            "n_residual_blocks",
            "n_filters",
            "n_simulations",
            "policy_output_dim",
        ):
            assert state["config"][field] == getattr(config, field)
        trainer.model.load_weights(str(source / "checkpoints" / state["model_file"]))
        print("INITIAL_WEIGHTS_RESTORED", baseline_run, flush=True)
        initial = evaluate(trainer.model, config, out, "initial-vs-random", 32, 10000)
    else:
        initial = evaluate(trainer.model, config, out, "initial-vs-random", 32, 10000)
    trainer._save_checkpoint(0)
    if preflight_run:
        source = Path("/results") / preflight_run
        assert json.loads((source / "result.json").read_text())["passed"]
        random_baseline = json.loads(
            (source / "random-bank-regression.json").read_text()
        )
        assert (
            random_baseline["summary"]["games"] == 64
            and random_baseline["summary"]["censored"] == 0
        )
        random_baseline["name"] = "random-vs-random"
        write_json(out / "random-vs-random.json", random_baseline)
        shutil.copyfile(
            source / "random-bank-regression-games.json",
            out / "random-vs-random-games.json",
        )
        print("RANDOM_CONTROL_RESTORED", preflight_run, flush=True)
    else:
        random_baseline = evaluate(
            trainer.model, config, out, "random-vs-random", 32, 20000, random_only=True
        )
    generations = []
    for generation in range(1, 61):
        if generation > 1 and time.monotonic() - start > 10800:
            break
        trainer.model.eval()
        mx.clear_cache()
        print("GENERATION_START", generation, flush=True)
        began = time.monotonic()
        results, failures, stats = run_pipe_games(
            trainer.model,
            config,
            training_worker,
            [(config, i, generation) for i in range(32)],
            32,
        )
        if failures or len(results) != 32:
            raise RuntimeError(
                f"Incomplete self-play: {len(results)} games, {failures} failures"
            )
        games = [(r["history"], r["white_result"]) for r in results]
        row = dict(
            generation=generation,
            games=len(games),
            failures=failures,
            positions=sum(len(h) for h, _ in games),
            white_wins=sum(v == 1 for _, v in games),
            black_wins=sum(v == -1 for _, v in games),
            draws=sum(v == 0 for _, v in games),
            inference=stats,
            self_play_seconds=time.monotonic() - began,
        )
        del results
        began = time.monotonic()
        # Keep batches lazy: production Trainer.train currently lists CUDA batches.
        row["losses"] = trainer.train_on_batches(
            create_batches(games, config.batch_size)
        )
        row["learning_seconds"] = time.monotonic() - began
        row["adam_step"] = int(trainer.optimizer.state["step"])
        assert row["adam_step"] == generation
        assert all(
            bool(mx.all(mx.isfinite(v)).item())
            for _, v in tree_flatten(trainer.model.parameters())
        )
        trainer._save_checkpoint(generation)
        del games
        gc.collect()
        mx.clear_cache()
        generations.append(row)
        write_json(out / "generations.json", generations)
        print("GENERATION_COMPLETE", json.dumps(row), flush=True)
        if generation % 10 == 0:
            probe = evaluate(
                trainer.model, config, out, f"probe-{generation:04d}", 8, 30000
            )
            if probe["summary"]["score_lower_bound"] >= 0.60:
                break
    # Verify that the actual saved checkpoint reproduces current parameters and Adam state.
    restored, state = ChessNet.load_checkpoint(
        out / "checkpoints", generations[-1]["generation"]
    )
    assert int(state["optimizer_state"]["step"]) == generations[-1]["adam_step"]
    assert all(
        bool(mx.array_equal(a, b).item())
        for (_, a), (_, b) in zip(
            tree_flatten(trainer.model.parameters()),
            tree_flatten(restored.parameters()),
        )
    )
    del trainer, state
    gc.collect()
    mx.clear_cache()
    final = evaluate(restored, config, out, "trained-vs-random", 32, 10000)
    before = json.loads((out / "initial-vs-random-games.json").read_text())
    after = json.loads((out / "trained-vs-random-games.json").read_text())
    pair_deltas = []
    for p in range(32):
        pair_deltas.append(
            np.mean(
                [
                    (a["score"] or 0) - (b["score"] or 0)
                    for a, b in zip(after, before)
                    if a["pair_id"] == p
                ]
            )
        )
    bootstrap = (
        np.random.default_rng(888)
        .choice(pair_deltas, (10000, 32), replace=True)
        .mean(axis=1)
    )
    result = dict(
        status="completed",
        goal_met=final["summary"]["beats_random"],
        generations=len(generations),
        elapsed_seconds=time.monotonic() - start,
        initial=initial,
        random_baseline=random_baseline,
        final=final,
        trained_minus_initial_score=float(np.mean(pair_deltas)),
        trained_minus_initial_pair_bootstrap_95=np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        checkpoint_roundtrip=True,
    )
    write_json(out / "result.json", result)
    print("RUN_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
