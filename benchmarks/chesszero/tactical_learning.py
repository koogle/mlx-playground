"""Controlled learning on one forced two-ply chess ending and its color mirror.

The oracle validates fixtures/moves only. Fresh self-play targets come from native
MCTS and completed outcomes; supervised targets are a separate positive control.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import chess
import numpy as np

FEN = "7k/8/5K2/6Q1/8/8/8/8 b - - 0 1"


def native_move(move):
    return (
        (chess.square_rank(move.from_square), chess.square_file(move.from_square)),
        (chess.square_rank(move.to_square), chess.square_file(move.to_square)),
    )


def legal_moves(board):
    return sorted(
        (source, target)
        for source, _ in board.get_all_pieces(board.get_current_turn())
        for target in board.get_valid_moves(source)
    )


def fixture(mirror=False):
    from chess_engine.bitboard import BitBoard

    ref = chess.Board(FEN)
    if mirror:
        ref = ref.mirror()
    board = BitBoard()
    board.state.fill(0)
    board.king_positions = {}
    for square, piece in ref.piece_map().items():
        color = 0 if piece.color else 1
        row, col = chess.square_rank(square), chess.square_file(square)
        board.state[6 * color + piece.piece_type - 1, row, col] = 1
        if piece.piece_type == chess.KING:
            board.king_positions[color] = (row, col)
    board.state[12].fill(int(ref.turn))
    board.en_passant_target = None
    board.position_history = {board.get_repetition_hash(): 1}
    assert ref.is_valid()
    assert legal_moves(board) == sorted(native_move(m) for m in ref.legal_moves)
    forced = list(ref.legal_moves)
    assert len(forced) == 1
    first = native_move(forced[0])
    child = board.copy()
    assert child.make_move(*first)
    ref.push(forced[0])
    mates = []
    for move in list(ref.legal_moves):
        ref.push(move)
        if ref.is_checkmate():
            mates.append(native_move(move))
        ref.pop()
    assert len(mates) == 1
    assert legal_moves(child) == sorted(native_move(m) for m in ref.legal_moves)
    terminal = child.copy()
    assert terminal.make_move(*mates[0])
    assert terminal.get_game_result(perspective_color=child.get_current_turn()) == 1
    return board, child, first, mates[0]


def probe(model, fixtures):
    import mlx.core as mx
    from model.move_encoding import encode_move

    boards = [b for f in fixtures for b in f[:2]]
    states = mx.array(np.stack([b.state for b in boards]))
    before = model.parameters()
    answer = {}
    for mode in ("eval", "train"):
        model.eval() if mode == "eval" else model.train()
        policy, values = model(states)
        mx.eval(policy, values, model.parameters())
        policies, predictions = np.array(policy), np.array(values).reshape(-1)
        rows = []
        for i, f in enumerate(fixtures):
            board = f[1]
            legal = legal_moves(board)
            ids = [encode_move(*m) for m in legal]
            p = policies[2 * i + 1, ids]
            p = p / p.sum()
            mate_id = legal.index(f[3])
            rows.append(
                {
                    "mate_probability_legal": float(p[mate_id]),
                    "mate_probability_full": float(
                        policies[2 * i + 1, encode_move(*f[3])]
                    ),
                    "greedy_mate": legal[int(np.argmax(p))] == f[3],
                    "mate_rank": 1 + int(np.sum(p > p[mate_id])),
                    "legal_moves": len(legal),
                }
            )
        expected = np.array([-1, 1, -1, 1])
        answer[mode] = {
            "finishes": rows,
            "values": predictions.tolist(),
            "value_mse": float(np.mean((predictions - expected) ** 2)),
        }
        model.update(before)
    model.eval()
    return answer


def fresh_games(model, config, fixtures):
    import mlx.core as mx
    from model.mcts import MCTS
    from model.self_play import get_policy_distribution

    games, records = [], []
    for mirror, f in enumerate(fixtures):
        board = f[0].copy()
        oracle = chess.Board(FEN)
        if mirror:
            oracle = oracle.mirror()
        history, moves, target_mate_probability = [], [], None
        for ply in range(32):
            if board.is_game_over():
                break
            search = MCTS(model, config, exploration_noise=False)
            move = search.get_move(board, temperature=0)
            if move not in legal_moves(board):
                raise AssertionError(f"Illegal native search move {move}")
            policy = get_policy_distribution(search.root_node, config.policy_output_dim)
            if ply == 1:
                from model.move_encoding import encode_move

                target_mate_probability = float(policy[encode_move(*f[3])].item())
            history.append((mx.array(board.state), policy, None))
            source, target = move
            reference = chess.Move(
                chess.square(source[1], source[0]), chess.square(target[1], target[0])
            )
            assert reference in oracle.legal_moves
            oracle.push(reference)
            assert board.make_move(*move)
            moves.append(reference.uci())
        outcome = board.get_game_result(perspective_color=0)
        reference = oracle.outcome(claim_draw=True)
        if outcome is None:
            raise RuntimeError(
                "Controlled self-play failed to finish within 32 plies; no draw target fabricated."
            )
        assert reference is not None
        expected = 0 if reference.winner is None else 1 if reference.winner else -1
        assert outcome == expected
        games.append((history, float(outcome)))
        records.append(
            {
                "mirror": bool(mirror),
                "moves": moves,
                "white_result": outcome,
                "mate_target_probability": target_mate_probability,
            }
        )
    return games, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--bn-check", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.repo) / "chesszero"))
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from model.training import Trainer
    from model.self_play import create_batches
    from model.move_encoding import encode_move
    from config.model_config import ModelConfig

    mx.set_default_device(mx.gpu)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(
        n_residual_blocks=19,
        n_filters=256,
        n_simulations=64,
        batch_size=4,
        n_games_per_iteration=2,
        learning_rate=0.001,
    )
    fixtures = [fixture(False), fixture(True)]
    metadata = {
        "config": asdict(config),
        "fen": FEN,
        "steps": args.steps,
        "seeds": [2026] if args.bn_check else [2026, 2027],
        "bn_check": args.bn_check,
        "revision": (Path(args.repo) / "REVISION").read_text().strip(),
        "random_mate_in_one_probability": 1 / len(legal_moves(fixtures[0][1])),
        "scope": "one tiny ending and its color mirror; no generalization claim",
        "self_play": "fresh complete games; native MCTS, temperature zero, no root noise; no hand-labelled policy",
        "supervised": "separate positive control fitting the verified forced line; not a self-play claim",
    }
    (out / "experiment.json").write_text(json.dumps(metadata, indent=2))
    all_runs = []
    for seed in [2026] if args.bn_check else [2026, 2027]:
        for mode in ["self_play"] if args.bn_check else ["supervised", "self_play"]:
            mx.random.seed(seed)
            np.random.seed(seed)
            trainer = Trainer(config, checkpoint_dir=out / f"{seed}-{mode}")
            mx.eval(trainer.model.parameters())
            mx.clear_cache()
            run = {"seed": seed, "mode": mode, "rows": [], "games": []}
            start = time.perf_counter()

            def record(step, loss=None):
                row = {
                    "step": step,
                    "losses": loss,
                    "probe": probe(trainer.model, fixtures),
                    "optimizer_step": int(trainer.optimizer.state["step"]),
                    "elapsed_seconds": time.perf_counter() - start,
                }
                run["rows"].append(row)
                (out / f"{seed}-{mode}.json").write_text(json.dumps(run, indent=2))
                print(json.dumps({"seed": seed, "mode": mode, **row}), flush=True)

            record(0)
            for step in range(1, args.steps + 1):
                trainer.model.eval()
                if mode == "supervised":
                    boards = [b for f in fixtures for b in f[:2]]
                    moves = [m for f in fixtures for m in f[2:]]
                    targets = np.zeros((4, config.policy_output_dim), dtype=np.float32)
                    for i, move in enumerate(moves):
                        targets[i, encode_move(*move)] = 1
                    batch = (
                        mx.array(np.stack([b.state for b in boards])),
                        mx.array(targets),
                        mx.array([-1.0, 1.0, -1.0, 1.0]),
                    )
                    losses = trainer.train_on_batches([batch])
                else:
                    games, details = fresh_games(trainer.model, config, fixtures)
                    run["games"].append({"step": step, "games": details})
                    losses = trainer.train_on_batches(
                        create_batches(games, config.batch_size)
                    )
                if step in (1, 5, 10, 20, 40, 60, 80, args.steps):
                    record(step, list(losses))
            if args.bn_check:
                before = trainer.model.parameters()
                learned = dict(tree_flatten(trainer.model.trainable_parameters()))
                optimizer_step = int(trainer.optimizer.state["step"])
                states = mx.array(np.stack([b.state for f in fixtures for b in f[:2]]))
                trainer.model.train()
                for _ in range(100):
                    outputs = trainer.model(states)
                    mx.eval(outputs, trainer.model.parameters())
                after_probe = probe(trainer.model, fixtures)
                identical = all(
                    bool(mx.all(v == learned[k]).item())
                    for k, v in tree_flatten(trainer.model.trainable_parameters())
                )
                assert (
                    identical and int(trainer.optimizer.state["step"]) == optimizer_step
                )
                run["bn_recalibration"] = {
                    "passes": 100,
                    "before": run["rows"][-1]["probe"],
                    "after": after_probe,
                    "learned_weights_unchanged": identical,
                    "optimizer_step_unchanged": optimizer_step,
                }
                trainer.model.update(before)
                trainer.model.eval()
            run["finite_parameters"] = all(
                bool(mx.all(mx.isfinite(v)).item())
                for _, v in tree_flatten(trainer.model.parameters())
            )
            (out / f"{seed}-{mode}.json").write_text(json.dumps(run, indent=2))
            all_runs.append(run)
            trainer = None
            mx.clear_cache()
    (out / "results.json").write_text(json.dumps(all_runs, indent=2))


if __name__ == "__main__":
    main()
