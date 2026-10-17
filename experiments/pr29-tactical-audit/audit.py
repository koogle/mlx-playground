"""Bounded CPU search audit against an exhaustive mate-in-two oracle."""

import json
from pathlib import Path
import random
import chess
import numpy as np
from config.model_config import ModelConfig
from model.mcts import MCTS
from test_board import from_fen


class ZeroUniform:
    def __call__(self, states):
        return np.full((len(states), 4672), 1 / 4672), np.zeros((len(states), 1))


def forced(board, attacker, depth):
    if board.is_checkmate():
        return board.turn != attacker
    if depth == 0 or board.is_stalemate() or board.is_insufficient_material():
        return False
    values = []
    attack_turn = board.turn == attacker
    for move in board.legal_moves:
        board.push(move)
        won = forced(board, attacker, depth - 1)
        board.pop()
        if attack_turn and won:
            return True
        if not attack_turn and not won:
            return False
        values.append(won)
    return bool(values) and (any(values) if attack_turn else all(values))


def winning_moves(board):
    result = []
    attacker = board.turn
    for move in board.legal_moves:
        board.push(move)
        won = forced(board, attacker, 2)
        board.pop()
        if won:
            result.append(move.uci())
    return result


def main():
    rng = random.Random(212016)
    positions = []
    for trial in range(400):
        wk, bk, q = rng.sample(range(64), 3)
        b = chess.Board(None)
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(q, chess.Piece(chess.QUEEN, chess.WHITE))
        if not b.is_valid() or b.is_game_over():
            continue
        moves = winning_moves(b)
        if moves:
            positions.append(b.fen())
            if len(positions) == 6:
                break
    assert len(positions) == 6
    out = []
    for fen in positions:
        original = chess.Board(fen)
        for oracle in (original, original.mirror()):
            expected = winning_moves(oracle)
            for budget in (64, 256):
                native, _ = from_fen(oracle.fen())
                search = MCTS(
                    ZeroUniform(),
                    ModelConfig(n_simulations=budget),
                    exploration_noise=False,
                )
                source, target = search.get_move(native, temperature=0)
                chosen = chess.Move(
                    chess.square(source[1], source[0]),
                    chess.square(target[1], target[0]),
                    promotion=target[2] + 1 if len(target) == 3 else None,
                )
                assert chosen in oracle.legal_moves
                row = dict(
                    fen=oracle.fen(),
                    budget=budget,
                    winning_moves=expected,
                    chosen=chosen.uci(),
                    solved=chosen.uci() in expected,
                    root_visits=search.root_node.visit_count,
                )
                out.append(row)
                print(json.dumps(row), flush=True)
    Path(__file__).with_name("result.json").write_text(
        json.dumps(
            dict(
                cases=out,
                source="python-chess exhaustive three-ply mate oracle",
                training_updates=0,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
