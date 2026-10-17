"""Independently reconstruct exact legal action sets and verify array alignment."""

import chess
import numpy as np
from model.move_encoding import encode_move


def build_masks(rows, arrays):
    states, policies, values = arrays
    masks = np.zeros(policies.shape, dtype=bool)
    index = 0
    for row in rows:
        board = chess.Board()
        for ply, uci in enumerate(row["moves"]):
            expected = np.zeros((19, 8, 8), dtype=np.float32)
            for square, piece in board.piece_map().items():
                rank, file = divmod(square, 8)
                expected[
                    piece.piece_type - 1 + (0 if piece.color else 6), rank, file
                ] = 1
            expected[12].fill(board.turn)
            for color in (0, 1):
                expected[13 + 2 * color].fill(
                    board.has_kingside_castling_rights(not color)
                )
                expected[14 + 2 * color].fill(
                    board.has_queenside_castling_rights(not color)
                )
            expected[17].fill(ply / 200)
            expected[18].fill(board.halfmove_clock / 100)
            np.testing.assert_allclose(states[index], expected, atol=1e-6)
            assert values[index] == (
                row["white_result"] if board.turn else -row["white_result"]
            )
            for move in board.legal_moves:
                source = divmod(move.from_square, 8)
                target = divmod(move.to_square, 8)
                if move.promotion:
                    target += (move.promotion - 1,)
                masks[index, encode_move(source, target)] = True
            assert masks[index].any()
            assert np.isfinite(policies[index]).all() and (policies[index] >= 0).all()
            assert np.isclose(policies[index].sum(), 1, atol=1e-5)
            assert not np.any(policies[index][~masks[index]] > 0)
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)
            index += 1
        outcome = board.outcome(claim_draw=True)
        assert outcome is not None
        assert row["white_result"] == (
            0 if outcome.winner is None else (1 if outcome.winner else -1)
        )
    assert index == len(states)
    return masks
