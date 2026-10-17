"""Color/rank reflection, preserving side-to-move outcomes."""

import numpy as np
import chess
from model.move_encoding import encode_move


def mirror_states(states):
    channels = [*range(6, 12), *range(6), 12, 15, 16, 13, 14, 17, 18]
    out = np.take(np.asarray(states)[..., ::-1, :], channels, axis=-3)
    out[..., 12, :, :] = 1 - out[..., 12, :, :]
    return out


def permutation():
    p = np.arange(4672)
    for source in range(64):
        for target in range(64):
            p[source * 64 + target] = (source ^ 56) * 64 + (target ^ 56)
    p[4096:4480] = 4096 + (np.arange(384) + 192) % 384
    assert np.array_equal(p[p], np.arange(4672))
    return p


def encode(move):
    target = divmod(move.to_square, 8)
    if move.promotion:
        target += (move.promotion - 1,)
    return encode_move(divmod(move.from_square, 8), target)


def oracle_state(board, ply):
    x = np.zeros((19, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        x[piece.piece_type - 1 + (0 if piece.color else 6), rank, file] = 1
    x[12].fill(board.turn)
    for color in (0, 1):
        x[13 + 2 * color].fill(board.has_kingside_castling_rights(not color))
        x[14 + 2 * color].fill(board.has_queenside_castling_rights(not color))
    x[17].fill(ply / 200)
    x[18].fill(board.halfmove_clock / 100)
    return x


def verify(rows, arrays, masks):
    states, policies, values = arrays
    reflected = mirror_states(states)
    p = permutation()
    np.testing.assert_array_equal(mirror_states(reflected), states)
    index = 0
    for row in rows:
        board = chess.Board()
        for ply, uci in enumerate(row["moves"]):
            mirrored = board.mirror()
            np.testing.assert_allclose(
                reflected[index], oracle_state(mirrored, ply), atol=1e-6
            )
            expected = np.zeros(4672, dtype=bool)
            for move in mirrored.legal_moves:
                expected[encode(move)] = True
            np.testing.assert_array_equal(masks[index][p], expected)
            assert not np.any(policies[index][p][~expected] > 0)
            mirrored_white_result = -row["white_result"]
            assert values[index] == (
                mirrored_white_result if mirrored.turn else -mirrored_white_result
            )
            board.push_uci(uci)
            index += 1
    assert index == len(states)
