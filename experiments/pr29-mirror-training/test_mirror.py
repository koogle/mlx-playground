import sys
from pathlib import Path
import chess
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
from mirror import mirror_states, permutation, encode, oracle_state


def test_oracle_reflection_and_policy_indices():
    p = permutation()
    boards = [
        chess.Board(),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("4k3/P7/8/3pP3/8/8/7p/4K3 w - d6 0 1"),
    ]
    for board in boards:
        mirrored = board.mirror()
        x = oracle_state(board, 20)
        np.testing.assert_array_equal(mirror_states(x), oracle_state(mirrored, 20))
        np.testing.assert_array_equal(mirror_states(mirror_states(x)), x)
        for move in board.legal_moves:
            reflected = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
            assert reflected in mirrored.legal_moves
            assert p[encode(move)] == encode(reflected)
    original = mx.array(np.arange(4672)[None, :])
    reflected = original[:, mx.array(p)]
    np.testing.assert_array_equal(np.asarray(reflected[:, mx.array(p)]), original)
