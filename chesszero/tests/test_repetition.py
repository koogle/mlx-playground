import unittest

import numpy as np

from chess_engine.bitboard import BitBoard


class TestRepetition(unittest.TestCase):
    def cycle_knights(self, board):
        for source, target in (
            ((0, 6), (2, 5)),
            ((7, 6), (5, 5)),
            ((2, 5), (0, 6)),
            ((5, 5), (7, 6)),
        ):
            self.assertTrue(board.make_move(source, target))

    def test_draw_on_third_occurrence_including_starting_position(self):
        board = BitBoard()
        self.assertFalse(board.is_draw())
        self.cycle_knights(board)
        self.assertFalse(board.is_draw())
        self.cycle_knights(board)
        self.assertTrue(board.is_draw())
        self.assertTrue(board.is_game_over())
        self.assertEqual(board.get_game_result(), 0.0)

    def test_repetition_ignores_counters_but_preserves_turn_and_castling(self):
        board = BitBoard()
        original = board.get_repetition_hash()
        changed = board.copy()
        changed.state[17:].fill(0.25)
        self.assertEqual(original, changed.get_repetition_hash())
        self.assertNotEqual(board.get_hash(), changed.get_hash())
        changed.state[12].fill(0)
        self.assertNotEqual(original, changed.get_repetition_hash())
        for plane in range(13, 17):
            changed = board.copy()
            changed.state[plane].fill(0)
            self.assertNotEqual(original, changed.get_repetition_hash())

    def test_rejected_move_does_not_change_repetition_history(self):
        board = BitBoard()
        history = board.position_history.copy()
        for _ in range(3):
            self.assertFalse(board.make_move((6, 0), (5, 0)))  # Wrong side to move.
        self.assertEqual(history, board.position_history)
        self.assertFalse(board.is_draw())

    def test_search_and_terminal_caches_distinguish_histories(self):
        repeated = BitBoard()
        self.cycle_knights(repeated)
        self.cycle_knights(repeated)
        fresh = repeated.copy()
        fresh.position_history = {fresh.get_repetition_hash(): 1}
        np.testing.assert_array_equal(repeated.state, fresh.state)
        self.assertNotEqual(repeated.get_hash(), fresh.get_hash())
        # Both lookup orders must work with the shared game-over cache.
        for boards in ((fresh, repeated), (repeated, fresh)):
            BitBoard._game_over_cache.clear()
            for board in boards:
                self.assertEqual(board.is_game_over(), board is repeated)

    def test_copy_keeps_history_independent(self):
        board = BitBoard()
        self.cycle_knights(board)
        clone = board.copy()
        self.assertEqual(board.position_history, clone.position_history)
        self.cycle_knights(clone)
        self.assertFalse(board.is_draw())
        self.assertTrue(clone.is_draw())

    def test_search_hash_includes_counts_of_other_positions(self):
        board = BitBoard()
        clone = board.copy()
        future = board.copy()
        self.assertTrue(future.make_move((0, 6), (2, 5)))
        clone.position_history[future.get_repetition_hash()] = 2
        self.assertEqual(board.get_repetition_hash(), clone.get_repetition_hash())
        self.assertNotEqual(board.get_hash(), clone.get_hash())


if __name__ == "__main__":
    unittest.main()
