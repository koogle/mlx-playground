import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS, Node


class TestBoardAllocations(unittest.TestCase):
    def test_copy_skips_initialization_and_keeps_mutable_state_independent(self):
        board = BitBoard()
        board.extra_immutable_metadata = (1, 2)
        board.position_history[123] = 1
        with patch.object(
            BitBoard, "__init__", side_effect=AssertionError("initialized")
        ):
            child = board.copy()
        np.testing.assert_array_equal(board.state, child.state)
        self.assertIsNot(board.state, child.state)
        self.assertIsNot(board.king_positions, child.king_positions)
        self.assertIsNot(board.position_history, child.position_history)
        self.assertEqual(child.extra_immutable_metadata, (1, 2))
        for name in ("knight_attacks", "king_attacks", "pawn_attacks"):
            self.assertIs(getattr(child, name), getattr(board, name))
            self.assertFalse(getattr(child, name).flags.writeable)
        child.state[0, 1, 0] = 0
        child.king_positions[0] = (2, 2)
        child.position_history.clear()
        self.assertEqual(board.state[0, 1, 0], 1)
        self.assertEqual(board.king_positions[0], (0, 4))
        self.assertNotEqual(child.position_history, board.position_history)

    def test_expansion_defers_boards_and_materializes_only_selected_child_once(self):
        board = BitBoard()
        search = MCTS(None, ModelConfig())
        root = Node(board)
        search._expand_node(root, mx.ones(4672) / 4672, 0.0)
        self.assertEqual(len(root.children), 20)
        self.assertTrue(all(child._board is None for child in root.children.values()))
        move, child = next(iter(root.children.items()))
        expected = board.copy()
        self.assertTrue(expected.make_move(*move))
        actual = child.board
        self.assertIs(child.board, actual)
        np.testing.assert_array_equal(actual.state, expected.state)
        self.assertEqual(actual.get_hash(), expected.get_hash())
        self.assertEqual(
            sum(node._board is not None for node in root.children.values()), 1
        )
        actual.state[0].fill(0)
        self.assertEqual(board.state[0].sum(), 8)

    def test_lazy_and_eager_search_have_identical_visits_and_values(self):
        from test_mcts_sequential import RecordingModel

        config = ModelConfig(n_simulations=16)
        lazy = MCTS(RecordingModel(), config, exploration_noise=False)
        eager = MCTS(RecordingModel(), config, exploration_noise=False)
        original_expand = eager._expand_node

        def expand(*args):
            original_expand(*args)
            for child in args[0].children.values():
                _ = child.board

        eager._expand_node = expand
        self.assertEqual(lazy.get_move(BitBoard()), eager.get_move(BitBoard()))

        def compare(left, right):
            self.assertEqual(left.visit_count, right.visit_count)
            self.assertAlmostEqual(left.value_sum, right.value_sum)
            self.assertEqual(left.children.keys(), right.children.keys())
            for move in left.children:
                compare(left.children[move], right.children[move])

        compare(lazy.root_node, eager.root_node)

    def test_search_owns_root_snapshot_when_game_advances(self):
        from test_mcts_sequential import RecordingModel

        board = BitBoard()
        search = MCTS(
            RecordingModel(), ModelConfig(n_simulations=1), exploration_noise=False
        )
        move = search.get_move(board)
        original = search.root_node.board.state.copy()
        self.assertTrue(board.make_move(*move))
        np.testing.assert_array_equal(search.root_node.board.state, original)
        self.assertIsNot(search.root_node.board, board)


if __name__ == "__main__":
    unittest.main()
