import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS, Node


class TestBoardAllocations(unittest.TestCase):
    def test_frozen_board_copy_skips_initialization_and_is_independent(self):
        board = BitBoard()
        board.freeze()
        with patch.object(
            BitBoard, "__init__", side_effect=AssertionError("initialized")
        ):
            child = board.copy()
        for name in ("knight_attacks", "king_attacks", "pawn_attacks"):
            self.assertIs(getattr(child, name), getattr(board, name))
            self.assertFalse(getattr(child, name).flags.writeable)
        self.assertTrue(child.make_move((1, 4), (3, 4)))
        self.assertEqual(board.get_piece_at(1, 4), (0, 0))
        child.king_positions[0] = (2, 2)
        child.position_history.clear()
        self.assertEqual(board.king_positions[0], (0, 4))
        self.assertEqual(len(board.position_history), 1)

    def test_expansion_defers_boards_and_reuses_materialized_child(self):
        board = BitBoard()
        root = Node(board)
        MCTS(None, ModelConfig())._expand_node(root, mx.ones(4672) / 4672, 0.0)
        self.assertEqual(len(root.children), 20)
        self.assertTrue(all(child._board is None for child in root.children.values()))
        move, child = next(iter(root.children.items()))
        expected = board.copy()
        self.assertTrue(expected.make_move(*move))
        actual = child.board
        self.assertIs(child.board, actual)
        np.testing.assert_array_equal(actual.state, expected.state)
        materialized = sum(node._board is not None for node in root.children.values())
        self.assertEqual(materialized, 1)
