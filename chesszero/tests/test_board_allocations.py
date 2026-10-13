import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS, Node


class TestBoardAllocations(unittest.TestCase):
    def test_expansion_defers_boards_and_reuses_materialized_child(self):
        board = BitBoard()
        board.freeze()
        search = MCTS(None, ModelConfig())
        root = Node(board, board_factory=search._get_child_board)
        search._expand_node(root, mx.ones(4672) / 4672, 0.0)
        self.assertEqual(len(root.children), 20)
        self.assertTrue(all(child._board is None for child in root.children.values()))
        move, child = next(iter(root.children.items()))
        with patch.object(
            BitBoard, "__init__", side_effect=AssertionError("initialized")
        ):
            expected = board.copy()
        self.assertTrue(expected.make_move(*move))
        actual = child.board
        self.assertIs(child.board, actual)
        np.testing.assert_array_equal(actual.state, expected.state)
        materialized = sum(node._board is not None for node in root.children.values())
        self.assertEqual(materialized, 1)
