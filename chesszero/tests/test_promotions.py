import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS, Node
from model.self_play import get_policy_distribution


class TestPromotionEncoding(unittest.TestCase):
    def setUp(self):
        self.search = MCTS(None, ModelConfig())
        for name in ("_tree_cache", "_transposition_table", "_all_moves_cache"):
            getattr(MCTS, name, {}).clear()

    def test_all_underpromotions_have_unique_reserved_indices(self):
        reserved = set()
        for from_row, to_row in ((6, 7), (1, 0)):
            for from_col in range(8):
                for to_col in range(max(0, from_col - 1), min(8, from_col + 2)):
                    source = (from_row, from_col)
                    indices = [
                        self.search.encode_move(source, (to_row, to_col, piece))
                        for piece in range(1, 5)
                    ]
                    self.assertEqual(len(set(indices)), 4)
                    ordinary = (from_row * 8 + from_col) * 64 + to_row * 8 + to_col
                    self.assertEqual(indices[3], ordinary)  # Queen keeps its old index.
                    for index in indices[:3]:
                        self.assertGreaterEqual(index, 4096)
                        self.assertLess(index, ModelConfig().policy_output_dim)
                        self.assertNotIn(index, reserved)
                        reserved.add(index)

    def test_ordinary_move_indices_are_unchanged(self):
        for source in range(64):
            for target in range(64):
                self.assertEqual(
                    self.search.encode_move(divmod(source, 8), divmod(target, 8)),
                    source * 64 + target,
                )

    def test_policy_targets_keep_each_promotion_visit_count(self):
        for source, target in (((6, 0), (7, 0)), ((1, 7), (0, 6))):
            with self.subTest(source=source):
                children = {
                    (source, (*target, piece)): SimpleNamespace(visit_count=piece)
                    for piece in range(1, 5)
                }
                policy = np.array(
                    get_policy_distribution(SimpleNamespace(children=children), 4672)
                )
                self.assertEqual(np.count_nonzero(policy), 4)
                for move, child in children.items():
                    index = self.search.encode_move(*move)
                    self.assertAlmostEqual(
                        float(policy[index]), child.visit_count / 10, places=6
                    )

    def test_expansion_reads_distinct_promotion_priors(self):
        for color, source, target in ((0, (6, 0), (7, 0)), (1, (1, 0), (0, 0))):
            with self.subTest(color=color):
                board = BitBoard()
                board.state[:12].fill(0)
                board.state[5, 0, 4] = 1
                board.state[11, 7, 4] = 1
                board.state[color * 6, source[0], source[1]] = 1
                board.state[12].fill(1 - color)
                board.state[13:17].fill(0)
                policy = np.zeros(4672, dtype=np.float32)
                for piece in range(1, 5):
                    index = self.search.encode_move(source, (*target, piece))
                    policy[index] = piece / 10
                root = Node(board)
                self.search._expand_node(root, mx.array(policy), 0.0)
                for piece in range(1, 5):
                    child = root.children[(source, (*target, piece))]
                    self.assertAlmostEqual(child.prior, piece / 10, places=6)
                    self.assertEqual(child.board.get_piece_at(*target), (color, piece))


if __name__ == "__main__":
    unittest.main()
