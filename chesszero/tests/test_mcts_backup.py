import unittest

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS, Node


def make_node(player, **kwargs):
    board = BitBoard()
    board.state[12] = 1 if player == 0 else 0
    return Node(board, **kwargs)


class TestMCTSBackup(unittest.TestCase):
    def test_value_is_from_leaf_perspective_at_every_depth(self):
        search = MCTS(None, ModelConfig())
        for length in range(1, 6):
            for value in (-1.0, 0.0, 0.8):
                with self.subTest(length=length, value=value):
                    path = [make_node(index % 2) for index in range(length)]
                    search.backup(path, value, value_player=(length - 1) % 2)
                    for index, node in enumerate(path):
                        expected = value * (-1) ** (length - 1 - index)
                        self.assertAlmostEqual(node.value(), expected)
                        self.assertEqual(node.visit_count, 1)

    def test_explicit_player_can_differ_from_leaf_player(self):
        search = MCTS(None, ModelConfig())
        for root_player in (0, 1):
            for length in (2, 3, 4):
                with self.subTest(root_player=root_player, length=length):
                    path = [make_node((root_player + i) % 2) for i in range(length)]
                    # Express a favorable White outcome from either player's perspective.
                    search.backup(path, 0.8, value_player=0)
                    search.backup(path, -0.8, value_player=1)
                    for node in path:
                        expected = 0.8 if node.board.get_current_turn() == 0 else -0.8
                        self.assertAlmostEqual(node.value(), expected)
                        self.assertEqual(node.visit_count, 2)

    def test_repeated_visits_average_independent_leaf_evaluations(self):
        root, leaf = make_node(0), make_node(1)
        search = MCTS(None, ModelConfig())
        for value in (1.0, 0.5, -0.75):
            search.backup([root, leaf], value, value_player=1)
        self.assertEqual(root.visit_count, 3)
        self.assertEqual(leaf.visit_count, 3)
        self.assertAlmostEqual(leaf.value(), 0.25)
        self.assertAlmostEqual(root.value(), -0.25)

    def test_parent_prefers_child_that_is_bad_for_opponent(self):
        root = make_node(0)
        favorable = make_node(1, parent=root, prior=0.5)
        unfavorable = make_node(1, parent=root, prior=0.5)
        root.children = {"good": favorable, "bad": unfavorable}
        search = MCTS(None, ModelConfig())
        search.backup([root, favorable], -0.8, value_player=1)
        search.backup([root, unfavorable], 0.8, value_player=1)
        move, _ = root.select_child(c_puct=0.0)
        self.assertEqual(move, "good")


if __name__ == "__main__":
    unittest.main()
