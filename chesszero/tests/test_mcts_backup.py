import unittest

from config.model_config import ModelConfig
from model.mcts import MCTS, Node


class TestMCTSBackup(unittest.TestCase):
    def test_value_is_from_leaf_perspective_at_every_depth(self):
        search = MCTS(None, ModelConfig())
        for length in range(1, 6):
            for value in (-1.0, 0.0, 0.8):
                with self.subTest(length=length, value=value):
                    path = [Node(None) for _ in range(length)]
                    search.backup(path, value)
                    for index, node in enumerate(path):
                        expected = value * (-1) ** (length - 1 - index)
                        self.assertAlmostEqual(node.value(), expected)
                        self.assertEqual(node.visit_count, 1)

    def test_repeated_visits_average_independent_leaf_evaluations(self):
        root, leaf = Node(None), Node(None)
        search = MCTS(None, ModelConfig())
        for value in (1.0, 0.5, -0.75):
            search.backup([root, leaf], value)
        self.assertEqual(root.visit_count, 3)
        self.assertEqual(leaf.visit_count, 3)
        self.assertAlmostEqual(leaf.value(), 0.25)
        self.assertAlmostEqual(root.value(), -0.25)

    def test_parent_prefers_child_that_is_bad_for_opponent(self):
        root = Node(None)
        favorable = Node(None, parent=root, prior=0.5)
        unfavorable = Node(None, parent=root, prior=0.5)
        root.children = {"good": favorable, "bad": unfavorable}
        search = MCTS(None, ModelConfig())
        search.backup([root, favorable], -0.8)
        search.backup([root, unfavorable], 0.8)
        move, _ = root.select_child(c_puct=0.0)
        self.assertEqual(move, "good")


if __name__ == "__main__":
    unittest.main()
