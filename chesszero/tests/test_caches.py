import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from chess_engine.bitboard import BitBoard
from chess_engine.cache import LRUCache
from config.model_config import ModelConfig
from model.mcts import MCTS, Node


class TestCaches(unittest.TestCase):
    def test_lru_bounds_reads_updates_and_overwrites(self):
        cache = LRUCache(max_size=2)
        cache.update(a=1, b=2)
        self.assertEqual(cache.get("a"), 1)
        cache["c"] = 3
        self.assertNotIn("b", cache)
        cache["a"] = 4
        cache["d"] = 5
        self.assertEqual(dict(cache), {"a": 4, "d": 5})
        self.assertEqual(len(cache), 2)

    def test_expansions_do_not_share_model_priors_or_statistics(self):
        board = BitBoard()
        search = MCTS(None, ModelConfig())
        e4, d4 = ((1, 4), (3, 4)), ((1, 3), (3, 3))
        first_policy, second_policy = np.zeros(4672), np.zeros(4672)
        first_policy[search.encode_move(*e4)] = 1.0
        second_policy[search.encode_move(*d4)] = 1.0
        first = Node(board)
        search._expand_node(first, first_policy, 0.8)
        search.backup([first, first.children[e4]], -0.8)
        # Applies equally to a competing model or changed weights on the same model.
        second = Node(board.copy())
        search._expand_node(second, second_policy, -0.5)
        self.assertGreater(first.children[e4].prior, first.children[d4].prior)
        self.assertGreater(second.children[d4].prior, second.children[e4].prior)
        for move, child in second.children.items():
            self.assertIsNot(child, first.children[move])
            self.assertIs(child.parent, second)
            self.assertEqual(child.visit_count, 0)
            self.assertEqual(child.value_sum, 0.0)

    def test_discarded_tree_can_be_collected(self):
        search = MCTS(None, ModelConfig())
        root = Node(BitBoard())
        search._expand_node(root, np.ones(4672) / 4672, 0.0)
        reference = weakref.ref(root)
        del root, search
        gc.collect()
        self.assertIsNone(reference())

    def test_rule_caches_stay_bounded_and_preserve_results(self):
        caches = [LRUCache(max_size=2) for _ in range(4)]
        with (
            patch.object(BitBoard, "_valid_moves_cache", caches[0]),
            patch.object(BitBoard, "_pin_cache", caches[1]),
            patch.object(BitBoard, "_game_over_cache", caches[2]),
            patch.object(MCTS, "_all_moves_cache", caches[3]),
        ):
            board = BitBoard()
            search = MCTS(None, ModelConfig())
            for count in range(10):
                board.state[17].fill(count / 200)
                self.assertEqual(board.get_valid_moves((1, 0)), {(2, 0), (3, 0)})
                self.assertFalse(board.is_game_over())
                self.assertEqual(
                    sum(map(len, search._get_all_valid_moves(board).values())), 20
                )
                for cache in caches:
                    self.assertLessEqual(len(cache), 2)


if __name__ == "__main__":
    unittest.main()
