import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from chess_engine.bitboard import BitBoard
from cachetools import LRUCache
from config.model_config import ModelConfig
from model.mcts import MCTS, Node


class TestCaches(unittest.TestCase):
    def test_lru_bounds_reads_updates_and_overwrites(self):
        cache = LRUCache(maxsize=2)
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
        search.backup([first, first.children[e4]], -0.8, value_player=1)
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

    def test_hash_collision_cannot_reuse_another_positions_legal_moves(self):
        with (
            patch("chess_engine.bitboard.hash", return_value=7, create=True),
        ):
            first = BitBoard()
            second = first.copy()
            second.state[0, 1, 0] = 0
            second.state[0, 2, 0] = 1
            search = MCTS(None, ModelConfig())
            self.assertEqual(first.get_valid_moves((1, 0)), {(2, 0), (3, 0)})
            self.assertEqual(second.get_valid_moves((1, 0)), set())
            self.assertIn((1, 0), search._get_all_valid_moves(first))
            self.assertNotIn((1, 0), search._get_all_valid_moves(second))
            self.assertIn((2, 0), search._get_all_valid_moves(second))
            self.assertNotEqual(first.get_hash(), second.get_hash())
            self.assertIsInstance(hash(first), int)

    def test_discarded_tree_can_be_collected(self):
        search = MCTS(None, ModelConfig())
        root = Node(BitBoard())
        search._expand_node(root, np.ones(4672) / 4672, 0.0)
        reference = weakref.ref(root)
        del root, search
        gc.collect()
        self.assertIsNone(reference())

    def test_rule_caches_stay_bounded_and_preserve_results(self):
        caches = [LRUCache(maxsize=2) for _ in range(4)]
        board = BitBoard()
        search = MCTS(None, ModelConfig())
        search._rule_caches = tuple(caches[:3])
        search._all_moves_cache = caches[3]
        search._attach_rule_caches(board)
        for count in range(10):
            board.state[17].fill(count / 200)
            self.assertEqual(board.get_valid_moves((1, 0)), {(2, 0), (3, 0)})
            self.assertFalse(board.is_game_over())
            self.assertEqual(
                sum(map(len, search._get_all_valid_moves(board).values())), 20
            )
            for cache in caches:
                self.assertLessEqual(len(cache), 2)

    def test_successor_reuse_skips_board_copy_and_move(self):
        board = BitBoard()
        search = MCTS(None, ModelConfig())
        move = ((1, 4), (3, 4))
        child = search._get_child_board(board, *move)
        with patch.object(board, "copy", side_effect=AssertionError("rebuilt board")):
            self.assertIs(search._get_child_board(board, *move), child)
        with self.assertRaisesRegex(ValueError, "read-only"):
            child.make_move((6, 4), (4, 4))
        with self.assertRaises(ValueError):
            child.state[0, 0, 0] = 1
        with self.assertRaises(TypeError):
            child.position_history[b"other"] = 1
        writable = child.copy()
        self.assertTrue(writable.make_move((6, 4), (4, 4)))
        self.assertNotEqual(writable.get_hash(), child.get_hash())

    def test_board_reuse_preserves_independent_nodes_and_priors(self):
        board = BitBoard()
        search = MCTS(None, ModelConfig())
        first, second = Node(board), Node(board.copy())
        first_policy, second_policy = np.ones(4672), np.ones(4672)
        move = ((1, 4), (3, 4))
        first_policy[search.encode_move(*move)] = 100
        search._expand_node(first, first_policy, 0.8)
        search.backup([first, first.children[move]], -0.8, value_player=1)
        with patch.object(
            BitBoard, "copy", side_effect=AssertionError("rebuilt board")
        ):
            search._expand_node(second, second_policy, -0.5)
        a, b = first.children[move], second.children[move]
        self.assertIs(a.board, b.board)
        self.assertIsNot(a, b)
        self.assertIs(a.parent, first)
        self.assertIs(b.parent, second)
        self.assertEqual(b.visit_count, 0)
        self.assertEqual(b.value_sum, 0)
        self.assertGreater(a.prior, b.prior)

    def test_repetition_history_is_not_merged_or_cached_as_same_result(self):
        first = BitBoard()
        second = first.copy()
        second.position_history[second.get_hash()] = 3
        self.assertEqual(first.get_hash(), second.get_hash())
        self.assertFalse(first.is_game_over())
        self.assertTrue(second.is_game_over())
        search = MCTS(None, ModelConfig())
        move = ((1, 4), (3, 4))
        a = search._get_child_board(first, *move)
        with patch.object(second, "copy", side_effect=AssertionError("rebuilt board")):
            b = search._get_child_board(second, *move)
        self.assertIs(a.state, b.state)
        self.assertIsNot(a, b)
        self.assertNotEqual(a.position_history, b.position_history)

    def test_real_chess_transposition_reuses_state_with_correct_path_history(self):
        first, second = BitBoard(), BitBoard()
        a = [((0, 6), (2, 5)), ((7, 6), (5, 5)), ((0, 1), (2, 2)), ((7, 1), (5, 2))]
        b = [a[2], a[3], a[0], a[1]]
        for board, moves in ((first, a), (second, b)):
            for move in moves:
                self.assertTrue(board.make_move(*move))
        self.assertEqual(first.get_hash(), second.get_hash())
        self.assertNotEqual(first.position_history, second.position_history)
        move = ((1, 4), (3, 4))
        expected = second.copy()
        self.assertTrue(expected.make_move(*move))
        search = MCTS(None, ModelConfig())
        original = search._get_child_board(first, *move)
        with patch.object(
            second, "copy", side_effect=AssertionError("rebuilt transposition")
        ):
            reused = search._get_child_board(second, *move)
        self.assertIs(original.state, reused.state)
        np.testing.assert_array_equal(reused.state, expected.state)
        self.assertEqual(reused.position_history, expected.position_history)
        self.assertNotEqual(original.position_history, reused.position_history)
        # The same physical successor can have a terminal result on only one path.
        second.position_history[reused.get_hash()] = 3
        repeated = search._get_child_board(second, *move)
        self.assertIs(original.state, repeated.state)
        self.assertFalse(original.is_game_over())
        self.assertTrue(repeated.is_game_over())

    def test_board_cache_eviction_reconstructs_without_changing_retained_boards(self):
        board = BitBoard()
        search = MCTS(None, ModelConfig())
        search._board_cache = LRUCache(maxsize=2)
        first_move = ((1, 0), (2, 0))
        original = search._get_child_board(board, *first_move)
        for move in (((1, 1), (2, 1)), ((1, 2), (2, 2))):
            search._get_child_board(board, *move)
        replacement = search._get_child_board(board, *first_move)
        self.assertIsNot(original, replacement)
        np.testing.assert_array_equal(original.state, replacement.state)
        self.assertEqual(len(search._board_cache), 2)

    def test_search_caches_are_isolated_and_cleared_on_success_and_failure(self):
        from tests.test_mcts_sequential import RecordingModel

        board = BitBoard()
        model = RecordingModel()
        search = MCTS(model, ModelConfig(n_simulations=2))
        search.get_move(board)
        retained_root = search.root_node
        self.assertIsNot(retained_root.board, board)
        self.assertIsNot(
            board._valid_moves_cache, retained_root.board._valid_moves_cache
        )
        for cache in (
            search._board_cache,
            search._all_moves_cache,
            *search._rule_caches,
        ):
            self.assertEqual(len(cache), 0)
        model.fail_on_call = len(model.inputs) + 2
        with self.assertRaisesRegex(RuntimeError, "inference failure"):
            search.get_move(board)
        self.assertIsNot(search.root_node, retained_root)
        for cache in (
            search._board_cache,
            search._all_moves_cache,
            *search._rule_caches,
        ):
            self.assertEqual(len(cache), 0)
        self.assertIsNot(BitBoard()._valid_moves_cache, board._valid_moves_cache)


if __name__ == "__main__":
    unittest.main()
