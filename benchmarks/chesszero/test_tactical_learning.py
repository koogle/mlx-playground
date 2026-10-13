import unittest
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from config.model_config import ModelConfig
from model.network import ChessNet
from tactical_learning import fixture, legal_moves, probe


class TacticalDiagnosticTests(unittest.TestCase):
    def test_forced_line_and_both_value_perspectives(self):
        for mirror in (False, True):
            root, child, forced, mate = fixture(mirror)
            self.assertEqual(legal_moves(root), [forced])
            self.assertEqual(len(legal_moves(child)), 25)
            self.assertTrue(child.make_move(*mate))
            self.assertTrue(child.is_game_over())
            self.assertEqual(child.get_game_result(perspective_color=int(mirror)), 1)
            self.assertEqual(
                child.get_game_result(perspective_color=1 - int(mirror)), -1
            )

    def test_probe_preserves_weights_and_batchnorm_state(self):
        model = ChessNet(ModelConfig(n_filters=2, n_residual_blocks=1))
        mx.eval(model.parameters())
        before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
        result = probe(model, [fixture(False), fixture(True)])
        self.assertEqual(set(result), {"eval", "train"})
        for k, v in tree_flatten(model.parameters()):
            np.testing.assert_array_equal(before[k], np.array(v), err_msg=k)


if __name__ == "__main__":
    unittest.main()
