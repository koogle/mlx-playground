"""Read-only comparison: loss derivatives and historical/current code semantics.
No optimizer updates, game search, training run, or GPU allocation.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from config.model_config import ModelConfig
from chess_engine.bitboard import BitBoard
from model.network import ChessNet
from model.mcts import MCTS, Node
from model.self_play import create_batches
from model.move_encoding import encode_move

root = Path(sys.argv[1])
base = root / "baseline/chesszero"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old_self = load("historical_self_play", base / "model/self_play.py")
old_mcts = load("historical_mcts", base / "model/mcts.py")
old_net = load("historical_network", base / "model/network.py")
old_engine = load("historical_bitboard", base / "chess_engine/bitboard.py")
report = {}
# Exact gradients on the persisted first-generation target distributions.
a = np.load(root / "training-samples-0001.npz")
p = mx.array(a["policies"][:64])
logits = mx.zeros(p.shape)
old = mx.grad(
    lambda x: nn.losses.cross_entropy(mx.softmax(x, axis=-1), p, reduction="mean")
)(logits)
new = mx.grad(lambda x: nn.losses.cross_entropy(x, p, reduction="mean"))(logits)
mx.eval(old, new)
report["policy_loss_gradient"] = {
    "initial_logits": "uniform",
    "targets": "64 persisted generation1 samples",
    "old_norm": float(mx.sqrt(mx.sum(old * old)).item()),
    "new_norm": float(mx.sqrt(mx.sum(new * new)).item()),
    "new_over_old": float(
        (mx.sqrt(mx.sum(new * new)) / mx.sqrt(mx.sum(old * old))).item()
    ),
    "expected_uniform_ratio": p.shape[-1],
}
# Run the actual historical/current batchers on the same four-position game.
board = BitBoard()
history = []
for u in "f2f3 e7e5 g2g4 d8h4".split():
    source = (int(u[1]) - 1, ord(u[0]) - 97)
    target = (int(u[3]) - 1, ord(u[2]) - 97)
    history.append((board.state.copy(), np.ones(4672, dtype=np.float32) / 4672, None))
    assert board.make_move(source, target)
old_history = [(mx.array(s), mx.array(p), None) for s, p, _ in history]
y_old = np.array(
    next(old_self.create_batches([(old_history, -1.0), (old_history, 1.0)], 64))[2]
)
y_new = np.array(next(create_batches([(history, -1.0)], 64))[2])
report["actual_batcher_labels"] = {
    "black_win_old_first_copy": y_old[:4].tolist(),
    "black_win_old_second_copy": y_old[4:8].tolist(),
    "new_once_per_game": y_new.tolist(),
    "old_mean_per_identical_state": ((y_old[:4] + y_old[4:8]) / 2).tolist(),
}
# Direct backup: Black leaf wins +0.75, so White parent must receive -0.75.
boards = [
    SimpleNamespace(get_current_turn=lambda: 0),
    SimpleNamespace(get_current_turn=lambda: 1),
]
old_path = [old_mcts.Node(b) for b in boards]
new_path = [Node(b) for b in boards]
old_mcts.MCTS.backup(None, old_path, 0.75)
MCTS.backup(None, new_path, 0.75, value_player=1)
report["backup_one_edge"] = {
    "leaf_player": "Black",
    "leaf_value": 0.75,
    "expected_root_child": [-0.75, 0.75],
    "old_root_child": [n.value() for n in old_path],
    "new_root_child": [n.value() for n in new_path],
}
# State encoding and repetition on a shared legal eight-ply knight sequence.
old_board, new_board = old_engine.BitBoard(), BitBoard()
for u in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split():
    move = ((int(u[1]) - 1, ord(u[0]) - 97), (int(u[3]) - 1, ord(u[2]) - 97))
    assert old_board.make_move(*move) and new_board.make_move(*move)
report["same_knight_repetition"] = {
    "old_draw": old_board.is_draw(),
    "new_draw": new_board.is_draw(),
    "old_plane17": float(old_board.state[17, 0, 0]),
    "new_plane17": float(new_board.state[17, 0, 0]),
    "old_plane18": float(old_board.state[18, 0, 0]),
    "new_plane18": float(new_board.state[18, 0, 0]),
    "old_counted_plies": old_board.get_move_count(),
    "new_counted_plies": new_board.get_move_count(),
}
old_network = old_net.ChessNet(
    ModelConfig(n_residual_blocks=0, n_filters=2, policy_output_dim=4)
)
new_network = ChessNet(
    ModelConfig(n_residual_blocks=0, n_filters=2, policy_output_dim=4)
)
report["batchnorm_initial_variance"] = {
    "old": old_network.bn_input.running_var.tolist(),
    "new": new_network.bn_input.running_var.tolist(),
}
# Different promotion metrics can accept/reject the same measured outcomes.
w, d, losses = 6, 30, 4
report["promotion_same_wdl"] = {
    "wins": w,
    "draws": d,
    "losses": losses,
    "old_decisive_win_fraction": w / (w + losses),
    "pr11_all_game_win_fraction": w / (w + d + losses),
    "current_draw_half_score": (w + 0.5 * d) / (w + d + losses),
    "old_accepts": w / (w + losses) > 0.55,
    "current_accepts": (w + 0.5 * d) / (w + d + losses) > 0.55,
}
# Historical class-level trees are not scoped to a model in evaluation.

moves = [((1, 4), (3, 4)), ((1, 3), (3, 3))]
policies = []
for m in moves:
    q = np.ones(4672, dtype=np.float32)
    q[encode_move(*m)] = 10000
    policies.append(q)
old_roots = []
new_roots = []
for policy in policies:
    board = BitBoard()
    search = old_mcts.MCTS(object(), ModelConfig())
    node = old_mcts.Node(board)
    search._expand_node(node, policy, 0.0)
    old_roots.append(node)
    search_new = MCTS(object(), ModelConfig(), exploration_noise=False)
    node_new = Node(
        board, exploration_noise=False, board_factory=search_new._get_child_board
    )
    search_new._expand_node(node_new, policy, 0.0)
    new_roots.append(node_new)
report["cross_model_tree_reuse"] = {
    "old_same_children_object": old_roots[0].children is old_roots[1].children,
    "old_top_priors": [
        str(max(n.children, key=lambda m: n.children[m].prior)) for n in old_roots
    ],
    "new_same_children_object": new_roots[0].children is new_roots[1].children,
    "new_top_priors": [
        str(max(n.children, key=lambda m: n.children[m].prior)) for n in new_roots
    ],
}
(root / "mechanics-results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
