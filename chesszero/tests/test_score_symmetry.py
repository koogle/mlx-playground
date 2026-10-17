"""Color/rank mirror regressions for scoring, independent of learned strength.

python-chess supplies the mirrored positions and expected outcomes. A mirror
swaps colors and reflects ranks: White-relative results reverse, while values
relative to the corresponding player (including side to move) stay unchanged.
An arbitrary neural network is not required to be mirror-equivariant.
"""

from types import SimpleNamespace
from unittest.mock import patch

import chess
import numpy as np
import pytest

from config.model_config import ModelConfig
from model.evaluation import play_evaluation_game
from model.mcts import MCTS, Node
from model.self_play import create_batches, play_single_game
from test_board import coords, from_fen, legal_moves


def mirror_move(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def board_pair(reference):
    original, _ = from_fen(reference.fen(en_passant="fen"))
    mirrored_reference = reference.mirror()
    mirrored, _ = from_fen(mirrored_reference.fen(en_passant="fen"))
    # Elapsed plies are metadata, not a function of which color moves first.
    mirrored.state[17] = original.state[17]
    return (original, reference), (mirrored, mirrored_reference)


def assert_result(board, reference):
    outcome = reference.outcome(claim_draw=False)
    # Native self-play automatically claims draws already reached on the board.
    if outcome is None and (reference.is_repetition(3) or reference.is_fifty_moves()):
        expected = 0.0
    elif outcome is None:
        expected = None
    else:
        expected = 0.0 if outcome.winner is None else (1.0 if outcome.winner else -1.0)
    assert board.is_game_over() == (expected is not None)
    assert board.get_game_result(0) == expected
    assert board.get_game_result(1) == (None if expected is None else -expected)
    stm_expected = (
        expected if reference.turn else (None if expected is None else -expected)
    )
    assert board.get_game_result() == stm_expected


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",  # Checkmate.
        "7k/6Q1/5K2/8/8/8/8/8 b - - 100 1",  # Mate beats 50-move draw.
        "7k/6Q1/5K2/8/8/8/8/8 b - - 150 1",  # Mate beats 75-move draw.
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",  # Stalemate.
        "7k/8/5K2/8/8/8/8/8 b - - 0 1",  # Bare kings.
        "7k/8/5K2/8/8/8/3B4/8 b - - 0 1",
        "7k/8/5K2/8/8/8/3N4/8 b - - 0 1",
        "7k/8/5K2/8/8/8/3R4/8 b - - 100 1",
        "7k/8/5K2/8/8/8/3R4/8 b - - 150 1",
    ],
)
def test_terminal_and_ongoing_scores_mirror(fen):
    pairs = board_pair(chess.Board(fen))
    for board, reference in pairs:
        assert reference.is_valid()
        assert_result(board, reference)
    original, mirrored = (pair[0] for pair in pairs)
    assert original.get_game_result() == mirrored.get_game_result()
    for color in (0, 1):
        assert original.get_game_result(color) == mirrored.get_game_result(1 - color)


@pytest.mark.parametrize(
    "uci_moves",
    [
        "f2f3 e7e5 g2g4 d8h4",  # Black wins.
        "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7",  # White wins.
        "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8",  # Threefold.
    ],
)
@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_replayed_games_and_training_targets_mirror(uci_moves, batch_size):
    pairs = board_pair(chess.Board())
    histories = [[], []]
    for uci in uci_moves.split():
        move = chess.Move.from_uci(uci)
        for index, ((board, reference), played) in enumerate(
            zip(pairs, (move, mirror_move(move)))
        ):
            assert_result(board, reference)
            assert played in reference.legal_moves
            policy = np.zeros(4672, dtype=np.float32)
            from model.move_encoding import encode_move

            policy[encode_move(*coords(played))] = 1
            histories[index].append((board.state.copy(), policy, None))
            assert board.make_move(*coords(played))
            reference.push(played)
            assert_result(board, reference)
    values = []
    for (board, reference), history in zip(pairs, histories):
        assert board.is_game_over()
        batches = list(
            create_batches([(history, board.get_game_result(0))], batch_size)
        )
        targets = np.concatenate([np.array(batch[2]) for batch in batches])
        winner = reference.outcome(claim_draw=True).winner
        expected = [
            0.0
            if winner is None
            else (1.0 if bool(state[12, 0, 0]) == winner else -1.0)
            for state, _, _ in history
        ]
        np.testing.assert_array_equal(targets, expected)
        assert sum(len(batch[0]) for batch in batches) == len(history)
        values.append(targets)
    np.testing.assert_array_equal(values[0], values[1])
    assert pairs[0][0].get_game_result(0) == -pairs[1][0].get_game_result(0)


@pytest.mark.parametrize("length", range(1, 7))
@pytest.mark.parametrize("value_player", [0, 1])
@pytest.mark.parametrize("value", [-1.0, 0.0, 0.35, 1.0])
def test_backup_mirrors_every_depth_and_explicit_perspective(
    length, value_player, value
):
    pairs = board_pair(chess.Board())
    paths = [[], []]
    for ply in range(length):
        for path, (board, _) in zip(paths, pairs):
            position = board.copy()
            position.state[12].fill((ply % 2) == board.get_current_turn())
            path.append(Node(position, exploration_noise=False))
    search = MCTS(None, ModelConfig())
    search.backup(paths[0], value, value_player=value_player)
    search.backup(paths[1], value, value_player=1 - value_player)
    for original, mirrored in zip(*paths):
        expected = (
            value if original.board.get_current_turn() == value_player else -value
        )
        assert original.value() == pytest.approx(expected)
        assert mirrored.value() == pytest.approx(expected)
        assert original.visit_count == mirrored.visit_count == 1


class UniformClient:
    """Symmetric side-to-move network stub; terminal scores still use real rules."""

    def __init__(self):
        self.closed = False
        self.models = []

    def __call__(self, states):
        return np.full((len(states), 4672), 1 / 4672), np.full((len(states), 1), 0.25)

    def for_model(self, model):
        self.models.append(model)
        return self

    def close(self):
        self.closed = True


class Connection:
    def __init__(self):
        self.results = []

    def send(self, result):
        self.results.append(result)


def test_real_search_selects_mirrored_mate_for_either_color():
    reference = chess.Board()
    for uci in "f2f3 e7e5 g2g4".split():
        reference.push_uci(uci)
    expected = chess.Move.from_uci("d8h4")
    for (board, oracle), move in zip(
        board_pair(reference), (expected, mirror_move(expected))
    ):
        search = MCTS(
            UniformClient(), ModelConfig(n_simulations=96), exploration_noise=False
        )
        with patch.object(search, "should_stop", return_value=False):
            selected = search.get_move(board, temperature=0)
        assert selected == coords(move)
        child = search.root_node.children[selected]
        assert child.value() == -1.0  # Mated opponent's perspective.
        assert child.board.get_game_result(board.get_current_turn()) == 1.0
        oracle.push(move)
        assert oracle.is_checkmate()


# A real two-ply game: Black has exactly one legal move, then White mates.
FORCED_FEN = "7k/8/5K2/6Q1/8/8/8/8 b - - 0 1"


def scripted_search(client, config, **kwargs):
    """Isolate worker result conversion; real MCTS is tested separately above."""

    class Search:
        def get_move(self, board, temperature):
            turn = board.get_current_turn()
            # Mate if possible; otherwise the fixture has exactly one move.
            moves = legal_moves(board)
            mates = []
            for move in moves:
                child = board.copy()
                assert child.make_move(*move)
                if child.is_checkmate(1 - turn):
                    mates.append(move)
            assert len(mates or moves) == 1
            selected = next(iter(mates or moves))
            self.root_node = SimpleNamespace(
                children={selected: SimpleNamespace(visit_count=1)}
            )
            return selected

        def advance_root(self, *args):
            pass

    return Search()


def test_self_play_worker_results_and_batch_targets_mirror():
    results = []
    for board, oracle in board_pair(chess.Board(FORCED_FEN)):
        from chess_engine.game import ChessGame

        game = ChessGame()
        game.board = board
        client, connection = UniformClient(), Connection()
        with (
            patch("model.self_play.ChessGame", return_value=game),
            patch("model.self_play.MCTS", side_effect=scripted_search),
        ):
            play_single_game(client, ModelConfig(), 0, connection, 1)
        assert client.closed
        assert len(connection.results) == 1
        result = connection.results[0]
        assert result is not None
        assert result["n_moves"] == 2
        assert result["white_result"] == (1.0 if oracle.turn == chess.BLACK else -1.0)
        assert result["black_result"] == -result["white_result"]
        _, _, targets = next(
            create_batches([(result["history"], result["white_result"])], 8)
        )
        np.testing.assert_array_equal(targets, [-1, 1])
        results.append(result)
    assert results[0]["white_result"] == -results[1]["white_result"]


@pytest.mark.parametrize("original_player", [0, 1])
@pytest.mark.parametrize("collect_history", [False, True])
def test_evaluation_worker_scores_tested_player_after_color_swap(
    original_player, collect_history
):
    scores = []
    for index, (board, _) in enumerate(board_pair(chess.Board(FORCED_FEN))):
        from chess_engine.game import ChessGame

        game = ChessGame()
        game.board = board
        player = original_player if index == 0 else 1 - original_player
        client, connection = UniformClient(), Connection()
        with (
            patch("model.evaluation.ChessGame", return_value=game),
            patch("model.evaluation.MCTS", side_effect=scripted_search),
        ):
            play_evaluation_game(
                client,
                ModelConfig(),
                0,
                player,
                False,
                42,
                connection,
                1,
                collect_history=collect_history,
            )
        assert client.closed
        assert client.models == [1]
        (result,) = connection.results
        assert result["perspective_color"] == player
        if collect_history:
            assert len(result["history"]) == 2
            white_result = result["result"] * (1 if player == 0 else -1)
            states, policies, targets = next(
                create_batches([(result["history"], white_result)], 8)
            )
            np.testing.assert_array_equal(targets, [-1, 1])
            np.testing.assert_allclose(np.asarray(policies).sum(axis=1), 1)
            # Both players' positions were copied before each move.
            assert states[0, 12, 0, 0].item() != states[1, 12, 0, 0].item()
        else:
            assert "history" not in result
        scores.append(result["result"])
    assert scores == ([1.0, 1.0] if original_player == 0 else [-1.0, -1.0])


def test_promotion_history_excludes_random_opening_positions():
    from chess_engine.game import ChessGame

    game = ChessGame()
    game.board, _ = from_fen(FORCED_FEN)
    connection = Connection()
    with (
        patch("model.evaluation.ChessGame", return_value=game),
        patch("model.evaluation.MCTS", side_effect=scripted_search),
    ):
        # Black's only legal move is the opening; White then delivers mate.
        play_evaluation_game(
            UniformClient(),
            ModelConfig(),
            0,
            0,
            False,
            42,
            connection,
            1,
            opening_plies=1,
            collect_history=True,
        )
    (result,) = connection.results
    assert result["result"] == 1
    assert len(result["history"]) == 1
    assert result["history"][0][0][12, 0, 0] == 1


@pytest.mark.parametrize(
    "fen",
    [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 12",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 12",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 12",
        "4k3/8/8/8/8/8/8/R3K3 b - - 99 12",
    ],
)
def test_position_mirrors_preserve_legal_moves_and_state_semantics(fen):
    pairs = board_pair(chess.Board(fen))
    (original, oracle), (mirrored, mirror_oracle) = pairs
    for board, reference in pairs:
        assert reference.is_valid()
        assert legal_moves(board) == {coords(move) for move in reference.legal_moves}
        assert_result(board, reference)
    assert legal_moves(mirrored) == {
        coords(mirror_move(move)) for move in oracle.legal_moves
    }
    # Build expectations independently from python-chess piece maps above.
    channels = list(range(6, 12)) + list(range(6)) + [12, 15, 16, 13, 14, 17, 18]
    expected = original.state[channels, ::-1, :].copy()
    expected[12] = 1 - expected[12]
    np.testing.assert_array_equal(mirrored.state, expected)
    assert mirrored.get_current_turn() == 1 - original.get_current_turn()
    if oracle.ep_square is not None:
        assert mirror_oracle.ep_square == chess.square_mirror(oracle.ep_square)
        row, col = original.en_passant_target
        assert mirrored.en_passant_target == (7 - row, col)


@pytest.mark.parametrize("root_color", [0, 1])
def test_selection_prefers_opponents_loss_for_both_colors(root_color):
    board = board_pair(chess.Board())[root_color][0]
    root = Node(board, exploration_noise=False)
    children = []
    for opponent_value in (-0.8, 0.8):
        child_board = board.copy()
        child_board.state[12] = 1 - board.state[12]
        child = Node(child_board, parent=root, prior=0.5)
        MCTS(None, ModelConfig()).backup(
            [root, child], opponent_value, value_player=1 - root_color
        )
        children.append(child)
    root.children = dict(zip(("win", "loss"), children))
    move, child = root.select_child(c_puct=0)
    assert move == "win"
    assert child.value() == -0.8
