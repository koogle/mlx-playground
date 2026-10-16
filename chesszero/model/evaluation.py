"""CPU evaluation games; all neural inference stays in the parent process."""

import numpy as np
from chess_engine.game import ChessGame
from model.mcts import MCTS
from model.self_play import get_policy_distribution


def play_evaluation_game(
    client,
    config,
    game_id,
    player_color,
    random_opponent,
    seed,
    connection,
    max_workers,
    opening_plies=0,
    collect_history=False,
):
    try:
        if collect_history and random_opponent:
            raise ValueError("Random-opponent evaluations cannot supply training data")
        rng = np.random.default_rng(seed)
        game = ChessGame()
        history = []
        searches = {player_color: MCTS(client, config, exploration_noise=False)}
        if not random_opponent:
            searches[1 - player_color] = MCTS(
                client.for_model(1), config, exploration_noise=False
            )
        # The same seed produces the same opening for both candidate colors.
        # Keep search deterministic after the opening and disable root noise.
        for _ in range(opening_plies):
            if game.board.is_game_over():
                break
            legal = sorted(
                (source, target)
                for source, _ in game.board.get_all_pieces(
                    game.board.get_current_turn()
                )
                for target in game.board.get_valid_moves(source)
            )
            move = legal[int(rng.integers(len(legal)))]
            if not game.make_move(*move):
                raise RuntimeError("Evaluation opening contains an illegal move")
        while not game.board.is_game_over():
            turn = game.board.get_current_turn()
            if random_opponent and turn != player_color:
                legal = [
                    (source, target)
                    for source, _ in game.board.get_all_pieces(turn)
                    for target in sorted(game.board.get_valid_moves(source))
                ]
                move = legal[int(rng.integers(len(legal)))]
            else:
                move = searches[turn].get_move(game.board, temperature=0)
                if collect_history and move is not None:
                    history.append(
                        (
                            np.array(game.board.state, dtype=np.float32),
                            get_policy_distribution(
                                searches[turn].root_node, config.policy_output_dim
                            ),
                            None,
                        )
                    )
            if move is None or not game.make_move(*move):
                raise RuntimeError(
                    "Evaluation selected no legal move in an ongoing game"
                )
            for search in searches.values():
                search.advance_root(move, game.board)
        result = game.board.get_game_result(player_color)
        if result is None:
            raise RuntimeError("Evaluation ended without a terminal result")
        record = dict(
            game_id=game_id,
            result=result,
            perspective_color=player_color,
            final_board=str(game.board),
        )
        if collect_history:
            record["history"] = history
        connection.send(record)
    finally:
        client.close()


def play_promotion_game(*args):
    """Promotion pairs share an eight-ply opening before deterministic search."""
    return play_evaluation_game(*args, opening_plies=8, collect_history=True)
