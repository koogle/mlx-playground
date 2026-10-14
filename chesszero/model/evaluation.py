"""CPU evaluation games; all neural inference stays in the parent process."""

import numpy as np
from chess_engine.game import ChessGame
from model.mcts import MCTS


def play_evaluation_game(
    client,
    config,
    game_id,
    player_color,
    random_opponent,
    seed,
    connection,
    max_workers,
):
    try:
        rng = np.random.default_rng(seed)
        game = ChessGame()
        searches = {player_color: MCTS(client, config, exploration_noise=False)}
        if not random_opponent:
            searches[1 - player_color] = MCTS(
                client.for_model(1), config, exploration_noise=False
            )
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
            if move is None or not game.make_move(*move):
                raise RuntimeError(
                    "Evaluation selected no legal move in an ongoing game"
                )
            for search in searches.values():
                search.advance_root(move, game.board)
        result = game.board.get_game_result(player_color)
        if result is None:
            raise RuntimeError("Evaluation ended without a terminal result")
        connection.send(
            dict(
                game_id=game_id,
                result=result,
                perspective_color=player_color,
                final_board=str(game.board),
            )
        )
    finally:
        client.close()
