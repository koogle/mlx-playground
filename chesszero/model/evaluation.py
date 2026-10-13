"""CPU chess evaluation workers use only central inference RPC clients."""

from multiprocessing.connection import Connection
import logging
import traceback
import mlx.core as mx
from tqdm import tqdm
from chess_engine.game import ChessGame
from config.model_config import ModelConfig
from model.inference import InferenceClient
from model.mcts import MCTS


def play_evaluation_game_worker(
    model: InferenceClient,
    best_model: InferenceClient,
    config: ModelConfig,
    game_id: int,
    player_color: int,
    result_connection: Connection,
    max_workers: int,
):
    """Worker process that plays a single evaluation game and reports its result."""
    try:
        mx.set_default_device(mx.cpu)
        game = ChessGame()

        pbar = tqdm(
            desc=f"Evaluation game {game_id}",
            position=game_id % max_workers,
            leave=False,
            unit="moves",
            unit_scale=True,
        )

        searches = {
            player_color: MCTS(model, config, exploration_noise=True),
            1 - player_color: MCTS(best_model, config, exploration_noise=True),
        }
        while not game.board.is_game_over():
            mcts = searches[game.board.get_current_turn()]

            move = mcts.get_move(game.board, temperature=0.0)
            if move is None:
                raise RuntimeError("Search returned no move for an ongoing evaluation")

            if not game.make_move(move[0], move[1]):
                raise RuntimeError("Search returned an illegal evaluation move")

            for search in searches.values():
                search.advance_root(move, game.board)
            pbar.update(1)

        pbar.close()

        result = game.board.get_game_result(player_color)
        if result is None:
            raise RuntimeError("Evaluation ended without a terminal result")
        result_connection.send(
            {
                "game_id": game_id,
                "result": result,
                "perspective_color": player_color,
                "final_board": str(game.board),
            }
        )

    except Exception as e:
        logging.error(f"Error in evaluation game {game_id}: {str(e)}")
        logging.error(traceback.format_exc())
        result_connection.send(None)

    finally:
        model.close()
        best_model.close()
