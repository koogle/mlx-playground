import numpy as np
from chess_engine.game import ChessGame
from chess_engine.bitboard import BitBoard
from model.network import ChessNet
from config.model_config import ModelConfig
from model.mcts import MCTS, GameSearchCache
from model.move_encoding import encode_move
from typing import List, Tuple
import mlx.core as mx
from tqdm import tqdm
import logging
from multiprocessing.connection import Connection
from model.game_workers import run_game_workers
import traceback


def play_single_game(
    model: ChessNet,
    config: ModelConfig,
    game_id: int,
    result_connection: Connection,
    max_workers: int,
):
    """Worker process that plays a single complete game"""
    logger = logging.getLogger(__name__)
    game_cache = GameSearchCache(model)
    try:
        model.eval()
        game = ChessGame()
        game_history = []

        pbar = tqdm(
            desc=f"Game {game_id}",
            position=game_id % max_workers,
            leave=False,
            unit="moves",
            unit_scale=True,
        )

        while not game.board.is_game_over():
            mcts = MCTS(model, config, cache=game_cache)
            temperature = (
                config.temperature
                if len(game_history) < config.temp_decay_steps
                else config.temp_final
            )
            move = mcts.get_move(game.board, temperature=temperature)
            if move is None:
                raise RuntimeError("Search returned no move for an ongoing game")

            policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)
            state = mx.array(game.board.state, dtype=mx.float32)
            game_history.append((state, policy, None))
            if not game.make_move(move[0], move[1]):
                raise RuntimeError("Search returned an illegal move")

            pbar.update(1)

        pbar.close()

        white_result = game.board.get_game_result(perspective_color=0)
        if white_result is None:
            raise RuntimeError("Self-play ended without a terminal result")
        if white_result == 0.0:
            black_result = 0.0
        else:
            black_result = -white_result

        result_connection.send(
            {
                "game_id": game_id,
                "history": game_history,
                "white_result": white_result,
                "black_result": black_result,
                "n_moves": len(game_history),
                "final_board": str(game.board),
            }
        )

    except Exception as e:
        logger.error(f"Error in game {game_id}: {str(e)}")
        logger.error(traceback.format_exc())
        result_connection.send(None)
    finally:
        game_cache.clear()


def generate_games(
    model: ChessNet, config: ModelConfig, max_workers: int = 8
) -> List[Tuple]:
    logger = logging.getLogger(__name__)
    if config.n_games_per_iteration < 1:
        raise ValueError("Games per iteration must be positive")
    jobs = [(model, config, game_id) for game_id in range(config.n_games_per_iteration)]
    results, failures = run_game_workers(play_single_game, jobs, max_workers)
    logger.info("Self-play generation: %d completed, %d failed", len(results), failures)
    if not results:
        raise RuntimeError("All self-play games failed; no training update can be made")
    for result in results:
        logger.info(
            "Game %d: %d moves, White result %s",
            result["game_id"],
            result["n_moves"],
            result["white_result"],
        )
    return [(result["history"], result["white_result"]) for result in results]


def create_batches(games, batch_size: int):
    """Create training batches from games
    Args:
        games: List of tuples (game_history, white_result), one per game.
            Value targets are converted to the side-to-move perspective of each state.
        batch_size: Size of training batches
    """
    print(
        f"Creating batches from {len(games)} game instances with batch_size {batch_size}"
    )

    all_states = []
    all_policies = []
    all_values = []

    for game_history, white_result in games:
        for state, policy, _ in game_history:
            all_states.append(state)
            all_policies.append(policy)

            # Plane 12 is 1 for White to move and 0 for Black to move.
            if state[12, 0, 0] == 1:
                value_target = white_result
            else:
                value_target = -white_result
            all_values.append(value_target)

    print(f"Total positions collected: {len(all_states)}")

    if not all_states:
        print("Warning: No positions to create batches from!")
        return

    states = mx.array(all_states)
    policies = mx.array(all_policies)
    values = mx.array(all_values)

    n_samples = len(states)
    indices = mx.arange(n_samples)
    print(f"Creating batches of size {batch_size} from {n_samples} samples")

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield (states[batch_indices], policies[batch_indices], values[batch_indices])


def get_policy_distribution(root_node, policy_output_dim: int):
    """Convert MCTS visit counts to policy distribution we are doing this in numpy"""
    policy = np.zeros(policy_output_dim, dtype=np.float32)

    for move, child in root_node.children.items():
        move_idx = encode_move(*move)
        policy[move_idx] = child.visit_count

    policy = policy / np.sum(policy) if np.sum(policy) > 0 else policy

    return mx.array(policy)


def encode_board(board: BitBoard) -> mx.array:
    return mx.array(board.state, dtype=mx.float32)
