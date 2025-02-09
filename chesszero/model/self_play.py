import numpy as np
from chess_engine.game import ChessGame
from chess_engine.bitboard import BitBoard
from model.network import ChessNet
from config.model_config import ModelConfig
from model.mcts import MCTS
from typing import List, Tuple
import mlx.core as mx
from tqdm import tqdm
import logging
import multiprocessing as mp
from queue import Empty
import traceback


def play_single_game(
    model: ChessNet,
    config: ModelConfig,
    game_id: int,
    result_queue: mp.Queue,
    max_workers: int,
):
    """Worker process that plays a single complete game"""
    logger = logging.getLogger(__name__)
    try:
        game = ChessGame()
        game_history = []

        pbar = tqdm(
            desc=f"Game {game_id}",
            position=game_id,
            leave=False,
            unit="moves",
            unit_scale=True,
        )

        while not game.board.is_game_over():
            mcts = MCTS(model, config)
            move = mcts.get_move(game.board, temperature=1.0)
            if not move:
                break

            policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)
            state = mx.array(game.board.state, dtype=mx.float32)
            game_history.append((state, policy, None))
            game.make_move(move[0], move[1])

            pbar.update(1)

        pbar.close()

        white_result = game.board.get_game_result(perspective_color=0)
        if white_result is None:
            white_result = 0.0
        if white_result == 0.0:
            black_result = 0.0
        else:
            black_result = -white_result

        result_queue.put(
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
        result_queue.put(None)


def generate_games(
    model: ChessNet, config: ModelConfig, max_workers: int = 5
) -> List[Tuple]:
    logger = logging.getLogger(__name__)
    games_data = []
    total_games_needed = config.n_games_per_iteration
    games_completed = 0
    game_id = 0

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    active_workers = []

    try:
        while games_completed < total_games_needed:
            while len(active_workers) < max_workers and game_id < total_games_needed:
                p = ctx.Process(
                    target=play_single_game,
                    args=(model, config, game_id, result_queue, max_workers),
                )
                p.start()
                active_workers.append((p, game_id))
                game_id += 1

            try:
                result = result_queue.get(timeout=1)
                if result is not None:
                    games_completed += 1

                    # Store both white and black perspectives
                    games_data.append((result["history"], result["white_result"]))
                    games_data.append((result["history"], result["black_result"]))

                    logger.info(
                        f"Game {result['game_id']} completed with {result['n_moves']} moves"
                    )
                    logger.info(
                        f"White result: {result['white_result']}, Black result: {result['black_result']}"
                    )
                    logger.info(f"Final board:\n{result['final_board']}")

            except Empty:
                logger.debug("No results available in queue yet")

            active_workers = [(p, gid) for p, gid in active_workers if p.is_alive()]

    finally:
        for p, _ in active_workers:
            if p.is_alive():
                p.terminate()
                p.join()

    return games_data


def create_batches(games, batch_size: int):
    """Create training batches from games
    Args:
        games: List of tuples (game_history, game_result) where each game appears twice:
            once with white's perspective result and once with black's perspective result
        batch_size: Size of training batches
    """
    print(
        f"Creating batches from {len(games)} game instances with batch_size {batch_size}"
    )

    all_states = []
    all_policies = []
    all_values = []

    for game_history, perspective_result in games:
        for move_num, (state, policy, _) in enumerate(game_history):
            all_states.append(state)
            all_policies.append(policy)

            is_perspective_move = move_num % 2 == 0
            value_target = (
                perspective_result if is_perspective_move else -perspective_result
            )
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
        if len(batch_indices) < batch_size:
            pad_size = batch_size - len(batch_indices)
            batch_indices = mx.concatenate([batch_indices, batch_indices[:pad_size]])

        yield (states[batch_indices], policies[batch_indices], values[batch_indices])


def get_policy_distribution(root_node, policy_output_dim: int):
    """Convert MCTS visit counts to policy distribution we are doing this in numpy"""
    policy = np.zeros(policy_output_dim, dtype=np.float32)

    for move, child in root_node.children.items():
        from_pos, to_pos = move
        from_idx = from_pos[0] * 8 + from_pos[1]
        to_idx = to_pos[0] * 8 + to_pos[1]
        move_idx = from_idx * 64 + to_idx
        if move_idx < len(policy):
            policy[move_idx] = child.visit_count

    policy = policy / np.sum(policy) if np.sum(policy) > 0 else policy

    return mx.array(policy)


def encode_board(board: BitBoard) -> mx.array:
    return mx.array(board.state, dtype=mx.float32)
