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


def mcts_worker(model, config, board, n_moves, result_queue):
    """Run MCTS for multiple moves in sequence"""
    moves_and_policies = []

    for _ in range(n_moves):
        if board.is_game_over():
            break

        mcts = MCTS(model, config)
        move = mcts.get_move(board, temperature=1.0)
        if not move:
            break

        policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)
        moves_and_policies.append((move, policy))

        # Update board for next move
        board.make_move(move[0], move[1])

        # Clean up MCTS instance
        del mcts

    result_queue.put(moves_and_policies)


def generate_games(_mcts: MCTS, model: ChessNet, config: ModelConfig) -> List[Tuple]:
    """Process-isolated self-play implementation"""
    logger = logging.getLogger(__name__)
    games = []

    # Create process context and queue
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    MOVES_PER_WORKER = 20  # Number of moves each worker handles

    for game_idx in range(config.n_games_per_iteration):
        game = ChessGame()
        game_history = []
        pbar = tqdm(total=200, desc=f"Game {game_idx+1}", leave=False)

        while not game.board.is_game_over() and len(game_history) < 200:
            # Start isolated process for batch of moves
            p = ctx.Process(
                target=mcts_worker,
                args=(
                    model,
                    config,
                    game.board,
                    MOVES_PER_WORKER,
                    result_queue,
                ),
            )
            p.start()

            # Wait with timeout
            p.join(timeout=MOVES_PER_WORKER * 2)  # Longer timeout for multiple moves

            # Handle timeouts
            if p.exitcode is None:
                p.terminate()
                p.join()
                logger.error("MCTS process timeout")
                break

            # Get results
            try:
                moves_and_policies = result_queue.get_nowait()
            except Exception:
                break

            # Process each move from the worker
            for move, policy in moves_and_policies:
                if game.board.is_game_over():
                    break

                state = mx.array(game.board.state, dtype=mx.float32)
                game_history.append((state, policy, None))
                game.make_move(move[0], move[1])
                pbar.update(1)

        pbar.close()

        # Process game results
        white_result = game.board.get_game_result(perspective_color=0)
        black_result = game.board.get_game_result(perspective_color=1)

        # Convert draws (0.0) to -0.5 to discourage drawing
        white_result = -0.5 if white_result == 0.0 else white_result
        black_result = -0.5 if black_result == 0.0 else black_result

        games.append((game_history, white_result))
        games.append((game_history, black_result))

        logger.info(f"Game {game_idx + 1} completed with {len(game_history)} moves")
        logger.info(f"White result: {white_result}, Black result: {black_result}")
        logger.info(f"Final board:\n{game.board}")

    return games


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
        # Process each move in the game
        for move_num, (state, policy, _) in enumerate(game_history):
            all_states.append(state)
            all_policies.append(policy)

            # If this is from white's perspective, even moves get perspective_result
            # If this is from black's perspective, odd moves get perspective_result
            is_perspective_move = move_num % 2 == 0
            value_target = (
                perspective_result if is_perspective_move else -perspective_result
            )
            all_values.append(value_target)

    print(f"Total positions collected: {len(all_states)}")

    if not all_states:
        print("Warning: No positions to create batches from!")
        return

    # Convert to arrays
    states = mx.array(all_states)
    policies = mx.array(all_policies)
    values = mx.array(all_values)

    # Create batches
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
    """Convert MCTS visit counts to policy distribution"""
    # Use numpy array for indexing, convert to MLX at the end
    policy = np.zeros(policy_output_dim, dtype=np.float32)

    for move, child in root_node.children.items():
        # Calculate move index directly since we don't have mcts instance
        from_pos, to_pos = move
        from_idx = from_pos[0] * 8 + from_pos[1]
        to_idx = to_pos[0] * 8 + to_pos[1]
        move_idx = from_idx * 64 + to_idx
        if move_idx < len(policy):
            policy[move_idx] = child.visit_count

    # Normalize using numpy first
    policy = policy / np.sum(policy) if np.sum(policy) > 0 else policy

    # Convert to MLX array at the end
    return mx.array(policy)


def encode_board(board: BitBoard) -> mx.array:
    """Convert BitBoard state to network input format"""
    return mx.array(board.state, dtype=mx.float32)
