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


def mcts_worker(model, config, board, result_queue):
    """Run MCTS in isolated process with full memory containment"""
    mcts = MCTS(model, config)
    move = mcts.get_move(board, temperature=1.0)
    policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)

    # Return simple types
    result_queue.put((move, policy))


def generate_games(_mcts: MCTS, model: ChessNet, config: ModelConfig) -> List[Tuple]:
    """Process-isolated self-play implementation"""
    logger = logging.getLogger(__name__)
    games = []

    # Serialize model parameters
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    for game_idx in range(config.n_games_per_iteration):
        game = ChessGame()
        game_history = []
        pbar = tqdm(total=200, desc=f"Game {game_idx+1}", leave=False)

        for _ in range(200):
            if game.board.is_game_over():
                break

            # Start isolated process
            p = ctx.Process(
                target=mcts_worker,
                args=(
                    model,
                    config,
                    game.board,
                    result_queue,
                ),
            )
            p.start()

            # Wait with timeout
            p.join(timeout=10)

            # Handle timeouts
            if p.exitcode is None:
                p.terminate()
                p.join()
                logger.error("MCTS process timeout")
                break

            # Get results
            try:
                move, policy = result_queue.get_nowait()
            except Exception:
                break

            if not move:
                break

            # Record and make move
            state = mx.array(game.board.state, dtype=mx.float32)
            game_history.append((state, policy, None))
            game.make_move(move[0], move[1])
            pbar.update(1)

        pbar.close()

        # Get game result from both perspectives
        white_result = game.board.get_game_result(perspective_color=0)
        black_result = game.board.get_game_result(perspective_color=1)

        # Add game twice - once from each perspective
        games.append((game_history, white_result))  # White's perspective
        games.append((game_history, black_result))  # Black's perspective

        logger.info(f"Game {game_idx + 1} completed with {len(game_history)} moves")
        logger.info(f"White result: {white_result}, Black result: {black_result}")
        logger.info(f"Final board:\n{game.board}")

    if len(games) == 0:
        logger.warning("No valid games were generated!")
    logger.info(
        f"Generated {len(games)} training instances from {config.n_games_per_iteration} games"
    )
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
