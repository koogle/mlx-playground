from functools import lru_cache
import numpy as np
from chess_engine.game import ChessGame
from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from utils.random_player import RandomPlayer
from model.mcts import MCTS
from typing import List, Tuple
import mlx.core as mx
from tqdm import tqdm
import logging


def play_self_play_game(mcts: MCTS, config) -> Tuple[List, List, List]:
    """Play a single game of self-play and return the training data"""
    game = ChessGame()
    states, policies, values = [], [], []

    print("\nStarting new self-play game")
    print("---------------------------")
    print("\nInitial position:")
    print(game.board)

    move_count = 0
    for move_count in range(201):  # Removed tqdm to see board positions clearly
        if game.is_over():
            break

        # Print current move number
        print(f"\nMove {move_count + 1}")

        # Get move with temperature
        temperature = 0.5 if move_count > 30 else 1.0
        move = mcts.get_move(game.board, temperature=temperature)
        if move is None:
            print("No valid moves found!")
            break

        # Store state and create policy from visits
        encoded_state = encode_board(game.board)
        policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)

        states.append(encoded_state)
        policies.append(policy)

        # Print the move being made
        from_square = f"{chr(move[0][1] + 97)}{move[0][0] + 1}"
        to_square = f"{chr(move[1][1] + 97)}{move[1][0] + 1}"
        print(f"AI plays: {from_square}{to_square}")

        # Make the move and print resulting position
        game.make_move_coords(move[0], move[1], f"{move[0]}{move[1]}")
        print("\nPosition after move:")
        print(game.board)

        move_count += 1

    # Game is over - print final position and outcome
    result = game.get_result()

    print("\n=== Game Complete ===")
    print("\nFinal position:")
    print(game.board)
    print("\nGame ended because:", end=" ")

    if game.board.is_checkmate(game.board.get_current_turn()):
        print("Checkmate!")
    elif game.board.is_stalemate(game.board.get_current_turn()):
        print("Stalemate!")
    elif game.board.is_draw():
        print("Draw by insufficient material or repetition!")
    elif move_count >= 200:
        print("Maximum moves reached!")
    else:
        print("Unknown reason")

    print(f"Result: {result}")
    print(f"Total moves: {move_count}")
    print("===================\n")

    # Fill in the values array based on game result
    values = [result if i % 2 == 0 else -result for i in range(len(states))]

    return states, policies, values


def generate_games(mcts: MCTS, config: ModelConfig) -> List[Tuple]:
    """Generate self-play games
    Returns:
        List of tuples (game_history, game_result) where each game generates two training instances:
        one from white's perspective and one from black's perspective
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\nGenerating {config.n_games_per_iteration} self-play games")
    games = []

    for game_idx in range(config.n_games_per_iteration):
        mcts.clear_all_caches()
        game = ChessGame()
        game_history = []

        # Create progress bar for each game with max 200 moves
        pbar = tqdm(total=200, desc=f"Game {game_idx+1}", leave=False)
        move_count = 0

        while not game.board.is_game_over() and move_count < 200:
            # Convert state to MLX array immediately
            state = mx.array(game.board.state, dtype=mx.float32)
            move = mcts.get_move(game.board, temperature=1.0)
            if not move:
                break

            # Use existing get_policy_distribution function
            policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)
            game_history.append((state, policy, None))
            game.make_move(move[0], move[1])

            move_count += 1
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


def generate_random_opponent_games(mcts: MCTS, config) -> List[Tuple]:
    """Generate games playing against a random opponent"""
    games_data = []
    random_player = RandomPlayer()

    for game_idx in range(config.n_games_per_iteration):
        mcts.clear_all_caches()

        if game_idx % 10 == 0:  # Progress update
            print(f"Generating game {game_idx + 1}/{config.n_games_per_iteration}")

        # Play as both white and black alternately
        color = 0 if game_idx % 2 == 0 else 1
        game = ChessGame()
        states, policies, values = [], [], []

        print(f"\nStarting new game as {'White' if color == 0 else 'Black'}")
        print("---------------------------")

        move_count = 0
        for move_count in tqdm(range(201), desc="Playing game"):  # 200 move limit
            if game.is_over():
                break

            # if move_count % 20 == 0:
            # print(f"\nMove {move_count}")
            # print(game.board)

            current_state = encode_board(game.board)

            if game.get_current_turn() == color:
                # MCTS player's turn
                move = mcts.get_move(game.board)
                policy = np.zeros(config.policy_output_dim)
                move_idx = mcts.encode_move(move[0], move[1])
                policy[move_idx] = 1.0

                states.append(current_state)
                policies.append(policy)
            else:
                # Random player's turn
                move = random_player.select_move(game.board)

            if move is None:
                print("No move found")
                print(game.board)
                print(game.get_current_turn())
                print(game.get_all_valid_moves())
                print("current color", color)

            # Make move
            game.make_move_coords(move[0], move[1], f"{move[0]}{move[1]}")
            move_count += 1

        # Game is over - get result
        result = game.get_result()
        print(f"\nGame over. Result: {result}")

        # Adjust result based on our player's color
        if color == 1:
            result = -result

        # Fill in the values array based on game result
        values = [result for _ in range(len(states))]

        if states:  # Only add if we have data
            games_data.append((states, policies, values))

    return games_data


@lru_cache(maxsize=1000)
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


@lru_cache(maxsize=1000)
def encode_board(board: BitBoard) -> mx.array:
    """Convert BitBoard state to network input format"""
    # Convert uint8 to float32 when creating MLX array
    return mx.array(board.state, dtype=mx.float32)
