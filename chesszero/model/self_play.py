import numpy as np
from chess_engine.game import ChessGame
from model.mcts import MCTS
from utils.board_utils import encode_board
from typing import List, Tuple
import mlx.core as mx


def play_self_play_game(mcts: MCTS, config) -> Tuple[List, List, List]:
    """Play a single game of self-play and return the training data"""
    game = ChessGame()
    states, policies, values = [], [], []

    print("\nStarting new self-play game")
    print("---------------------------")

    while not game.is_over():
        print(f"\nMove {len(game.move_history) + 1}")
        print(game.board)  # Print current position

        # Get move from MCTS
        move = mcts.get_move(game.board)

        # Store game state and policy
        encoded_state = encode_board(game.board)
        policy = np.zeros(config.policy_output_dim)
        move_idx = mcts.encode_move(
            move[0], move[1]
        )  # Assuming move is (from_pos, to_pos)
        policy[move_idx] = 1.0

        states.append(encoded_state)
        policies.append(policy)

        # Make move
        game.make_move_coords(move[0], move[1], f"{move[0]}{move[1]}")

    # Game is over - get result
    result = game.get_result()
    print(f"\nGame over. Result: {result}")

    # Fill in the values array based on game result
    values = [result if i % 2 == 0 else -result for i in range(len(states))]

    return states, policies, values


def generate_games(mcts: MCTS, config) -> List[Tuple]:
    """Generate multiple self-play games"""
    games_data = []

    for game_idx in range(config.n_games_per_iteration):
        if game_idx % 10 == 0:  # Progress update
            print(f"Generating game {game_idx + 1}/{config.n_games_per_iteration}")

        game_data = play_self_play_game(mcts, config)
        if game_data:
            games_data.append(game_data)

    return games_data


def create_batches(games: List[Tuple], batch_size: int):
    """Create training batches from game data"""
    # Flatten all games into (state, policy, value) tuples
    all_examples = []
    for states, policies, values in games:
        all_examples.extend(zip(states, policies, values))

    # Shuffle examples
    np.random.shuffle(all_examples)

    # Create batches
    for i in range(0, len(all_examples), batch_size):
        batch = all_examples[i : i + batch_size]
        if len(batch) < batch_size:
            continue

        # Unzip batch
        states, policies, values = zip(*batch)

        # Convert to MLX arrays
        states = mx.array(states)
        policies = mx.array(policies)
        values = mx.array(values)

        yield states, policies, values
