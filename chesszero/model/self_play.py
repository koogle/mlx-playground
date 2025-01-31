import numpy as np
from chess_engine.game import ChessGame
from config.model_config import ModelConfig
from utils.random_player import RandomPlayer
from model.mcts import MCTS
from utils.board_utils import encode_board
from typing import List, Tuple
import mlx.core as mx
from tqdm import tqdm
import time


def play_self_play_game(mcts: MCTS, config) -> Tuple[List, List, List]:
    """Play a single game of self-play and return the training data"""
    perf_stats = {
        "move_generation": 0.0,
        "state_encoding": 0.0,
        "policy_creation": 0.0,
        "move_execution": 0.0,
        "total_time": 0.0,
    }

    start_time = time.time()
    game = ChessGame()
    states, policies, values = [], [], []

    print("\nStarting new self-play game")
    print("---------------------------")

    move_count = 0
    for move_count in tqdm(range(201), desc="Playing game"):
        if game.is_over():
            break

        if move_count % 20 == 0:
            print(f"\nMove {move_count}")
            print(game.board)

        # Time move generation
        t0 = time.time()
        move = mcts.get_move(game.board)
        perf_stats["move_generation"] += time.time() - t0

        # Time state encoding
        t0 = time.time()
        encoded_state = encode_board(game.board)
        perf_stats["state_encoding"] += time.time() - t0

        # Time policy creation
        t0 = time.time()
        policy = np.zeros(config.policy_output_dim)
        move_idx = mcts.encode_move(move[0], move[1])
        policy[move_idx] = 1.0
        perf_stats["policy_creation"] += time.time() - t0

        states.append(encoded_state)
        policies.append(policy)

        # Time move execution
        t0 = time.time()
        game.make_move_coords(move[0], move[1], f"{move[0]}{move[1]}")
        perf_stats["move_execution"] += time.time() - t0

        move_count += 1

    perf_stats["total_time"] = time.time() - start_time

    # Print performance statistics
    print("\nSelf-play Performance Statistics:")
    print(f"Total game time: {perf_stats['total_time']:.2f}s")
    print(f"Moves: {move_count}")
    print(f"Time per move: {perf_stats['total_time']/move_count:.3f}s")
    print("\nTime breakdown:")
    print(
        f"Move generation: {perf_stats['move_generation']/perf_stats['total_time']*100:.1f}%"
    )
    print(
        f"State encoding: {perf_stats['state_encoding']/perf_stats['total_time']*100:.1f}%"
    )
    print(
        f"Policy creation: {perf_stats['policy_creation']/perf_stats['total_time']*100:.1f}%"
    )
    print(
        f"Move execution: {perf_stats['move_execution']/perf_stats['total_time']*100:.1f}%"
    )

    # Game is over - get result
    result = game.get_result()
    print(f"\nGame over. Result: {result}")

    # Fill in the values array based on game result
    values = [result if i % 2 == 0 else -result for i in range(len(states))]

    return states, policies, values


def generate_games(mcts, config: ModelConfig) -> List:
    """Generate multiple self-play games sequentially"""
    print(f"Generating {config.n_games_per_iteration} games")

    all_games = []
    for i in range(config.n_games_per_iteration):
        print(f"\nGenerating game {i+1}/{config.n_games_per_iteration}")
        states, policies, values = play_self_play_game(mcts, config)
        all_games.append((states, policies, values))

    return all_games


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

        # Convert to MLX arrays with explicit type conversion
        states = mx.array(np.array(states, dtype=np.float32))
        policies = mx.array(np.array(policies, dtype=np.float32))
        values = mx.array(np.array(values, dtype=np.float32))[
            :, None
        ]  # Add batch dimension

        yield states, policies, values


def generate_random_opponent_games(mcts: MCTS, config) -> List[Tuple]:
    """Generate games playing against a random opponent"""
    games_data = []
    random_player = RandomPlayer()

    for game_idx in range(config.n_games_per_iteration):
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
