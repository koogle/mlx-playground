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
import random
import logging


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


def generate_games(mcts: MCTS, config: ModelConfig) -> List[Tuple]:
    """Generate self-play games"""
    logger = logging.getLogger(__name__)
    logger.info(f"\nGenerating {config.n_games_per_iteration} self-play games")
    games = []

    for game_idx in tqdm(range(config.n_games_per_iteration), desc="Generating games"):
        game = ChessGame()
        total_moves = 0  # Track total moves

        states, policies, values = [], [], []

        # Add a maximum move limit to prevent infinite games
        max_moves = 200
        while not game.board.is_game_over() and total_moves < max_moves:
            state = game.board.state
            move = mcts.get_move(
                game.board
            )  # This will set up the MCTS tree and set root_node

            if not move:
                logger.info(f"No valid moves found after {total_moves} moves")
                break

            # Ensure we have a root_node
            if not mcts.root_node:
                continue

            policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)
            states.append(state)
            policies.append(policy)

            game.make_move(move[0], move[1])
            total_moves += 1

            # Log board state periodically
            if total_moves % 10 == 0:
                logger.debug(
                    f"\nGame {game_idx + 1}, Move {total_moves}:\n{game.board}"
                )

        # Add game result
        result = game.board.get_game_result()
        values.extend([result] * len(states))

        if len(states) > 0:  # Only add if we have moves
            games.append((states, policies, values))
        logger.info(
            f"Game {game_idx + 1} completed with {total_moves} moves, result: {result}"
            + (f" (max moves reached)" if total_moves >= max_moves else "")
        )

    if len(games) == 0:
        logger.warning("No valid games were generated!")
    logger.info(f"Generated {len(games)} games with valid moves")
    return games


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
