import numpy as np
from chess_engine.game import ChessGame
from chesszero.chess_engine.bitboard import BitBoard
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
        move = get_move_with_temperature(mcts, game.board, temperature)
        if move is None:
            print("No valid moves found!")
            break

        # Store state and create policy from visits
        encoded_state = encode_board(game.board)
        policy = create_policy_from_visits(mcts.root_node, config.policy_output_dim)

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
    """Generate self-play games"""
    logger = logging.getLogger(__name__)
    logger.info(f"\nGenerating {config.n_games_per_iteration} self-play games")
    games = []

    for game_idx in range(config.n_games_per_iteration):
        print(f"\n=== Game {game_idx + 1}/{config.n_games_per_iteration} ===")
        game = ChessGame()
        total_moves = 0
        states, policies, values = [], [], []

        print("\nInitial position:")
        print(game.board)

        while not game.board.is_game_over() and total_moves < 200:
            print(f"\nMove {total_moves + 1}")

            state = game.board.state
            move = mcts.get_move(game.board)

            if not move:
                print("No valid moves found!")
                break

            if not mcts.root_node:
                continue

            # Print the move being made
            from_square = f"{chr(move[0][1] + 97)}{move[0][0] + 1}"
            to_square = f"{chr(move[1][1] + 97)}{move[1][0] + 1}"
            print(f"AI plays: {from_square}{to_square}")

            policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)
            states.append(state)
            policies.append(policy)

            game.make_move(move[0], move[1])
            print("\nPosition after move:")
            print(game.board)

            total_moves += 1

        # Game is over - print outcome
        result = game.board.get_game_result()
        values.extend([result] * len(states))

        print("\n=== Game Complete ===")
        print("\nFinal position:")
        print(game.board)
        print(f"Result: {result}")
        print(f"Total moves: {total_moves}")
        print("===================\n")

        if len(states) > 0:
            games.append((states, policies, values))

        logger.info(
            f"Game {game_idx + 1} completed with {total_moves} moves, result: {result}"
            + (" (max moves reached)" if total_moves >= 200 else "")
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


def get_move_with_temperature(mcts: MCTS, board, temperature: float):
    """Select move based on visit count distribution with temperature"""
    mcts.get_move(board)  # Run MCTS simulations

    visits = np.array([child.visit_count for child in mcts.root_node.children.values()])
    moves = list(mcts.root_node.children.keys())

    if temperature == 0:
        # Select most visited move
        return moves[np.argmax(visits)]

    # Apply temperature
    visits = visits ** (1 / temperature)
    probs = visits / np.sum(visits)

    # Sample move based on visit count distribution
    return moves[np.random.choice(len(moves), p=probs)]


def create_policy_from_visits(root_node, policy_output_dim: int):
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
    # Convert uint8 to float32 when creating MLX array
    return mx.array(board.state, dtype=mx.float32)
