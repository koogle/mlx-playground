import numpy as np
from chess_engine.game import ChessGame
from model.mcts import MCTS
from utils.board_utils import encode_board


def generate_games(model, config):
    """Generate self-play games and return training data"""
    games_data = []
    mcts = MCTS(model, config)

    for _ in range(config.n_games_per_iteration):
        game = ChessGame()
        states, policies, values = [], [], []

        while True:
            # Store current state
            states.append(encode_board(game.board))

            # Get move probabilities from MCTS
            root = mcts.get_move(game.board)
            if not root:
                break

            # Store policy (visit count distribution)
            policy = np.zeros(config.policy_output_dim)
            for move, child in root.children.items():
                move_idx = mcts.encode_move(move[0], move[1])
                if move_idx < len(policy):
                    policy[move_idx] = child.visit_count
            policy = policy / np.sum(policy)
            policies.append(policy)

            # Make move
            game.make_move(root[0], root[1])

            # Check game end
            state = game.get_game_state()
            if state != "Normal":
                # Determine game outcome
                if "Checkmate" in state:
                    value = 1.0 if "White wins" in state else -1.0
                else:
                    value = 0.0  # Draw

                # Fill in values for all positions
                values = [value * ((-1) ** i) for i in range(len(states))]
                games_data.append((states, policies, values))
                break

    return games_data


def create_batches(games, batch_size):
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

        # Convert to arrays
        states = np.stack(states)
        policies = np.stack(policies)
        values = np.array(values)

        yield states, policies, values
