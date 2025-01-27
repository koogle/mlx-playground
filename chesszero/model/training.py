import mlx.core as mx
import mlx.optimizers as optim
from typing import Tuple, List, Optional
from chess_engine.board import Color
from chess_engine.game import ChessGame
from model.network import ChessNet
from model.mcts import MCTS
from utils.random_player import RandomPlayer
from utils.board_utils import encode_board
from config.model_config import ModelConfig


class Trainer:
    """Handles training, self-play, and evaluation of the chess model"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = ChessNet(config)
        self.optimizer = optim.SGD(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.mcts = MCTS(self.model, config)

    def train(self, n_epochs: Optional[int] = None):
        """Main training loop"""
        n_epochs = n_epochs or self.config.n_epochs

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            # Generate self-play games
            print("Generating self-play games...")
            games = self.generate_games()

            # Train on game data
            print("Training on game data...")
            total_loss = 0
            n_batches = 0

            for batch in self.create_batches(games):
                loss = self.train_on_batch(batch)
                total_loss += loss
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f"Average loss: {avg_loss:.4f}")

            # Evaluate periodically
            if (epoch + 1) % 5 == 0:
                win_rate = self.evaluate()
                print(f"Win rate vs random: {win_rate:.2%}")

                # Early success check
                if epoch < 20 and win_rate > 0.8:
                    print("Early success achieved!")

    def generate_games(self) -> List[Tuple]:
        """Generate self-play games and return training data"""
        games_data = []

        for game_idx in range(self.config.n_games_per_iteration):
            if game_idx % 10 == 0:  # Progress update
                print(
                    f"Generating game {game_idx + 1}/{self.config.n_games_per_iteration}"
                )

            game_data = self.play_self_play_game()
            if game_data:
                games_data.append(game_data)

        return games_data

    def play_self_play_game(self) -> Optional[Tuple]:
        """Play a single game of self-play and return the training data"""
        game = ChessGame()
        states, policies, values = [], [], []

        while True:
            # Store current state
            states.append(encode_board(game.board))

            # Get move from MCTS
            move = self.mcts.get_move(game.board)
            if not move:
                break

            # Store policy (visit count distribution)
            policy = self.get_policy_distribution(move)
            policies.append(policy)

            # Make move
            game.make_move(move[0], move[1])

            # Check game end
            state = game.get_game_state()
            if state != "Normal":
                value = self.get_game_outcome(state)
                # Fill in values for all positions
                values = [value * ((-1) ** i) for i in range(len(states))]
                return states, policies, values

        return None

    def get_policy_distribution(self, root_node):
        """Convert MCTS visit counts to policy distribution"""
        policy = mx.zeros(self.config.policy_output_dim)

        for move, child in root_node.children.items():
            move_idx = self.mcts.encode_move(move[0], move[1])
            if move_idx < len(policy):
                policy[move_idx] = child.visit_count

        # Normalize
        policy = policy / mx.sum(policy)
        return policy

    def get_game_outcome(self, state: str) -> float:
        """Convert game state to value"""
        if "Checkmate" in state:
            return 1.0 if "White wins" in state else -1.0
        return 0.0  # Draw

    def create_batches(self, games):
        """Create training batches from game data"""
        # Flatten all games into (state, policy, value) tuples
        all_examples = []
        for states, policies, values in games:
            all_examples.extend(zip(states, policies, values))

        # Shuffle examples
        mx.random.shuffle(all_examples)

        # Create batches
        for i in range(0, len(all_examples), self.config.batch_size):
            batch = all_examples[i : i + self.config.batch_size]
            if len(batch) < self.config.batch_size:
                continue

            # Unzip batch
            states, policies, values = zip(*batch)

            # Convert to MLX arrays
            states = mx.array(states)
            policies = mx.array(policies)
            values = mx.array(values)

            yield states, policies, values

    def train_on_batch(self, batch):
        """Train on a single batch of data"""
        states, policies, values = batch
        loss = self.model.loss_fn(states, policies, values)
        self.optimizer.step(loss)
        return loss.item()

    def evaluate(self, n_games: int = 100) -> float:
        """Evaluate current model against random player"""
        print("\nEvaluating against random player...")
        random_player = RandomPlayer()
        total_wins = 0
        total_games = 0

        # Play as both white and black
        for color in [Color.WHITE, Color.BLACK]:
            for game_idx in range(n_games // 2):
                if game_idx % 10 == 0:  # Progress update
                    print(
                        f"Playing game {game_idx + 1}/{n_games//2} as {'White' if color == Color.WHITE else 'Black'}"
                    )

                wins, games = self.play_evaluation_game(random_player, color)
                total_wins += wins
                total_games += games

        return total_wins / total_games

    def play_evaluation_game(self, opponent, mcts_player_color) -> Tuple[int, int]:
        """Play a single evaluation game"""
        game = ChessGame()

        while True:
            state = game.get_game_state()
            if state != "Normal":
                if "Checkmate" in state:
                    winner = "Black" if "White wins" in state else "White"
                    if (winner == "White" and mcts_player_color == Color.WHITE) or (
                        winner == "Black" and mcts_player_color == Color.BLACK
                    ):
                        return 1, 1
                    return 0, 1
                return 0.5, 1  # Draw

            if game.get_current_turn() == mcts_player_color:
                move = self.mcts.get_move(game.board)
            else:
                move = opponent.select_move(game.board)

            if not move:
                return 0.5, 1  # Draw - no valid moves

            game.make_move(move[0], move[1])
