import mlx.core as mx
import mlx.optimizers as optim
from typing import Tuple, List, Optional
from chess_engine.board import Color
from chess_engine.game import ChessGame
from model.network import ChessNet
from model.mcts import MCTS
from model.self_play import (
    generate_games,
    create_batches,
    generate_random_opponent_games,
)
from utils.random_player import RandomPlayer
from utils.board_utils import encode_board
from config.model_config import ModelConfig
import numpy as np


class Trainer:
    """Handles training, self-play, and evaluation of the chess model"""

    def __init__(self, config: ModelConfig, start_with_random: bool = False):
        self.config = config
        self.model = ChessNet(config)
        self.optimizer = optim.SGD(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.mcts = MCTS(self.model, config)
        self.start_with_random = start_with_random
        self.random_player = RandomPlayer()

    def train(self, n_epochs: Optional[int] = None):
        """Main training loop"""
        n_epochs = n_epochs or self.config.n_epochs

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            # Generate training games
            print("Generating training games...")
            if (
                self.start_with_random and epoch < 5
            ):  # Use random opponent for first 5 epochs
                print("Playing against random opponent for initial training...")
                games = generate_random_opponent_games(self.mcts, self.config)
            else:
                games = generate_games(self.mcts, self.config)

            # Train on game data
            print("Training on game data...")
            total_loss = 0
            n_batches = 0

            for batch in create_batches(games, self.config.batch_size):
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

    def train_on_batch(self, batch):
        """Train on a single batch of data"""
        states, policies, values = batch
        # Ensure model is in training mode
        self.model.train()
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
