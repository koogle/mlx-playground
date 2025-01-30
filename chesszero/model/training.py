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
from pathlib import Path


class Trainer:
    """Handles training, self-play, and evaluation of the chess model"""

    def __init__(
        self,
        config: ModelConfig,
        checkpoint_dir: str = "checkpoints",
        resume_epoch: Optional[int] = None,
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if resume_epoch is not None:
            print(f"Resuming from epoch {resume_epoch}")
            self.model, state = ChessNet.load_checkpoint(checkpoint_dir, resume_epoch)
            self.start_epoch = resume_epoch + 1
            # Recreate optimizer with saved state if needed
            self.optimizer = optim.SGD(
                learning_rate=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
            if state.get("optimizer_state"):
                self.optimizer.state.update(state["optimizer_state"])
        else:
            self.model = ChessNet(config)
            self.start_epoch = 0
            self.optimizer = optim.SGD(
                learning_rate=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )

        self.mcts = MCTS(self.model, config)

    def train(self, n_epochs: Optional[int] = None):
        """Main training loop"""
        self.mcts.training = True  # Enable training mode
        n_epochs = n_epochs or self.config.n_epochs

        for epoch in range(self.start_epoch, n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            # Initial training against random opponent
            if epoch < 5:  # First 5 epochs
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

            # Save checkpoint every N epochs
            if (epoch + 1) % 5 == 0:
                print("Saving checkpoint...")
                self.model.save_checkpoint(
                    self.checkpoint_dir, epoch + 1, optimizer_state=self.optimizer.state
                )

            # Evaluate periodically
            if (epoch + 1) % 5 == 0:
                win_rate = self.evaluate()
                print(f"Win rate vs random: {win_rate:.2%}")

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

        @mx.compile
        def loss_fn(params, states, policies, values):
            self.model.update(params)
            pred_policies, pred_values = self.model(states)
            policy_loss = -mx.mean(policies * mx.log(pred_policies + 1e-8))
            value_loss = mx.mean((values - pred_values) ** 2)
            return policy_loss + value_loss

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(
            self.model.parameters(), states, policies, values
        )

        # Update model using gradients
        self.optimizer.update(self.model, grads)

        # Evaluate to get actual loss value
        mx.eval(loss)

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
