import mlx.core as mx
import mlx.optimizers as optim
from typing import Tuple, Optional
from chess_engine.game import ChessGame
from model.network import ChessNet
from model.mcts import MCTS
from model.self_play import (
    generate_games,
    create_batches,
)
from utils.random_player import RandomPlayer
from config.model_config import ModelConfig
from pathlib import Path
import time
import logging
from tqdm import tqdm


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

        # Set up training log file
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.training_log = self.log_dir / f"training_log_{timestamp}.txt"

        self.last_eval_time = time.time()
        self.logger = logging.getLogger(__name__)

        # Add file handler for training log
        file_handler = logging.FileHandler(self.training_log)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        # Log initial setup
        self.logger.info("=== Training Session Started ===")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Model config:\n{vars(config)}")

        if resume_epoch is not None:
            self.logger.info(f"Resuming from epoch {resume_epoch}")
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
        n_epochs = n_epochs or self.config.n_epochs
        start_time = time.time()

        for epoch in range(self.start_epoch, n_epochs):
            epoch_start_time = time.time()

            self.logger.info(f"Epoch {epoch + 1}/{n_epochs}")

            # Set model and MCTS to training mode
            self.model.train()  # Explicitly set to train mode
            self.mcts.training = True
            self.mcts.debug = self.config.debug

            # Generate self-play games
            game_start_time = time.time()
            games = generate_games(self.mcts, self.config)
            game_time = time.time() - game_start_time
            self.logger.info(f"Self-play completed in {game_time:.1f}s")

            # Log game statistics
            total_positions = sum(len(states) for states, _, _ in games)
            avg_moves = total_positions / len(games) if games else 0
            self.logger.info(f"Average moves per game: {avg_moves:.1f}")
            self.logger.info(f"Total training positions: {total_positions}")

            # Train on game data
            self.logger.info("Training on game data...")
            total_loss = 0
            n_batches = 0
            policy_loss = 0
            value_loss = 0

            batches = list(create_batches(games, self.config.batch_size))
            for batch in tqdm(batches, desc="Training batches"):
                loss, p_loss, v_loss = self.train_on_batch(batch)
                total_loss += loss
                policy_loss += p_loss
                value_loss += v_loss
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            avg_policy_loss = policy_loss / n_batches if n_batches > 0 else 0
            avg_value_loss = value_loss / n_batches if n_batches > 0 else 0
            epoch_time = time.time() - epoch_start_time

            # Log detailed training statistics
            self.logger.info(f"Epoch completed in {epoch_time:.1f}s")
            self.logger.info(f"Average total loss: {avg_loss:.4f}")
            self.logger.info(f"Average policy loss: {avg_policy_loss:.4f}")
            self.logger.info(f"Average value loss: {avg_value_loss:.4f}")

            # Check if it's time for evaluation
            current_time = time.time()
            minutes_since_last_eval = (current_time - self.last_eval_time) / 60

            if minutes_since_last_eval >= self.config.eval_interval_minutes:
                self.logger.info("\n=== Running Evaluation ===")
                win_rate = self.evaluate()
                self.logger.info(f"Win rate vs random: {win_rate:.2%}")

                # Save checkpoint after evaluation
                self.logger.info("Saving checkpoint...")
                self.model.save_checkpoint(
                    self.checkpoint_dir, epoch + 1, optimizer_state=self.optimizer.state
                )
                self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")

                # Update last eval time
                self.last_eval_time = current_time

        total_time = time.time() - start_time
        self.logger.info("\n=== Training Completed ===")
        self.logger.info(f"Total training time: {total_time/3600:.1f} hours")

    def get_game_outcome(self, state: str) -> float:
        """Convert game state to value with adjusted rewards

        Returns:
            float: 1.0 for win, -1.0 for loss, -0.5 for draw/timeout
        """
        if "Checkmate" in state:
            return 1.0 if "White wins" in state else -1.0
        # Draws and timeouts are slightly negative to encourage finding wins
        return -0.5

    def train_on_batch(self, batch):
        """Train on a single batch of data with adjusted loss calculation"""
        states, policies, values = batch

        # Debug output for input tensors
        self.logger.debug(f"States shape: {states.shape}, dtype: {states.dtype}")
        self.logger.debug(f"Policies shape: {policies.shape}, dtype: {policies.dtype}")
        self.logger.debug(f"Values shape: {values.shape}, dtype: {values.dtype}")
        self.logger.debug(f"Sample values: {values[:5]}")  # Show first 5 values
        self.logger.debug(
            f"Sample policy sums: {[mx.sum(p).item() for p in policies[:5]]}"
        )  # Should be close to 1

        # Ensure model is in training mode
        self.model.train()

        @mx.compile
        def loss_fn(model_params, states, policies, values):
            self.model.update(model_params)
            pred_policies, pred_values = self.model(states)

            # Debug predictions
            self.logger.debug(f"Pred values: {pred_values[:5]}")
            self.logger.debug(
                f"Pred policy sums: {[mx.sum(p).item() for p in pred_policies[:5]]}"
            )

            # Policy loss calculation
            p_loss = -mx.mean(mx.sum(policies * mx.log(pred_policies + 1e-8), axis=1))

            # Value loss calculation
            v_loss = mx.mean(mx.square(values - pred_values))

            # L2 regularization
            l2_lambda = 5e-5
            l2_reg = l2_lambda * sum(
                mx.sum(mx.square(p)) for p in model_params.values()
            )

            # Debug loss components
            self.logger.debug(f"Policy loss: {p_loss.item()}")
            self.logger.debug(f"Value loss: {v_loss.item()}")
            self.logger.debug(f"L2 reg: {l2_reg.item()}")

            total_loss = p_loss + v_loss + l2_reg
            return total_loss, (p_loss, v_loss)

        # Compute loss and gradients
        (loss, (p_loss, v_loss)), grads = mx.value_and_grad(loss_fn, has_aux=True)(
            self.model.parameters(), states, policies, values
        )

        # Debug gradients
        grad_norms = {name: mx.sum(mx.square(g)).item() for name, g in grads.items()}
        self.logger.debug(f"Gradient norms: {grad_norms}")

        # Update model parameters
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        # Final loss values
        self.logger.debug(
            f"Final losses - Total: {loss.item()}, Policy: {p_loss.item()}, Value: {v_loss.item()}"
        )

        return loss.item(), p_loss.item(), v_loss.item()

    def evaluate(self, n_games: int = 100) -> float:
        """Evaluate current model against random player"""
        self.logger.info("\nEvaluating against random player...")

        # Set MCTS to evaluation mode
        self.mcts.training = False
        self.mcts.debug = self.config.debug  # Ensure debug flag is passed through

        random_player = RandomPlayer()
        total_wins = 0
        total_games = 0

        # Play as both white and black
        for color in [0, 1]:
            for game_idx in range(n_games // 2):
                # Show board for first evaluation game of each color if enabled
                show_board = self.config.display_eval_game and game_idx == 0

                wins, games = self.play_evaluation_game(
                    random_player, color, show_board=show_board
                )
                total_wins += wins
                total_games += games

        return total_wins / total_games

    def play_evaluation_game(
        self, opponent, mcts_player_color, show_board: bool = False
    ) -> Tuple[int, int]:
        """Play a single evaluation game with adjusted scoring"""
        game = ChessGame()
        moves_without_progress = 0
        max_moves_without_progress = 50  # Fifty move rule

        while True:
            state = game.get_game_state()
            if state != "Normal":
                if "Checkmate" in state:
                    winner = "Black" if "White wins" in state else "White"
                    if (winner == "White" and mcts_player_color == 0) or (
                        winner == "Black" and mcts_player_color == 1
                    ):
                        return 1, 1  # Win
                    return 0, 1  # Loss
                return 0.25, 1  # Draw is worth more than a loss but less than a win

            # Check for move limit or repetition
            if moves_without_progress >= max_moves_without_progress:
                return 0.25, 1  # Draw due to no progress

            if game.get_current_turn() == mcts_player_color:
                move = self.mcts.get_move(game.board)
            else:
                move = opponent.select_move(game.board)

            if not move:
                return 0.25, 1  # Draw - no valid moves

            # Track moves without progress
            if game.board.is_capture_or_pawn_move(move[0], move[1]):
                moves_without_progress = 0
            else:
                moves_without_progress += 1

            game.make_move(move[0], move[1])

            if show_board:
                print(game.board)
