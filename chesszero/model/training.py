import mlx.core as mx
import mlx.optimizers as optim
from typing import Tuple, Optional
from chess_engine.game import ChessGame
from model.network import ChessNet
from model.mcts import MCTS
from model.self_play import (
    create_batches,
    generate_games,
)
from utils.random_player import RandomPlayer
from config.model_config import ModelConfig
from pathlib import Path
import time
import logging
from tqdm import tqdm
import gc
import psutil
import os
import traceback


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

        self.logger = logging.getLogger(__name__)

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

    def train(self, n_epochs: Optional[int] = None, n_workers: int = 5):
        """Main training loop with parallel game generation"""
        n_epochs = n_epochs or self.config.n_epochs
        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, n_epochs):
                epoch_start_time = time.time()
                self.logger.info(f"Epoch {epoch + 1}/{n_epochs}")
                self.logger.info(f"Generating games using {n_workers} workers...")
                self.model.train()

                # Generate self-play games in parallel
                games = generate_games(self.model, self.config, n_workers)

                # Create batches with memory tracking
                batches = list(create_batches(games, self.config.batch_size))

                # Clear games data after creating batches
                del games
                gc.collect()

                # Train on game data with periodic cleanup
                self.logger.info("Training on game data...")
                total_loss = 0
                n_batches = 0
                policy_loss = 0
                value_loss = 0

                for batch_idx, batch in enumerate(
                    tqdm(batches, desc="Training batches")
                ):
                    try:

                        loss, p_loss, v_loss = self.train_on_batch(batch)
                        total_loss += loss
                        policy_loss += p_loss
                        value_loss += v_loss
                        n_batches += 1

                    except Exception as e:
                        self.logger.error(f"\nError in batch {batch_idx}:")
                        self.logger.error(str(e))
                        self.logger.error(traceback.format_exc())
                        continue

                # Calculate averages and log results
                avg_loss = total_loss / n_batches if n_batches > 0 else 0
                avg_policy_loss = policy_loss / n_batches if n_batches > 0 else 0
                avg_value_loss = value_loss / n_batches if n_batches > 0 else 0

                epoch_time = time.time() - epoch_start_time

                # Log detailed training statistics
                self.logger.info(f"Epoch completed in {epoch_time:.1f}s")
                self.logger.info(f"Average total loss: {avg_loss:.4f}")
                self.logger.info(f"Average policy loss: {avg_policy_loss:.4f}")
                self.logger.info(f"Average value loss: {avg_value_loss:.4f}")

                # Check if it's time for evaluation based on epoch interval
                if (epoch + 1) % self.config.eval_interval_epochs == 0:
                    self.logger.info("\n=== Running Evaluation ===")
                    win_rate, wins, losses, draws = self.evaluate()
                    self.logger.info(f"Win rate vs random: {win_rate:.2%}")
                    self.logger.info(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

                    # Save checkpoint after evaluation
                    self.logger.info("Saving checkpoint...")
                    self.model.save_checkpoint(
                        self.checkpoint_dir,
                        epoch + 1,
                        optimizer_state=self.optimizer.state,
                    )
                    self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")

            total_time = time.time() - start_time
            self.logger.info("\n=== Training Completed ===")
            self.logger.info(f"Total training time: {total_time/3600:.1f} hours")

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted. Saving final checkpoint...")
            self.model.save_checkpoint(
                self.checkpoint_dir,
                epoch + 1,
                optimizer_state=self.optimizer.state,
                interrupted=True,
            )
            self.logger.info(f"Final checkpoint saved at epoch {epoch + 1}")
            raise

    def train_on_batch(self, batch):
        """Train on a single batch of data with memory optimizations"""
        states, policies, values = batch  # Unpack the batch correctly

        try:
            # Track memory usage
            mem_before = self._get_memory_usage()
            self.logger.debug(f"Memory before training step: {mem_before}")

            @mx.compile  # Compile for better performance
            def loss_fn(model_params, states, policies, values):
                self.model.update(model_params)
                pred_policies, pred_values = self.model(states)

                # Ensure pred_values is a proper MLX array and reshape if needed
                if isinstance(pred_values, dict):
                    pred_values = mx.array(list(pred_values.values()), dtype=mx.float32)
                pred_values = mx.reshape(pred_values, values.shape)

                # Policy loss calculation (with numerical stability)
                p_loss = -mx.mean(
                    mx.sum(policies * mx.log(pred_policies + 1e-8), axis=1)
                )

                # Value loss calculation needs to be scaled against policy loss
                v_loss = mx.mean(mx.square(values - pred_values)) * 10

                total_loss = p_loss + v_loss
                return total_loss, (p_loss, v_loss)

            # Compute loss and gradients in one step
            (loss, (p_loss, v_loss)), grads = mx.value_and_grad(loss_fn)(
                self.model.parameters(), states, policies, values
            )

            # Update model parameters and immediately evaluate
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)

            # Clean up intermediate tensors
            del grads
            gc.collect()  # Force garbage collection

            mem_after = self._get_memory_usage()
            self.logger.debug(f"Memory after training step: {mem_after}")

            # Memory change logging
            mem_diff = {k: mem_after[k] - mem_before[k] for k in mem_before}
            self.logger.debug(f"Memory change during step: {mem_diff}")

            return loss.item(), p_loss.item(), v_loss.item()

        except Exception as e:
            self.logger.error("\nError in training step:")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception message: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.logger.error(f"Memory state: {self._get_memory_usage()}")
            self.logger.error(
                f"Batch shapes - States: {states.shape}, Policies: {policies.shape}, Values: {values.shape}"
            )
            raise

    def _get_memory_usage(self):
        """Get current memory usage information"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss": mem_info.rss / (1024 * 1024),  # RSS in MB
            "vms": mem_info.vms / (1024 * 1024),  # VMS in MB
            "system_percent": psutil.virtual_memory().percent,
        }

    def evaluate(self, n_games: int = 100) -> Tuple[float, int, int, int]:
        """Evaluate current model against random player"""
        self.logger.info("\nEvaluating against random player...")

        random_player = RandomPlayer()
        wins = 0
        losses = 0
        draws = 0
        total_games = 0

        # Play as both white and black
        for color in [0, 1]:
            for _ in range(n_games // 2):
                # Show board for first evaluation game of each color if enabled
                # show_board = self.config.display_eval_game and game_idx == 0

                result = self.play_evaluation_game(random_player, color)
                if result == 1:
                    wins += 1
                elif result == 0:
                    losses += 1
                else:
                    draws += 1
                total_games += 1

        win_rate = wins / total_games
        return win_rate, wins, losses, draws

    def play_evaluation_game(self, opponent, mcts_player_color) -> float:
        """Play a single evaluation game with adjusted scoring"""
        game = ChessGame()
        move_count = 0
        position_history = {}  # Track repeated positions
        opponent_color = 1 - mcts_player_color

        # Maximum moves before forced draw (100 moves = 50 full moves)
        MAX_MOVES = 200
        print("Playing eval game")

        while not game.board.is_game_over():
            # Check for draw by repetition
            pos_hash = game.board.get_hash()
            position_history[pos_hash] = position_history.get(pos_hash, 0) + 1
            if position_history[pos_hash] >= 3:
                return 0.5  # Draw by threefold repetition

            # Check for move limit exceeded
            if move_count >= MAX_MOVES:
                return 0.5  # Draw by move limit

            if game.get_current_turn() == mcts_player_color:
                mcts = MCTS(self.model, self.config)
                move = mcts.get_move(game.board, temperature=1.0)
            else:
                move = opponent.select_move(game.board)

            if not move:
                return 0.5  # Draw - no valid moves

            game.make_move(move[0], move[1])

            print(game.board)
            print(move_count)
            move_count += 1

        # Check game outcome using BitBoard methods
        if game.board.is_checkmate(mcts_player_color):
            return 0  # Loss - we're in checkmate
        elif game.board.is_checkmate(opponent_color):
            return 1  # Win - opponent is in checkmate
        elif game.board.is_stalemate(mcts_player_color) or game.board.is_stalemate(
            opponent_color
        ):
            return 0.5  # Draw - stalemate
        elif game.board.is_draw():
            return 0.5  # Draw - insufficient material or other draw condition

        return 0.5  # Fallback draw
