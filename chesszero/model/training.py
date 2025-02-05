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
from config.model_config import ModelConfig
from pathlib import Path
import time
import logging
from tqdm import tqdm
import gc
import psutil
import os
import traceback
import multiprocessing as mp
from queue import Empty
from mlx.utils import tree_flatten


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

        # Add best model tracking
        self.best_model = None
        self.best_model_win_rate = 0.0

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
            if state.get("best_model_win_rate"):
                self.best_model_win_rate = state.get("best_model_win_rate", 0.0)
                self.best_model, _ = ChessNet.load_checkpoint(
                    checkpoint_dir, resume_epoch
                )
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
                self.logger.info(f"Training on game data in epoch {epoch + 1}...")
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

                # Evaluate against best model
                if (epoch + 1) % self.config.eval_interval_epochs == 0:
                    self.logger.info("\n=== Running Evaluation ===")
                    win_rate, wins, losses, draws = self.evaluate_against_best()

                    self.logger.info(f"Win rate vs best: {win_rate:.2%}")
                    self.logger.info(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

                    if win_rate > 0.55:
                        # New model is better, update best model
                        self.logger.info("New model is better, updating best model")
                        self.best_model.load_weights(
                            tree_flatten(self.model.parameters())
                        )
                        self.best_model_win_rate = win_rate
                    else:
                        # Revert to best model
                        self.logger.info("Reverting to best model")
                        self.model.load_weights(
                            tree_flatten(self.best_model.parameters())
                        )

                    # Save checkpoint after evaluation
                    self.logger.info("Saving checkpoint...")
                    self.model.save_checkpoint(
                        self.checkpoint_dir,
                        epoch + 1,
                        optimizer_state=self.optimizer.state,
                        best_model_win_rate=self.best_model_win_rate,
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
        states, policies, values = batch

        try:
            # Track memory usage
            mem_before = self._get_memory_usage()
            self.logger.debug(f"Memory before training step: {mem_before}")

            @mx.compile
            def loss_fn(model_params, states, policies, values):
                self.model.update(model_params)
                pred_policies, pred_values = self.model(states)

                # Simplified loss calculation without manual bonuses/penalties
                p_loss = -mx.mean(
                    mx.sum(policies * mx.log(pred_policies + 1e-8), axis=1)
                )
                if isinstance(pred_values, dict):
                    pred_values = mx.array(list(pred_values.values()), dtype=mx.float32)
                v_loss = mx.mean(mx.square(values - pred_values))
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

    def play_evaluation_game_worker(
        self,
        model: ChessNet,
        best_model: ChessNet,
        config: ModelConfig,
        game_id: int,
        mcts_player_color: int,
        result_queue: mp.Queue,
    ):
        """Worker process that plays a single evaluation game"""
        try:
            game = ChessGame()
            move_count = 0

            opponent_color = 1 - mcts_player_color

            while not game.board.is_game_over():
                # Check for draw by repetition
                if game.get_current_turn() == mcts_player_color:
                    mcts = MCTS(model, config)
                    move = mcts.get_move(game.board, temperature=1.0)
                else:
                    mcts = MCTS(best_model, config)
                    move = mcts.get_move(game.board, temperature=1.0)

                if not move:
                    result_queue.put((game_id, 0.4))  # Slight penalty for draws
                    return

                game.make_move(move[0], move[1])
                move_count += 1

            # Determine game result
            if game.board.is_checkmate(mcts_player_color):
                result = 0  # Loss
            elif game.board.is_checkmate(opponent_color):
                result = 1  # Win
            elif game.board.is_stalemate(mcts_player_color) or game.board.is_stalemate(
                opponent_color
            ):
                result = 0.4  # Slight penalty for draws
            elif game.board.is_draw():
                result = 0.4  # Slight penalty for draws
            else:
                result = 0.4  # Slight penalty for draws

            result_queue.put((game_id, result))

        except Exception as e:
            logging.error(f"Error in evaluation game {game_id}: {str(e)}")
            logging.error(traceback.format_exc())
            result_queue.put((game_id, None))

    def evaluate_against_best(
        self, n_games: int = 40, max_workers: int = 10
    ) -> Tuple[float, int, int, int]:
        """Evaluate current model against best model"""
        self.logger.info("\nEvaluating against best model...")

        if self.best_model is None:
            self.logger.info("No best model yet, using current model as best")
            self.best_model = ChessNet(self.config)
            self.best_model.load_state_dict(self.model.state_dict())
            return 0.1, 0, 0, 0  # Default to 10% win rate for first evaluation

        wins = 0
        losses = 0
        draws = 0
        total_games = 0

        # Create process context and result queue
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        active_workers = []
        game_id = 0
        games_completed = 0

        try:
            # Play as both white and black
            for color in [0, 1]:
                games_per_color = n_games // 2
                while games_completed < total_games + games_per_color:
                    # Launch new workers up to max_workers
                    while (
                        len(active_workers) < max_workers and game_id < games_per_color
                    ):
                        p = ctx.Process(
                            target=self.play_evaluation_game_worker,
                            args=(
                                self.model,
                                self.best_model,
                                self.config,
                                game_id,
                                color,
                                result_queue,
                            ),
                        )
                        p.start()
                        active_workers.append((p, game_id))
                        game_id += 1

                    # Collect results
                    try:
                        game_id, result = result_queue.get(timeout=1)
                        if result is not None:
                            games_completed += 1
                            if result == 1:
                                wins += 1
                            elif result == 0:
                                losses += 1
                            else:
                                draws += 1
                            self.logger.info(
                                f"Evaluation game {game_id} completed with result: {result}"
                            )

                    except Empty:
                        pass

                    # Clean up completed workers
                    active_workers = [
                        (p, gid) for p, gid in active_workers if p.is_alive()
                    ]

                # Reset game_id for next color
                game_id = 0
                total_games = games_completed

        finally:
            # Clean up any remaining workers
            for p, _ in active_workers:
                if p.is_alive():
                    p.terminate()
                    p.join()

        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0
        return win_rate, wins, losses, draws
