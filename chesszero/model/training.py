import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn.losses as mx_losses
import mlx.nn as nn
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
from mlx.utils import tree_flatten, tree_map


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

        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.training_log = self.log_dir / f"training_log_{timestamp}.txt"

        self.logger = logging.getLogger(__name__)

        self.logger.info("=== Training Session Started ===")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Model config:\n{vars(config)}")

        self.best_model = None
        self.best_model_win_rate = 0.0

        if resume_epoch is not None:
            self.logger.info(f"Resuming from epoch {resume_epoch}")
            self.model, state = ChessNet.load_checkpoint(checkpoint_dir, resume_epoch)
            self.start_epoch = resume_epoch + 1
            self.optimizer = optim.Adam(learning_rate=config.learning_rate)
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
            self.optimizer = optim.Adam(learning_rate=config.learning_rate)

        self._loss_and_grad = nn.value_and_grad(self.model, self._loss_fn)

    def train(self, n_epochs: Optional[int] = None, n_workers: int = 8):
        """Main training loop with parallel game generation"""
        n_epochs = n_epochs or self.config.n_epochs
        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, n_epochs):
                epoch_start_time = time.time()
                self.logger.info(f"Epoch {epoch + 1}/{n_epochs}")
                self.logger.info(f"Generating games using {n_workers} workers...")
                self.model.eval()

                games = generate_games(self.model, self.config, n_workers)

                batches = list(create_batches(games, self.config.batch_size))

                del games
                gc.collect()

                self.logger.info(f"Training on game data in epoch {epoch + 1}...")
                (avg_loss, avg_policy_loss, avg_value_loss, avg_l2_loss) = (
                    self.train_on_batches(tqdm(batches, desc="Accumulating gradients"))
                )

                epoch_time = time.time() - epoch_start_time

                self.logger.info(f"Epoch completed in {epoch_time:.1f}s")
                self.logger.info(f"Average policy loss: {avg_policy_loss:.4f}")
                self.logger.info(f"Average l2 loss: {avg_l2_loss:.4f}")
                self.logger.info(f"Average value loss: {avg_value_loss:.4f}")
                self.logger.info(f"Average total loss: {avg_loss:.4f}")

                if (epoch + 1) % self.config.eval_interval_epochs == 0:
                    self.logger.info("\n=== Running Evaluation ===")

                    updated = False
                    result = self.evaluate_against_best()

                    if result is None:
                        self.logger.info(
                            "No best model yet, using current model as best"
                        )
                        self.best_model = ChessNet(self.config)
                        self.best_model.load_weights(
                            tree_flatten(self.model.parameters())
                        )
                        updated = True
                    else:
                        win_rate, wins, losses, draws = result

                        self.logger.info(f"Win rate vs best: {win_rate:.2%}")
                        self.logger.info(
                            f"Wins: {wins}, Losses: {losses}, Draws: {draws}"
                        )

                        if win_rate > 0.55:
                            self.logger.info("New model is better, updating best model")
                            self.best_model.load_weights(
                                tree_flatten(self.model.parameters())
                            )
                            self.best_model_win_rate = win_rate
                            updated = True
                        else:
                            self.logger.info("Reverting to best model")
                            self.model.load_weights(
                                tree_flatten(self.best_model.parameters())
                            )

                    if updated:
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

    def _loss_fn(self, states, policies, values):
        policy_logits, pred_values = self.model.forward_logits(states)
        p_loss = mx_losses.cross_entropy(
            logits=policy_logits, targets=policies, reduction="mean"
        )
        v_loss = mx_losses.mse_loss(
            predictions=pred_values.reshape(-1), targets=values, reduction="mean"
        )
        l2_loss = sum(
            1e-6 * mx.sum(mx.square(parameter))
            for _, parameter in tree_flatten(self.model.trainable_parameters())
            if len(parameter.shape) > 1
        )
        return p_loss + v_loss + l2_loss, (p_loss, v_loss, l2_loss)

    def train_on_batches(self, batches):
        """Average gradients over fresh positions and apply exactly one update.

        Learned parameters remain fixed across microbatches. BatchNorm still uses
        each microbatch's statistics; this does not emulate one giant BN batch.
        """
        self.model.train()
        accumulated = None
        weighted_losses = mx.zeros(4)
        n_positions = 0
        # MLX arrays are immutable; retaining references is enough for rollback.
        original_parameters = self.model.parameters()
        try:
            for batch in batches:
                batch_size = len(batch[0])
                if batch_size == 0:
                    continue
                (loss, components), grads = self._loss_and_grad(*batch)
                if accumulated is None:
                    accumulated = tree_map(lambda g: g * batch_size, grads)
                else:
                    accumulated = tree_map(
                        lambda total, g: total + g * batch_size, accumulated, grads
                    )
                weighted_losses = (
                    weighted_losses + mx.stack([loss, *components]) * batch_size
                )
                n_positions += batch_size
                # Materialize each contribution so previous activation graphs can
                # be released instead of accumulating across the entire generation.
                mx.eval(accumulated, weighted_losses, self.model.parameters())

        except Exception:
            # A failed microbatch must not result in a partial generation update.
            self.model.update(original_parameters)
            self.logger.exception(
                "Gradient accumulation failed; generation was not completed"
            )
            raise

        if not n_positions:
            return 0.0, 0.0, 0.0, 0.0

        averaged = tree_map(lambda total: total / n_positions, accumulated)
        self.optimizer.update(self.model, averaged)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return tuple((weighted_losses / n_positions).tolist())

    def train_on_batch(self, batch):
        """Compatibility wrapper for an update containing one microbatch."""
        return self.train_on_batches((batch,))

    def _get_memory_usage(self):
        """Get current memory usage information"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss": mem_info.rss / (1024 * 1024),
            "vms": mem_info.vms / (1024 * 1024),
            "system_percent": psutil.virtual_memory().percent,
        }

    def play_evaluation_game_worker(
        self,
        model: ChessNet,
        best_model: ChessNet,
        config: ModelConfig,
        game_id: int,
        player_color: int,
        result_queue: mp.Queue,
        max_workers: int,
    ):
        """Worker process that plays a single evaluation game and reports its result."""
        try:
            model.eval()
            best_model.eval()
            game = ChessGame()

            pbar = tqdm(
                desc=f"Evaluation game {game_id}",
                position=game_id % max_workers,
                leave=False,
                unit="moves",
                unit_scale=True,
            )

            while not game.board.is_game_over():
                if game.get_current_turn() == player_color:
                    mcts = MCTS(model, config)
                else:
                    mcts = MCTS(best_model, config)

                move = mcts.get_move(game.board, temperature=1.0)
                if not move:
                    break

                game.make_move(move[0], move[1])

                pbar.update(1)

            pbar.close()

            result_queue.put(
                {
                    "game_id": game_id,
                    "result": game.board.get_game_result(player_color),
                    "perspective_color": player_color,
                    "final_board": str(game.board),
                }
            )

        except Exception as e:
            logging.error(f"Error in evaluation game {game_id}: {str(e)}")
            logging.error(traceback.format_exc())
            result_queue.put(None)

    def evaluate_against_best(
        self, n_games: int = 40, max_workers: int = 10
    ) -> Optional[Tuple[float, int, int, int]]:
        """Evaluate current model against best model and return score counts."""
        self.logger.info("\nEvaluating against best model...")

        if self.best_model is None:
            return None

        wins = 0
        losses = 0
        draws = 0
        total_games = 0

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        active_workers = []
        game_id = 0
        games_completed = 0

        try:
            for color in [0, 1]:
                games_per_color = n_games // 2
                while games_completed < total_games + games_per_color:
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
                                max_workers,
                            ),
                        )
                        p.start()
                        active_workers.append((p, game_id))
                        game_id += 1

                    try:
                        result = result_queue.get(timeout=1)
                        if result is not None:
                            games_completed += 1
                            game_result = result["result"]

                            if game_result == 1:
                                wins += 1
                            elif game_result == -1:
                                losses += 1
                            else:
                                draws += 1

                            self.logger.info(
                                f"Evaluation game {result['game_id']} completed with result for {result['perspective_color']}: {game_result} and board:\n{result['final_board']}"
                            )

                    except Empty:
                        pass

                    active_workers = [
                        (p, gid) for p, gid in active_workers if p.is_alive()
                    ]

                game_id = 0
                total_games = games_completed

        finally:
            for p, _ in active_workers:
                if p.is_alive():
                    p.terminate()
                    p.join()

        total_games = wins + losses + draws
        win_rate = wins / (total_games - draws) if (total_games - draws) > 0 else 0

        return win_rate, wins, losses, draws
