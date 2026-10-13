import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn.losses as mx_losses
import mlx.nn as nn
from typing import Tuple, Optional
from model.network import ChessNet
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
from model.game_workers import run_game_workers
from model.inference import BatchedInference
from model.evaluation import play_evaluation_game_worker
from mlx.utils import tree_flatten, tree_map


class IncompleteEvaluationError(RuntimeError):
    """Some evaluation games failed, so no promotion decision is available."""


class Trainer:
    """Handles training, self-play, and evaluation of the chess model"""

    def __init__(
        self,
        config: ModelConfig,
        checkpoint_dir: str = "checkpoints",
        resume_epoch: Optional[int] = None,
        resume_interrupted: bool = False,
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
        self.best_optimizer_state = None

        if resume_epoch is not None:
            self.logger.info(f"Resuming from epoch {resume_epoch}")
            self.model, state = ChessNet.load_checkpoint(
                checkpoint_dir, resume_epoch, interrupted=resume_interrupted
            )
            self.config = self.model.config
            self.start_epoch = state["completed_epochs"]
            self.optimizer = optim.Adam(learning_rate=self.config.learning_rate)
            if state.get("optimizer_state") is not None:
                self.optimizer.state = state["optimizer_state"]
            self.best_model = state.get("best_model")
            self.best_optimizer_state = state.get("best_optimizer_state")
            self.best_model_win_rate = state.get("best_model_win_rate") or 0.0
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
                self._release_unused_device_cache()

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

                    try:
                        result = self.evaluate_against_best(max_workers=n_workers)
                    except IncompleteEvaluationError as error:
                        self.logger.warning("Skipping model acceptance: %s", error)
                        self._save_checkpoint(epoch + 1)
                        continue
                    if result is None:
                        self._accept_current_model()
                    else:
                        score, wins, losses, draws = result
                        self.logger.info(
                            f"Evaluation score: {score:.2%}; W/D/L: {wins}/{draws}/{losses}"
                        )
                        if score > 0.55:
                            self._accept_current_model(score)
                        else:
                            self._revert_to_best()
                    self._save_checkpoint(epoch + 1)

            total_time = time.time() - start_time
            self.logger.info("\n=== Training Completed ===")
            self.logger.info(f"Total training time: {total_time/3600:.1f} hours")

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted. Saving final checkpoint...")
            self._save_checkpoint(epoch + 1, interrupted=True, completed_epochs=epoch)
            raise

    @staticmethod
    def _release_unused_device_cache():
        """Return idle training buffers before other processes use the GPU."""
        mx.clear_cache()

    def _accept_current_model(self, score=0.0):
        if self.best_model is None:
            self.logger.info("No best model yet, accepting current model as best")
        else:
            self.logger.info(
                "Accepting new best model with win rate %.2f%%", score * 100
            )
        self.best_model = ChessNet(self.config)
        self.best_model.load_weights(tree_flatten(self.model.parameters()))
        # Arrays are immutable; copy containers so future Adam updates cannot
        # overwrite the accepted model's moments/step.
        self.best_optimizer_state = tree_map(lambda x: x, self.optimizer.state)
        self.best_model_win_rate = score

    def _revert_to_best(self):
        self.logger.info("Reverting to best model and its optimizer state")
        self.model.load_weights(tree_flatten(self.best_model.parameters()))
        if self.best_optimizer_state is not None:
            self.optimizer.state = tree_map(lambda x: x, self.best_optimizer_state)
        else:
            self.optimizer = optim.Adam(learning_rate=self.optimizer.learning_rate)
            self.logger.info("No saved best optimizer state; initialized fresh Adam")

    def _save_checkpoint(self, epoch, interrupted=False, completed_epochs=None):
        kind = "interrupted" if interrupted else "regular"
        self.logger.info("Saving %s checkpoint for epoch %d...", kind, epoch)
        self.model.save_checkpoint(
            self.checkpoint_dir,
            epoch,
            optimizer_state=self.optimizer.state,
            interrupted=interrupted,
            best_model_win_rate=self.best_model_win_rate,
            best_model=self.best_model,
            best_optimizer_state=self.best_optimizer_state,
            completed_epochs=completed_epochs,
        )
        self.logger.info(
            "Saved %s checkpoint for epoch %d in %s", kind, epoch, self.checkpoint_dir
        )

    @staticmethod
    def _evaluation_score(wins, losses, draws):
        """Fraction of all evaluation games won; draws contribute no credit."""
        total = wins + losses + draws
        if total <= 0:
            raise ValueError("Evaluation requires completed games")
        return wins / total

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

    def evaluate_against_best(
        self, n_games: int = 40, max_workers: int = 10
    ) -> Optional[Tuple[float, int, int, int]]:
        """Evaluate current model against best model and return score counts."""
        self._release_unused_device_cache()
        self.logger.info("\nEvaluating against best model...")

        if self.best_model is None:
            return None

        if n_games < 2 or n_games % 2 or max_workers < 1:
            raise ValueError(
                "Evaluation needs a positive worker count and an even n_games >= 2"
            )

        jobs = [
            (self.config, game_id, color)
            for color in (0, 1)
            for game_id in range(n_games // 2)
        ]
        with BatchedInference(
            self.model, self.config, reference_model=self.best_model
        ) as inference:
            results, failures = run_game_workers(
                play_evaluation_game_worker,
                jobs,
                max_workers,
                clients=(inference.client(), inference.reference_client()),
            )
        self.logger.info("Evaluation: %d completed, %d failed", len(results), failures)
        if failures:
            raise IncompleteEvaluationError(f"{failures} of {n_games} games failed")
        wins = sum(result["result"] == 1 for result in results)
        losses = sum(result["result"] == -1 for result in results)
        draws = sum(result["result"] == 0 for result in results)
        return self._evaluation_score(wins, losses, draws), wins, losses, draws
