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
    get_policy_distribution,
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
            self.optimizer = optim.Adam(
                learning_rate=state.get(
                    "optimizer_learning_rate", self.config.learning_rate
                )
            )
            if state.get("optimizer_state") is not None:
                self.optimizer.state = state["optimizer_state"]
            self.best_model = state.get("best_model")
            self.best_optimizer_state = state.get("best_optimizer_state")
            self.best_model_win_rate = state.get("best_model_win_rate") or 0.0
        else:
            self.model = ChessNet(config)
            self.start_epoch = 0
            self.optimizer = optim.Adam(learning_rate=config.learning_rate)

    def train(self, n_epochs: Optional[int] = None, n_workers: int = 5):
        """Main training loop with parallel game generation"""
        n_epochs = n_epochs or self.config.n_epochs
        start_time = time.time()
        eval_games = None

        try:
            for epoch in range(self.start_epoch, n_epochs):
                epoch_start_time = time.time()
                self.logger.info(f"Epoch {epoch + 1}/{n_epochs}")
                self.logger.info(f"Generating games using {n_workers} workers...")
                self.model.eval()

                games = generate_games(self.model, self.config, n_workers)

                if eval_games is not None:
                    self.logger.info(
                        f"Adding {len(eval_games)} evaluation games to training data"
                    )
                    games.extend(eval_games)
                    eval_games = None

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

                    result = self.evaluate_against_best()
                    if result is None:
                        self._accept_current_model()
                    else:
                        score, wins, losses, draws, eval_games = result
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

    def _accept_current_model(self, score=0.0):
        self.best_model = ChessNet(self.config)
        self.best_model.load_weights(tree_flatten(self.model.parameters()))
        # Arrays are immutable; copy containers so future Adam updates cannot
        # overwrite the accepted model's moments/step.
        self.best_optimizer_state = tree_map(lambda x: x, self.optimizer.state)
        self.best_model_win_rate = score

    def _revert_to_best(self):
        self.model.load_weights(tree_flatten(self.best_model.parameters()))
        if self.best_optimizer_state is not None:
            self.optimizer.state = tree_map(lambda x: x, self.best_optimizer_state)
        else:
            self.optimizer = optim.Adam(learning_rate=self.optimizer.learning_rate)

    def _save_checkpoint(self, epoch, interrupted=False, completed_epochs=None):
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

    @staticmethod
    def _evaluation_score(wins, losses, draws):
        total = wins + losses + draws
        if total <= 0:
            raise ValueError("Evaluation requires completed games")
        return (wins + 0.5 * draws) / total

    def _compute_gradients(self, batch):
        """Reuse a compiled gradient function while explicitly tracking model state."""
        self.model.train()
        if getattr(self, "_gradient_model", None) is not self.model:
            model = self.model

            def loss_fn(states, policies, values):
                policy_logits, pred_values = model.forward_logits(states)
                p_loss = mx_losses.cross_entropy(
                    logits=policy_logits, targets=policies, reduction="mean"
                )
                v_loss = mx_losses.mse_loss(
                    predictions=pred_values.reshape(-1),
                    targets=values,
                    reduction="mean",
                )
                l2_loss = sum(
                    1e-6 * mx.sum(mx.square(parameter))
                    for _, parameter in tree_flatten(model.trainable_parameters())
                    if len(parameter.shape) > 1
                )
                return p_loss + v_loss + l2_loss, (p_loss, v_loss, l2_loss)

            # Include mutable BatchNorm running statistics in the compiled state,
            # but differentiate only trainable parameters, not those statistics.
            self._loss_and_grad = mx.compile(
                nn.value_and_grad(model, loss_fn),
                inputs=model.state,
                outputs=model.state,
            )
            self._gradient_model = model
        return self._loss_and_grad(*batch)

    def train_on_batches(self, batches):
        """Average gradients over fresh positions and apply exactly one update.

        Learned parameters remain fixed across microbatches. BatchNorm still uses
        each microbatch's statistics; this does not emulate one giant BN batch.
        """
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
                (loss, components), grads = self._compute_gradients(batch)
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
        """Worker process that plays a single evaluation game and collects training data"""
        try:
            model.eval()
            best_model.eval()
            game = ChessGame()
            game_history = []

            pbar = tqdm(
                desc=f"Evaluation game {game_id}",
                position=game_id % max_workers,
                leave=False,
                unit="moves",
                unit_scale=True,
            )

            while not game.board.is_game_over():
                if game.get_current_turn() == player_color:
                    mcts = MCTS(model, config, exploration_noise=False)
                else:
                    mcts = MCTS(best_model, config, exploration_noise=False)

                move = mcts.get_move(game.board, temperature=0.0)
                if not move:
                    break

                policy = get_policy_distribution(
                    mcts.root_node, config.policy_output_dim
                )
                state = mx.array(game.board.state, dtype=mx.float32)
                game_history.append((state, policy, None))
                game.make_move(move[0], move[1])

                pbar.update(1)

            pbar.close()

            result_queue.put(
                {
                    "game_id": game_id,
                    "result": game.board.get_game_result(player_color),
                    "history": game_history,
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
    ) -> Tuple[float, int, int, int]:
        """Evaluate current model against best model and collect training data"""
        self.logger.info("\nEvaluating against best model...")

        if self.best_model is None:
            return None

        if n_games < 2 or n_games % 2 or max_workers < 1:
            raise ValueError(
                "Evaluation needs a positive worker count and an even n_games >= 2"
            )

        wins = 0
        losses = 0
        draws = 0
        evaluation_games_data = []
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

                            evaluation_games_data.append(
                                (
                                    result["history"],
                                    (
                                        game_result
                                        if result["perspective_color"] == 0
                                        else -game_result
                                    ),
                                )
                            )

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
        win_rate = self._evaluation_score(wins, losses, draws)

        return win_rate, wins, losses, draws, evaluation_games_data
