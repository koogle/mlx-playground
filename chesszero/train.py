import argparse
from config.model_config import ModelConfig
import logging
from pathlib import Path
from datetime import datetime
import cProfile


def setup_logging(checkpoint_dir: str):
    # Set up logging to file and console
    log_dir = Path(checkpoint_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def main():
    # Spawned CPU game workers re-import this entrypoint. Only the parent
    # may import MLX/Trainer, after the __main__ guard calls main().
    from model.training import Trainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, help="Resume from epoch number")
    parser.add_argument(
        "--resume-interrupted",
        action="store_true",
        help="Load the separate interrupted checkpoint for --resume",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory for checkpoints"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable cProfile profiling"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for game generation",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        help="Fresh games collected before one training update (defaults to at least --workers)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total completed epochs, including a resumed checkpoint",
    )
    parser.add_argument("--simulations", type=int, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, help="Training microbatch size")
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate")
    parser.add_argument(
        "--eval-interval", type=int, help="Evaluate against best every N epochs"
    )
    parser.add_argument(
        "--evaluation-games", type=int, help="Even number of games against best"
    )
    parser.add_argument(
        "--evaluate-random",
        type=int,
        help="After training, play this even number of games against random",
    )
    args = parser.parse_args()
    for name in (
        "epochs",
        "simulations",
        "batch_size",
        "learning_rate",
        "eval_interval",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("evaluation_games", "evaluate_random"):
        value = getattr(args, name)
        if value is not None and (value < 2 or value % 2):
            parser.error(f"--{name.replace('_', '-')} must be even and at least 2")
    if args.workers < 1 or (
        args.games_per_iteration is not None and args.games_per_iteration < 1
    ):
        parser.error("Worker and game counts must be positive")

    if args.resume_interrupted and args.resume is None:
        parser.error("--resume-interrupted requires --resume")

    # Setup logging
    logger = setup_logging(args.checkpoint_dir)

    # Initialize
    config = ModelConfig()
    trainer = Trainer(
        config,
        checkpoint_dir=args.checkpoint_dir,
        resume_epoch=args.resume,
        resume_interrupted=args.resume_interrupted,
    )

    trainer.config.n_games_per_iteration = args.games_per_iteration or max(
        trainer.config.n_games_per_iteration, args.workers
    )

    for argument, field in (
        ("epochs", "n_epochs"),
        ("simulations", "n_simulations"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("eval_interval", "eval_interval_epochs"),
        ("evaluation_games", "n_eval_games"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(trainer.config, field, value)
    if args.learning_rate is not None:
        trainer.optimizer.learning_rate = args.learning_rate

    import mlx.core as mx

    logger.info("Effective model config: %s", vars(trainer.config))
    logger.info("MLX device: %s", mx.default_device())

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        logger.info("Profiling enabled")

    try:
        # Start training with specified number of workers
        trainer.train(n_workers=args.workers)
        if args.evaluate_random is not None:
            score, wins, losses, draws = trainer.evaluate_against_random(
                args.evaluate_random, args.workers
            )
            logger.info(
                "Against random: score %.2f%%; W/D/L %d/%d/%d",
                score * 100,
                wins,
                draws,
                losses,
            )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted.")
        if args.profile:
            profiler.disable()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = Path(f"mcts_stats_{timestamp}.stats")
            profiler.dump_stats(str(stats_file))
            logger.info(f"Profile stats saved to {stats_file}")


if __name__ == "__main__":
    main()
