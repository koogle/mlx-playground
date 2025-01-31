import argparse
from config.model_config import ModelConfig
from model.training import Trainer
import cProfile
import pstats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, help="Resume from epoch number")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory for checkpoints"
    )
    args = parser.parse_args()

    # Run with profiler
    profiler = cProfile.Profile()
    profiler.enable()
    # Initialize
    config = ModelConfig()
    trainer = Trainer(
        config, checkpoint_dir=args.checkpoint_dir, resume_epoch=args.resume
    )

    try:
        # Start training
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving profiler stats...")
        profiler.disable()

        # Save stats
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.dump_stats("mcts_profile.prof")


if __name__ == "__main__":
    main()
