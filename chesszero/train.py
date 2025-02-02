import argparse
from config.model_config import ModelConfig
from model.training import Trainer
import logging
from pathlib import Path
from datetime import datetime


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, help="Resume from epoch number")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory for checkpoints"
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.checkpoint_dir)

    # Initialize
    config = ModelConfig()
    trainer = Trainer(
        config, checkpoint_dir=args.checkpoint_dir, resume_epoch=args.resume
    )

    try:
        # Start training
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted.")


if __name__ == "__main__":
    main()
