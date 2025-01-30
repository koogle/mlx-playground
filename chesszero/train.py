import argparse
from config.model_config import ModelConfig
from model.training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, help="Resume from epoch number")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory for checkpoints"
    )
    args = parser.parse_args()

    # Initialize
    config = ModelConfig()
    trainer = Trainer(
        config, checkpoint_dir=args.checkpoint_dir, resume_epoch=args.resume
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
