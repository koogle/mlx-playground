from config.model_config import ModelConfig
from model.training import Trainer


def main():
    # Initialize
    config = ModelConfig()
    trainer = Trainer(config, start_with_random=True)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
