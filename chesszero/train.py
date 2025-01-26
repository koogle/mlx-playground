import mlx.core as mx
import mlx.optimizers as optim
from model.network import ChessNet
from config.model_config import ModelConfig
from evaluate import evaluate_against_random


def train():
    # Initialize
    config = ModelConfig()
    model = ChessNet(config)
    optimizer = optim.SGD(learning_rate=config.learning_rate)

    # Training loop
    for epoch in range(config.n_epochs):
        # Generate self-play games
        games = generate_games(model, config)

        # Train on game data
        for batch in create_batches(games, config.batch_size):
            states, policies, values = batch

            # Forward pass and loss calculation
            loss = model.loss_fn(states, policies, values)

            # Backward pass and optimization
            optimizer.step(loss)

        # Evaluate periodically
        if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
            win_rate = evaluate_against_random(model, config, n_games=100)
            print(f"Epoch {epoch + 1}: Win rate vs random: {win_rate:.2%}")

            # Early success check
            if epoch < 20 and win_rate > 0.8:
                print("Early success achieved!")


if __name__ == "__main__":
    train()
