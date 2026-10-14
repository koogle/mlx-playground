from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Training hyperparameters:
    - Network architecture details
    - MCTS simulation count
    - Training batch size, learning rate etc.
    - Self-play settings
    """

    # Network Architecture
    n_residual_blocks: int = 19
    n_filters: int = 256
    input_shape: tuple = (8, 8, 19)  # Board size + input planes (19 channels)
    policy_output_dim: int = 4672  # From/to moves + reserved underpromotion outputs

    # MCTS
    n_simulations: int = 1600  # Fewer but higher quality simulations
    c_puct: float = 1  # Increased exploration
    dirichlet_alpha: float = 0.55  # More noise
    dirichlet_epsilon: float = 0.3  # Stronger noise influence

    # Central inference batches requests already queued by parallel games.
    inference_batch_size: int = 64
    prediction_cache_size: int = 10_000

    # Training
    batch_size: int = 1024  # 2048
    n_epochs: int = 1000
    learning_rate: float = 1e-3  # Conservative starting rate for Adam

    # Self-play
    n_games_per_iteration: int = 512  # One fresh generation before each Adam update
    temperature: float = 1.0  # Initial temperature for move selection
    temp_decay_steps: int = 30  # Slower decay
    temp_final: float = 0.5  # Maintain some exploration

    # Debug options
    debug: bool = False

    # Logging
    eval_interval_epochs: int = 5  # Run evaluation every 5 epochs
