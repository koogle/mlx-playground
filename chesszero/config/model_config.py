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
    n_simulations: int = 256  # Search budget per move; increase for longer runs
    c_puct: float = 1  # Increased exploration
    dirichlet_alpha: float = 0.55  # More noise
    dirichlet_epsilon: float = 0.3  # Stronger noise influence

    # Training
    batch_size: int = 1024  # 2048
    n_epochs: int = 1000
    learning_rate: float = 1e-3  # Conservative starting rate for Adam
    # Positions are kept across generations and replayed with many minibatch
    # steps per generation instead of one averaged update.
    replay_buffer_size: int = 50_000  # Most recent self-play positions retained
    training_passes_per_generation: float = 2.0  # Buffer passes per generation
    max_training_steps: int = 500  # Upper bound on optimizer steps per generation
    batchnorm_recalibration_batches: int = 16  # Refresh BN stats after training

    # Self-play
    n_games_per_iteration: int = 25  # Fewer games per iteration for more frequent evals
    temperature: float = 1.0  # Initial temperature for move selection
    temp_decay_steps: int = 30  # Slower decay
    temp_final: float = 0.5  # Maintain some exploration

    # Debug options
    debug: bool = False

    # Logging
    eval_interval_epochs: int = 5  # Run evaluation every 5 epochs
    acceptance_threshold: float = 0.55  # Score (draws = 0.5) to become best
