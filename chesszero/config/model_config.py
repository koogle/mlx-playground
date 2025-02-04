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
    policy_output_dim: int = 4672  # All possible moves: 8x8x73

    # MCTS
    n_simulations: int = 1600  # Reduced number of simulations for faster training
    c_puct: float = 1.0  # Exploration constant
    dirichlet_alpha: float = 0.3  # Dirichlet noise parameter
    dirichlet_epsilon: float = 0.25  # Weight of Dirichlet noise

    # Training
    batch_size: int = 1024  # 2048
    n_epochs: int = 1000
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Self-play
    n_games_per_iteration: int = 25  # Fewer games per iteration for more frequent evals
    temperature: float = 1.0  # Initial temperature for move selection
    temp_decay_steps: int = 10  # Number of moves before temperature decay
    temp_final: float = 0.1  # Final temperature after decay

    # Debug options
    debug: bool = True

    # Logging
    eval_interval_epochs: int = 1  # Run evaluation every 5 epochs
    display_eval_game: bool = True  # Show board positions during evaluation
