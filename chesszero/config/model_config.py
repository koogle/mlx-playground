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
    input_shape: tuple = (8, 8, 119)  # Board size + input planes
    policy_output_dim: int = 4672  # All possible moves: 8x8x73

    # MCTS
    n_simulations: int = 1600  # Number of MCTS simulations per move
    c_puct: float = 1.0  # Exploration constant
    dirichlet_alpha: float = 0.3  # Dirichlet noise parameter
    dirichlet_epsilon: float = 0.25  # Weight of Dirichlet noise

    # Training
    """
    batch_size: int = 2048
    n_epochs: int = 500
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    """
    batch_size: int = 2048
    n_epochs: int = 500
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Self-play
    n_games_per_iteration: int = 1  # 5000
    temperature: float = 1.0  # Initial temperature for move selection
    temp_decay_steps: int = 10  # Number of moves before temperature decay
    temp_final: float = 0.1  # Final temperature after decay

    # Evaluation
    eval_games: int = 400
    winning_threshold: float = 0.55  # Win rate needed to update network

    # History
    n_history_moves: int = 8  # Number of previous board states to include

    # Debug options
    debug: bool = True  # Print debug information including tensor shapes
