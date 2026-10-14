import numpy as np
from chess_engine.game import ChessGame
from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS, GameSearchCache
from model.move_encoding import encode_move
from typing import List, Tuple, TYPE_CHECKING
from tqdm import tqdm
import logging
from multiprocessing.connection import Connection
from model.game_workers import run_game_workers
from model.inference import BatchedInference, InferenceClient
import traceback

if TYPE_CHECKING:
    from model.network import ChessNet


def play_single_game(
    model_inference: InferenceClient,
    config: ModelConfig,
    game_id: int,
    result_connection: Connection,
    max_workers: int,
):
    """CPU worker that plays a complete game using the central inference client"""
    logger = logging.getLogger(__name__)
    game_cache = GameSearchCache(model_inference)
    try:
        game = ChessGame()
        game_history = []

        pbar = tqdm(
            desc=f"Game {game_id}",
            position=game_id % max_workers,
            leave=False,
            unit="moves",
            unit_scale=True,
        )

        mcts = MCTS(model_inference, config, cache=game_cache)
        while not game.board.is_game_over():
            temperature = (
                config.temperature
                if len(game_history) < config.temp_decay_steps
                else config.temp_final
            )
            move = mcts.get_move(game.board, temperature=temperature)
            if move is None:
                raise RuntimeError("Search returned no move for an ongoing game")

            policy = get_policy_distribution(mcts.root_node, config.policy_output_dim)
            state = np.array(game.board.state, dtype=np.float32)
            game_history.append((state, policy, None))
            if not game.make_move(move[0], move[1]):
                raise RuntimeError("Search returned an illegal move")

            mcts.advance_root(move, game.board)
            pbar.update(1)

        pbar.close()

        white_result = game.board.get_game_result(perspective_color=0)
        if white_result is None:
            raise RuntimeError("Self-play ended without a terminal result")
        if white_result == 0.0:
            black_result = 0.0
        else:
            black_result = -white_result

        result_connection.send(
            {
                "game_id": game_id,
                "history": game_history,
                "white_result": white_result,
                "black_result": black_result,
                "n_moves": len(game_history),
                "final_board": str(game.board),
            }
        )

    except Exception as e:
        logger.error(f"Error in game {game_id}: {str(e)}")
        logger.error(traceback.format_exc())
        result_connection.send(None)
    finally:
        game_cache.clear()
        model_inference.close()


def generate_games(
    model: "ChessNet", config: ModelConfig, max_workers: int = 512
) -> List[Tuple]:
    logger = logging.getLogger(__name__)
    if config.n_games_per_iteration < 1:
        raise ValueError("Games per iteration must be positive")
    jobs = [(config, game_id) for game_id in range(config.n_games_per_iteration)]
    with BatchedInference(model, config) as inference:
        results, failures = run_game_workers(
            play_single_game, jobs, max_workers, clients=(inference.client(),)
        )
    logger.info("Self-play generation: %d completed, %d failed", len(results), failures)
    if not results:
        raise RuntimeError("All self-play games failed; no training update can be made")
    for result in results:
        logger.info(
            "Game %d: %d moves, White result %s",
            result["game_id"],
            result["n_moves"],
            result["white_result"],
        )
    return [(result["history"], result["white_result"]) for result in results]


def create_batches(games, batch_size: int):
    """Create training batches from games
    Args:
        games: List of tuples (game_history, white_result), one per game.
            Value targets are converted to the side-to-move perspective of each state.
        batch_size: Size of training batches
    """
    import mlx.core as mx

    print(
        f"Creating batches from {len(games)} game instances with batch_size {batch_size}"
    )

    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    n_samples = sum(len(history) for history, _ in games)
    print(f"Total positions collected: {n_samples}")
    if not n_samples:
        return
    print(f"Creating batches of size {batch_size} from {n_samples} samples")

    # Upload only the next microbatch. A generation-wide CUDA allocation can
    # block at large game counts and retains data the current step does not use.
    states, policies, values = [], [], []
    for history, white_result in games:
        for state, policy, _ in history:
            states.append(state)
            policies.append(policy)
            # Plane 12 is 1 for White to move and 0 for Black to move.
            values.append(white_result if state[12, 0, 0] == 1 else -white_result)
            if len(states) == batch_size:
                yield (
                    mx.array(np.stack(states)),
                    mx.array(np.stack(policies)),
                    mx.array(values),
                )
                states, policies, values = [], [], []
    if states:
        yield (
            mx.array(np.stack(states)),
            mx.array(np.stack(policies)),
            mx.array(values),
        )


def get_policy_distribution(root_node, policy_output_dim: int):
    """Convert MCTS visit counts to policy distribution we are doing this in numpy"""
    policy = np.zeros(policy_output_dim, dtype=np.float32)

    for move, child in root_node.children.items():
        move_idx = encode_move(*move)
        policy[move_idx] = child.visit_count

    policy = policy / np.sum(policy) if np.sum(policy) > 0 else policy

    return policy


def encode_board(board: BitBoard) -> np.ndarray:
    return np.array(board.state, dtype=np.float32)
