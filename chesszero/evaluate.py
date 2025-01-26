from typing import Tuple
import numpy as np
from chess_engine.game import ChessGame
from utils.random_player import RandomPlayer
from model.mcts import MCTS
from config.model_config import ModelConfig


def play_game(mcts: MCTS, opponent, mcts_player_color) -> Tuple[int, int]:
    """Play a single game between MCTS and opponent

    Returns:
        Tuple[int, int]: (wins, games) - 1 for win, 0.5 for draw, 0 for loss
    """
    game = ChessGame()

    while True:
        state = game.get_game_state()
        if state != "Normal":
            if "Checkmate" in state:
                winner = "Black" if "White wins" in state else "White"
                if (winner == "White" and mcts_player_color == Color.WHITE) or (
                    winner == "Black" and mcts_player_color == Color.BLACK
                ):
                    return 1, 1
                return 0, 1
            return 0.5, 1  # Draw

        if game.get_current_turn() == mcts_player_color:
            # MCTS player's turn
            move = mcts.get_move(game.board)
        else:
            # Opponent's turn
            move = opponent.select_move(game.board)

        if not move:
            return 0.5, 1  # Draw - no valid moves

        game.make_move(move[0], move[1])


def evaluate_against_random(model, config: ModelConfig, n_games: int = 100) -> float:
    """Evaluate model against random player

    Returns:
        float: Win rate against random player
    """
    mcts = MCTS(model, config)
    random_player = RandomPlayer()

    total_wins = 0
    total_games = 0

    # Play as both white and black
    for color in [Color.WHITE, Color.BLACK]:
        for _ in range(n_games // 2):
            wins, games = play_game(mcts, random_player, color)
            total_wins += wins
            total_games += games

    return total_wins / total_games
