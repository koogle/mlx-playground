from board import Color
from game import ChessGame
import random
import argparse
import time


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Chess game with optional AI opponent")
    parser.add_argument(
        "--mode",
        choices=["human", "ai", "auto"],
        default="human",
        help="Game mode: human vs human, human vs AI, or AI vs AI",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between moves in auto mode (seconds)",
    )
    args = parser.parse_args()

    game = ChessGame()

    # If playing against AI, randomly assign colors
    ai_color = None
    if args.mode == "ai":
        ai_color = random.choice([Color.WHITE, Color.BLACK])
        print(f"\nYou are playing as {'Black' if ai_color == Color.WHITE else 'White'}")

    print("Welcome to Chess!")
    print(game)
    if args.mode != "auto":
        print("\nCommands:")
        print("- Standard algebraic notation: e4, Nf3, exd5, O-O")
        print("- 'history' to show move history")
        print("- 'quit' to exit")

    while True:
        try:
            game_state = game.get_game_state()
            if "Checkmate" in game_state:
                print(f"\n{game_state}")
                print("\nFinal game history:")
                print_move_history(game)
                break

            current_color = game.get_current_turn()
            print(
                f"\n{'White' if current_color == Color.WHITE else 'Black'}: ",
                end="",
            )
            if "Check" in game_state:
                print(game_state)

            # AI's turn (either in ai mode or auto mode)
            if args.mode == "auto" or (args.mode == "ai" and current_color == ai_color):
                if not handle_ai_turn(game):
                    print("\nFinal game history:")
                    print_move_history(game)
                    break
                if args.mode == "auto":
                    time.sleep(args.delay)
                continue

            # Human's turn
            if not handle_human_turn(game):
                print("\nGame aborted. Final game history:")
                print_move_history(game)
                break

        except (IndexError, ValueError):
            print(
                "\nInvalid input! Please use standard algebraic notation (e.g., e4, Nf3, exd5)"
            )


def handle_ai_turn(game):
    valid_moves = game.get_all_valid_moves()
    if not valid_moves:
        print("No valid moves available!")
        return False

    ai_move = random.choice(valid_moves)
    print(f"{ai_move}")

    if game.make_move(ai_move):
        print(game)
    return True


def handle_human_turn(game):
    move = input("Enter move: ").strip()

    if move.lower() == "quit":
        return False
    elif move.lower() == "history":
        print_move_history(game)
        return True

    if game.make_move(move):
        print(game)
        if "x" in move:
            print("A piece was captured!")
    else:
        print("\nInvalid move! Try again.")
    return True


def print_move_history(game):
    if game.move_history:
        print("\nMove history:")
        for i, move in enumerate(game.move_history, 1):
            if i % 2 == 1:
                print(f"{(i+1)//2}. {move}", end="")
            else:
                print(f" {move}")
        if len(game.move_history) % 2 == 1:
            print()
    else:
        print("\nNo moves played yet.")


if __name__ == "__main__":
    main()
