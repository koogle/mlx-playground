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
    parser.add_argument(
        "--history",
        type=str,
        help="Load and replay a game from a history file",
    )
    args = parser.parse_args()

    game = ChessGame()

    print("Welcome to Chess!")
    if args.mode != "auto":
        print("\nCommands:")
        print("- Standard algebraic notation: e4, Nf3, exd5, O-O")
        print("- 'history' to show move history")
        print("- 'quit' to exit")

    # Load game history if provided
    if args.history:
        try:
            with open(args.history, "r") as f:
                history = f.read()
            print("\nLoading game history...")
            if game.load_game_history(history):
                print("\nGame loaded successfully!")
                print("\nCurrent position:")
                print(game)
            else:
                print("Failed to load game history")
                return
        except FileNotFoundError:
            print(f"History file not found: {args.history}")
            return
    else:
        print(game)  # Print initial board state

    # If playing against AI, randomly assign colors
    ai_color = None
    if args.mode == "ai":
        ai_color = random.choice([0, 1])
        print(f"\nYou are playing as {'Black' if ai_color == 0 else 'White'}")

    while not game.is_game_over():
        print(game)  # Show the board and current turn

        current_color = game.get_current_turn()

        # AI's turn (either in ai mode or auto mode)
        if args.mode == "auto" or (args.mode == "ai" and current_color == ai_color):
            if not handle_ai_turn(game):
                break
            if args.mode == "auto":
                time.sleep(args.delay)
        else:
            # Human's turn
            if not handle_human_turn(game):
                print("\nGame aborted.")
                break

        # if game.move_history:
        #    print("\nLast move:", game.move_history[-1])

    print("\nGame Over!")
    print(game.get_game_state())
    print("\nFinal game history:")
    print_move_history(game)


def handle_ai_turn(game: ChessGame) -> bool:
    """Handle AI move selection and execution"""
    # Get all pieces of current color
    current_color = game.get_current_turn()
    pieces = game.board.get_all_pieces(current_color)

    # Collect all valid moves
    valid_moves = []
    for pos, _ in pieces:
        moves = game.board.get_valid_moves(pos)
        for move in moves:
            valid_moves.append((pos, move))

    if not valid_moves:
        return False

    # Choose and make a random move
    from_pos, to_pos = random.choice(valid_moves)
    move_str = game._coords_to_algebraic(from_pos, to_pos)
    print(f"AI plays: {move_str}")

    return game.make_move(from_pos, to_pos)


def handle_human_turn(game: ChessGame) -> bool:
    """Handle a human player's turn"""
    while True:
        try:
            move = input("Enter move: ").strip()
            if move.lower() == "quit":
                return False

            # Use make_move_algebraic instead of make_move
            if game.make_move_algebraic(move):
                return True
            else:
                print("Invalid move, try again")
        except (ValueError, IndexError) as e:
            print(f"Invalid move format: {e}")
            print("Please use formats like: e2e4, e4, Nf3, or O-O")


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
