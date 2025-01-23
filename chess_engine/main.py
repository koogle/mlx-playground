from game import ChessGame
import random
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Chess game with optional AI opponent")
    parser.add_argument("--ai", action="store_true", help="Play against AI")
    args = parser.parse_args()

    game = ChessGame()

    # If playing against AI, randomly assign colors
    ai_color = None
    if args.ai:
        ai_color = random.choice([Color.WHITE, Color.BLACK])
        print(f"\nYou are playing as {'Black' if ai_color == Color.WHITE else 'White'}")

    print("Welcome to Chess!")
    print(game)
    print("\nCommands:")
    print("- Standard algebraic notation: e4, Nf3, exd5, O-O")
    print("- 'history' to show move history")
    print("- 'quit' to exit")

    while True:
        try:
            game_state = game.get_game_state()
            if "Checkmate" in game_state:
                print(f"\n{game_state}")
                break

            print(
                f"\nCurrent turn: {'White' if game.get_current_turn().name == 'WHITE' else 'Black'}"
            )
            if "Check" in game_state:
                print(game_state)

            # AI's turn
            if args.ai and game.get_current_turn() == ai_color:
                valid_moves = game.get_all_valid_moves()
                if not valid_moves:
                    print("No valid moves available!")
                    break

                ai_move = random.choice(valid_moves)
                print(f"\nAI plays: {ai_move}")

                if game.make_move(ai_move):
                    print("\nMove successful!")
                    print(game)
                continue

            # Human's turn
            move = input("Enter move: ").strip()

            if move.lower() == "quit":
                break
            elif move.lower() == "history":
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
                continue

            if game.make_move(move):
                print("\nMove successful!")
                print(game)
            else:
                print("\nInvalid move! Try again.")

        except (IndexError, ValueError) as e:
            print(
                "\nInvalid input! Please use standard algebraic notation (e.g., e4, Nf3, exd5)"
            )


if __name__ == "__main__":
    main()
