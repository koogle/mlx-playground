from game import ChessGame


def main():
    game = ChessGame()
    print("Welcome to Chess!")
    print(game)
    print("\nCommands:")
    print("- Standard algebraic notation: e4, Nf3, exd5, O-O")
    print("- 'history' to show move history")
    print("- 'quit' to exit")

    while True:
        try:
            print(
                f"\nCurrent turn: {'White' if game.get_current_turn().name == 'WHITE' else 'Black'}"
            )
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
