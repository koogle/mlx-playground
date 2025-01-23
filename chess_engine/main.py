from game import ChessGame


def main():
    game = ChessGame()
    print("Welcome to Chess!")
    print(game)

    while True:
        try:
            print(
                f"\nCurrent turn: {'White' if game.get_current_turn().name == 'WHITE' else 'Black'}"
            )
            from_pos = input("Enter the position of the piece to move (e.g., e2): ")
            if from_pos.lower() == "quit":
                break
            to_pos = input("Enter the destination position (e.g., e4): ")

            if game.make_move(from_pos, to_pos):
                print("\nMove successful!")
                print(game)
            else:
                print("\nInvalid move! Try again.")

        except (IndexError, ValueError) as e:
            print("\nInvalid input! Please use the format 'e2' for positions.")


if __name__ == "__main__":
    main()
