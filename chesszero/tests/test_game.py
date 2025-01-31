import unittest
from chess_engine.game import ChessGame


class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = ChessGame()

    def test_parse_move(self):
        """Test basic move parsing for initial position"""
        # Test pawn moves
        self.assertEqual(self.game.parse_move("e4"), ((1, 4), (3, 4)))  # e2-e4
        self.assertEqual(self.game.parse_move("d4"), ((1, 3), (3, 3)))  # d2-d4
        self.assertEqual(self.game.parse_move("a3"), ((1, 0), (2, 0)))  # a2-a3

        # Test knight moves
        self.assertEqual(self.game.parse_move("Nf3"), ((0, 6), (2, 5)))  # Ng1-f3
        self.assertEqual(self.game.parse_move("Nc3"), ((0, 1), (2, 2)))  # Nb1-c3

        # Test coordinate notation
        self.assertEqual(self.game.parse_move("e2e4"), ((1, 4), (3, 4)))
        self.assertEqual(self.game.parse_move("g1f3"), ((0, 6), (2, 5)))

        # Test castling notation (just format, not legality)
        self.assertEqual(
            self.game.parse_move("O-O"), ((0, 4), (0, 6))
        )  # White kingside
        self.assertEqual(
            self.game.parse_move("O-O-O"), ((0, 4), (0, 2))
        )  # White queenside

        # Test invalid moves
        self.assertEqual(self.game.parse_move(""), (None, None))
        self.assertEqual(self.game.parse_move("invalid"), (None, None))
        self.assertEqual(self.game.parse_move("e9"), (None, None))
        self.assertEqual(self.game.parse_move("i4"), (None, None))

    def test_parse_move_black(self):
        """Test move parsing from Black's perspective"""
        # Make a move to change turn to black
        self.game.make_move_algebraic("e4")

        # Test black pawn moves
        self.assertEqual(self.game.parse_move("e5"), ((6, 4), (4, 4)))  # e7-e5
        self.assertEqual(self.game.parse_move("d5"), ((6, 3), (4, 3)))  # d7-d5

        # Test black knight moves
        self.assertEqual(self.game.parse_move("Nf6"), ((7, 6), (5, 5)))  # Ng8-f6
        self.assertEqual(self.game.parse_move("Nc6"), ((7, 1), (5, 2)))  # Nb8-c6

    def test_parse_move_with_game_progress(self):
        """Test move parsing throughout a sequence of moves"""
        # Test opening sequence
        moves = [
            ("e4", ((1, 4), (3, 4))),  # White
            ("e5", ((6, 4), (4, 4))),  # Black
            ("Nf3", ((0, 6), (2, 5))),  # White
            ("Nc6", ((7, 1), (5, 2))),  # Black
        ]

        for move_str, expected in moves:
            self.assertEqual(self.game.parse_move(move_str), expected)
            self.assertTrue(self.game.make_move_algebraic(move_str))

    def test_parse_move_after_setup(self):
        """Test parsing moves in specific board positions"""
        # Set up a position with e4 e5
        self.game.make_move_algebraic("e4")
        self.game.make_move_algebraic("e5")

        # Now test specific moves in this position
        self.assertEqual(
            self.game.parse_move("Nf3"), ((0, 6), (2, 5))
        )  # White knight to f3
        self.assertTrue(self.game.make_move_algebraic("Nf3"))

        self.assertEqual(
            self.game.parse_move("Nc6"), ((7, 1), (5, 2))
        )  # Black knight to c6

    def test_castling_notation(self):
        """Test that castling notation is parsed correctly"""
        # Test castling notation (just format, not legality)
        self.assertEqual(
            self.game.parse_move("O-O"), ((0, 4), (0, 6))
        )  # White kingside
        self.assertEqual(
            self.game.parse_move("O-O-O"), ((0, 4), (0, 2))
        )  # White queenside

        # Invalid castling notation
        self.assertEqual(self.game.parse_move("0-0"), (None, None))
        self.assertEqual(self.game.parse_move("o-o"), (None, None))

    def test_invalid_moves(self):
        """Test handling of invalid moves"""
        invalid_moves = [
            # Basic invalid moves
            "e5",  # Can't move e pawn to e5 directly (needs e2e4 first)
            "Nf6",  # Black's move during White's turn
            "a9",  # Invalid rank
            "i4",  # Invalid file
            "Nx",  # Incomplete move
            "e2e5",  # Invalid pawn leap (can only move 2 squares to e4)
            # Blocked piece moves
            "e1e2",  # King blocked by own pawn
            "d1e2",  # Queen blocked by own pawn
            "f1e2",  # Bishop blocked by own pawn
            # Invalid piece moves
            "Na4",  # Knight can't reach a4 from starting position
            "Bg4",  # Bishop can't reach g4 from starting position
            "Ra4",  # Rook blocked by own pawn
            # Invalid pawn moves
            "e3e4",  # Can't move pawn from e3 (no pawn there)
            "a2a5",  # Pawn can't move 3 squares
            "h2h1",  # Pawn can't move backwards
        ]

        for move in invalid_moves:
            # Test that parse_move returns None
            result = self.game.parse_move(move)
            self.assertEqual(
                result,
                (None, None),
                f"Move {move} should be invalid but got {result}",
            )

            # Double check that make_move_algebraic also fails
            self.assertFalse(
                self.game.make_move_algebraic(move),
                f"Move {move} should be invalid but was accepted",
            )

    def test_invalid_moves_after_e4(self):
        """Test invalid moves in a position after 1.e4"""
        self.game.make_move_algebraic("e4")

        invalid_moves = [
            # Black invalid moves
            "e4",  # Square already occupied
            "e6e5",  # No pawn on e6
            "Nf3",  # White piece move during Black's turn
            "O-O",  # Can't castle with pieces in between
            "d7d3",  # Pawn can't move 4 squares
            "e7e6e5",  # Invalid move format
            "Nb8c7",  # Knight blocked by own pawn
        ]

        for move in invalid_moves:
            result = self.game.parse_move(move)
            self.assertEqual(
                result,
                (None, None),
                f"Move {move} should be invalid but got {result}",
            )
            self.assertFalse(
                self.game.make_move_algebraic(move),
                f"Move {move} should be invalid but was accepted",
            )


if __name__ == "__main__":
    unittest.main()
