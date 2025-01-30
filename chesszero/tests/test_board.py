import unittest
from chess_engine.board import Board, Color, Piece, PieceType


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_initial_position(self):
        """Test initial board setup"""
        # Test white pieces
        self.assertEqual(len(self.board.white_pieces), 16)
        self.assertEqual(len(self.board.black_pieces), 16)

        # Test pawns
        for col in range(8):
            self.assertTrue(isinstance(self.board.squares[1][col], Piece))
            self.assertEqual(self.board.squares[1][col].piece_type, PieceType.PAWN)
            self.assertEqual(self.board.squares[1][col].color, Color.WHITE)

            self.assertTrue(isinstance(self.board.squares[6][col], Piece))
            self.assertEqual(self.board.squares[6][col].piece_type, PieceType.PAWN)
            self.assertEqual(self.board.squares[6][col].color, Color.BLACK)

    def test_pawn_moves(self):
        """Test pawn movement rules"""
        # Test initial two-square move
        valid_moves = self.board.get_valid_moves((1, 0))  # White pawn at a2
        self.assertIn((2, 0), valid_moves)  # One square
        self.assertIn((3, 0), valid_moves)  # Two squares

        # Test blocked pawn
        self.board.squares[2][0] = Piece(PieceType.PAWN, Color.BLACK)
        valid_moves = self.board.get_valid_moves((1, 0))
        self.assertEqual(len(valid_moves), 0)

        # Test diagonal capture
        self.board.squares[2][1] = Piece(PieceType.PAWN, Color.BLACK)
        valid_moves = self.board.get_valid_moves((1, 0))
        self.assertIn((2, 1), valid_moves)  # Diagonal capture

    def test_knight_moves(self):
        """Test knight movement"""
        # Clear the board
        self.board = Board()
        self.board.squares = [[None for _ in range(8)] for _ in range(8)]

        # Place a knight in the center
        knight = Piece(PieceType.KNIGHT, Color.WHITE)
        self.board.squares[3][3] = knight
        self.board.white_pieces = [(knight, (3, 3))]

        valid_moves = self.board.get_valid_moves((3, 3))
        expected_moves = {
            (1, 2),
            (1, 4),  # Up 2, left/right 1
            (5, 2),
            (5, 4),  # Down 2, left/right 1
            (2, 1),
            (4, 1),  # Left 2, up/down 1
            (2, 5),
            (4, 5),  # Right 2, up/down 1
        }
        self.assertEqual(valid_moves, expected_moves)

    def test_king_in_check(self):
        """Test check detection and valid moves in check"""
        # Clear the board
        self.board = Board()
        self.board.squares = [[None for _ in range(8)] for _ in range(8)]

        # Setup: White king at e1, Black rook at e8
        white_king = Piece(PieceType.KING, Color.WHITE)
        black_rook = Piece(PieceType.ROOK, Color.BLACK)

        self.board.squares[0][4] = white_king  # e1
        self.board.squares[7][4] = black_rook  # e8

        self.board.white_pieces = [(white_king, (0, 4))]
        self.board.black_pieces = [(black_rook, (7, 4))]
        self.board.current_turn = Color.WHITE

        # Verify king is in check
        self.assertTrue(self.board.is_in_check(Color.WHITE))

        # Verify valid moves (king must move out of check)
        valid_moves = self.board.get_valid_moves((0, 4))
        expected_moves = {
            (0, 3),
            (0, 5),
            (1, 3),
            (1, 5),
        }  # Removed (1, 4) as it's still in check
        self.assertEqual(valid_moves, expected_moves)

    def test_pinned_pieces(self):
        """Test pinned piece movement restrictions"""
        # Clear the board
        self.board = Board()
        self.board.squares = [[None for _ in range(8)] for _ in range(8)]

        # Setup: White king at e1, White bishop at e2, Black queen at e7
        white_king = Piece(PieceType.KING, Color.WHITE)
        white_bishop = Piece(PieceType.BISHOP, Color.WHITE)
        black_queen = Piece(PieceType.QUEEN, Color.BLACK)

        self.board.squares[0][4] = white_king  # e1
        self.board.squares[1][4] = white_bishop  # e2
        self.board.squares[6][4] = black_queen  # e7

        self.board.white_pieces = [(white_king, (0, 4)), (white_bishop, (1, 4))]
        self.board.black_pieces = [(black_queen, (6, 4))]
        self.board.current_turn = Color.WHITE

        # Bishop should only be able to move to capture the queen
        valid_moves = self.board.get_valid_moves((1, 4))
        expected_moves = {(6, 4)}  # Can only move to capture the queen
        self.assertEqual(valid_moves, expected_moves)

    def test_castling(self):
        """Test castling rules"""
        # Clear the board
        self.board = Board()
        self.board.squares = [[None for _ in range(8)] for _ in range(8)]

        # Setup: Initial position for castling
        white_king = Piece(PieceType.KING, Color.WHITE)
        white_rook1 = Piece(PieceType.ROOK, Color.WHITE)
        white_rook2 = Piece(PieceType.ROOK, Color.WHITE)

        self.board.squares[0][4] = white_king  # e1
        self.board.squares[0][0] = white_rook1  # a1
        self.board.squares[0][7] = white_rook2  # h1

        self.board.white_pieces = [
            (white_king, (0, 4)),
            (white_rook1, (0, 0)),
            (white_rook2, (0, 7)),
        ]

        # Test both castling moves are valid
        valid_moves = self.board.get_valid_moves((0, 4))
        self.assertIn((0, 2), valid_moves)  # Queenside castling
        self.assertIn((0, 6), valid_moves)  # Kingside castling

        # Test castling is prevented after king moves
        white_king.has_moved = True
        valid_moves = self.board.get_valid_moves((0, 4))
        self.assertNotIn((0, 2), valid_moves)
        self.assertNotIn((0, 6), valid_moves)

    def test_board_copy(self):
        """Test board copying maintains correct state"""
        # Clear the board first to have a controlled test
        self.board = Board()
        self.board.squares = [[None for _ in range(8)] for _ in range(8)]

        # Add specific pieces to test
        white_pawn = Piece(PieceType.PAWN, Color.WHITE)
        black_pawn = Piece(PieceType.PAWN, Color.BLACK)

        # Place pieces
        self.board.squares[3][4] = white_pawn  # e4
        self.board.squares[4][4] = black_pawn  # e5

        self.board.white_pieces = [(white_pawn, (3, 4))]
        self.board.black_pieces = [(black_pawn, (4, 4))]
        self.board.current_turn = Color.WHITE

        # Copy board and verify state
        copied_board = self.board.copy()

        # Test piece positions
        self.assertEqual(len(copied_board.white_pieces), 1)
        self.assertEqual(len(copied_board.black_pieces), 1)

        # Test piece positions match
        self.assertEqual(copied_board.squares[3][4].piece_type, PieceType.PAWN)
        self.assertEqual(copied_board.squares[3][4].color, Color.WHITE)
        self.assertEqual(copied_board.squares[4][4].piece_type, PieceType.PAWN)
        self.assertEqual(copied_board.squares[4][4].color, Color.BLACK)

    def test_checkmate_detection(self):
        """Test checkmate detection"""
        # Setup fool's mate position
        self.board.move_piece((1, 5), (2, 5))  # f2 to f3
        self.board.move_piece((6, 4), (4, 4))  # e7 to e5
        self.board.move_piece((1, 6), (3, 6))  # g2 to g4
        self.board.move_piece((7, 3), (3, 7))  # Qd8 to h4

        # Verify checkmate
        self.assertTrue(self.board.is_checkmate(Color.WHITE))
        self.assertFalse(self.board.is_checkmate(Color.BLACK))

    def test_piece_list_consistency(self):
        """Test that piece lists stay consistent with board state"""
        self.board = Board()

        # Make a move
        from_pos = (1, 4)  # e2
        to_pos = (3, 4)  # e4
        self.board.move_piece(from_pos, to_pos)

        # Verify piece lists match board state
        white_pieces_on_board = []
        black_pieces_on_board = []

        for row in range(8):
            for col in range(8):
                piece = self.board.squares[row][col]
                if piece:
                    if piece.color == Color.WHITE:
                        white_pieces_on_board.append((piece, (row, col)))
                    else:
                        black_pieces_on_board.append((piece, (row, col)))

        # Verify lists match
        self.assertEqual(
            set((p[1] for p in self.board.white_pieces)),
            set((p[1] for p in white_pieces_on_board)),
        )
        self.assertEqual(
            set((p[1] for p in self.board.black_pieces)),
            set((p[1] for p in black_pieces_on_board)),
        )


if __name__ == "__main__":
    unittest.main()
