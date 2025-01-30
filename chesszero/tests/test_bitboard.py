import unittest
from chess_engine.bitboard import BitBoard


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = BitBoard()

    def test_initial_position(self):
        # Test white pieces
        assert self.board.get_piece_at(0, 0) == (0, 3)  # White rook
        assert self.board.get_piece_at(0, 1) == (0, 1)  # White knight
        assert self.board.get_piece_at(0, 2) == (0, 2)  # White bishop
        assert self.board.get_piece_at(0, 3) == (0, 4)  # White queen
        assert self.board.get_piece_at(0, 4) == (0, 5)  # White king

        # Test black pieces
        assert self.board.get_piece_at(7, 0) == (1, 3)  # Black rook
        assert self.board.get_piece_at(7, 1) == (1, 1)  # Black knight
        assert self.board.get_piece_at(7, 2) == (1, 2)  # Black bishop
        assert self.board.get_piece_at(7, 3) == (1, 4)  # Black queen
        assert self.board.get_piece_at(7, 4) == (1, 5)  # Black king

    def test_pawn_moves(self):
        # Test white pawn initial moves
        moves = self.board.get_valid_moves((1, 0))
        assert (2, 0) in moves  # Single push
        assert (3, 0) in moves  # Double push
        assert len(moves) == 2

        # Test black pawn initial moves
        self.board.state[12] = 0  # Set black to move
        moves = self.board.get_valid_moves((6, 0))
        assert (5, 0) in moves  # Single push
        assert (4, 0) in moves  # Double push
        assert len(moves) == 2

    def test_knight_moves(self):
        # Test white knight moves
        moves = self.board.get_valid_moves((0, 1))
        assert (2, 0) in moves
        assert (2, 2) in moves
        assert len(moves) == 2  # Only these moves are possible initially

    def test_check_detection(self):
        """Test check detection with queen attacking king"""
        # Clear the pawns blocking the path
        self.board.state[0, 1, 5] = 0  # Remove white f2 pawn
        self.board.state[6, 6, 7] = 0  # Remove black h7 pawn

        # Now move white queen to attack black king
        self.board.make_move((0, 3), (4, 7))  # Moving from d1 to h5
        assert self.board.is_in_check(1)  # Black should be in check
        assert not self.board.is_in_check(0)  # White should not be in check

    def test_castling_rights(self):
        # Initially castling should be possible
        assert self.board.can_castle_kingside(0)
        assert self.board.can_castle_queenside(0)

        # Move king, should lose all castling rights
        self.board.make_move((0, 4), (0, 5))
        self.board.make_move((7, 0), (7, 1))  # Black move
        self.board.make_move((0, 5), (0, 4))  # Move king back

        assert not self.board.can_castle_kingside(0)
        assert not self.board.can_castle_queenside(0)

    def test_pinned_pieces(self):
        """Test pinned piece movement restrictions"""
        # Clear the board
        self.board.state.fill(0)

        # Setup: White king at e1, White bishop at f2, Black queen at h4
        # This creates a diagonal pin
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[2, 1, 5] = 1  # White bishop at f2
        self.board.state[10, 3, 7] = 1  # Black queen at h4
        self.board.state[12] = 1  # White to move

        # Get valid moves for the pinned bishop
        valid_moves = self.board.get_valid_moves((1, 5))

        # The bishop can only move along the diagonal between the king and queen
        expected_moves = {(2, 6), (3, 7)}  # Can move to block or capture
        assert valid_moves == expected_moves

    def test_checkmate_detection(self):
        """Test checkmate detection"""
        # Setup fool's mate position
        self.board.make_move((1, 5), (2, 5))  # f2 to f3
        self.board.make_move((6, 4), (4, 4))  # e7 to e5
        self.board.make_move((1, 6), (3, 6))  # g2 to g4
        self.board.make_move((7, 3), (3, 7))  # Qd8 to h4

        # Verify checkmate
        assert self.board.is_checkmate(0)  # White should be in checkmate
        assert not self.board.is_checkmate(1)  # Black should not be in checkmate

    def test_board_copy(self):
        """Test board copying maintains correct state"""
        # Clear the board first
        self.board.state.fill(0)

        # Add specific pieces to test
        self.board.state[0, 3, 4] = 1  # White pawn at e4
        self.board.state[6, 4, 4] = 1  # Black pawn at e5
        self.board.state[12] = 1  # White to move

        # Copy board and verify state
        copied_board = self.board.copy()

        # Test piece positions match
        assert copied_board.get_piece_at(3, 4) == (0, 0)  # White pawn
        assert copied_board.get_piece_at(4, 4) == (1, 0)  # Black pawn
        assert copied_board.get_current_turn() == 0  # White to move

    def test_piece_list_consistency(self):
        """Test that piece lists stay consistent with board state"""
        # Make a move
        from_pos = (1, 4)  # e2
        to_pos = (3, 4)  # e4
        self.board.make_move(from_pos, to_pos)

        # Get all pieces from the board state
        white_pieces = self.board.get_all_pieces(0)
        black_pieces = self.board.get_all_pieces(1)

        # Verify piece counts
        assert len(white_pieces) == 16
        assert len(black_pieces) == 16

        # Verify the moved pawn is in the correct position
        assert (to_pos, 0) in white_pieces  # Pawn at e4

    def test_blocked_pawn(self):
        """Test blocked pawn movement"""
        # Block a white pawn at a2 (1,0) with a black piece at a3 (2,0)
        self.board.state[6, 2, 0] = (
            1  # Put a black pawn at a3 to block white pawn at a2
        )
        moves = self.board.get_valid_moves((1, 0))  # Get moves for white pawn at a2
        assert len(moves) == 0  # Pawn should have no valid moves

    def test_pawn_captures(self):
        """Test pawn capture moves"""
        # Place a black piece for white pawn to capture
        self.board.state[6, 2, 1] = 1  # Black pawn at diagonal
        moves = self.board.get_valid_moves((1, 0))
        assert (2, 1) in moves  # Should be able to capture diagonally
