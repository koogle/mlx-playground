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
        # Clear the board first
        self.board.state.fill(0)

        # Set up a simple position:
        # White queen at e4 attacking black king at e8
        self.board.state[4, 3, 4] = 1  # White queen at e4
        self.board.state[5, 0, 4] = 1  # White king at e1 (needed!)
        self.board.state[11, 7, 4] = 1  # Black king at e8
        self.board.state[12] = 1  # White to move

        # Verify check detection
        assert self.board.is_in_check(
            1
        )  # Black should be in check (queen attacks on e-file)
        assert not self.board.is_in_check(0)  # White should not be in check

    def test_castling_rights(self):
        """Test castling rights with valid moves"""
        # Clear the pieces blocking castling
        self.board.state[2, 0, 5] = 0  # Clear bishop at f1
        self.board.state[1, 0, 6] = 0  # Clear knight at g1

        # Verify initial position
        assert self.board.state[5, 0, 4] == 1, "White king should be at e1"
        assert self.board.state[3, 0, 7] == 1, "White rook should be at h1"
        assert self.board.state[13, 0, 4] == 1, "White king should have castling rights"
        assert self.board.state[13, 0, 7] == 1, "White rook should have castling rights"

        # Check if squares between are empty
        assert all(
            self.board.get_piece_at(0, col)[0] == -1 for col in range(5, 7)
        ), "Squares f1-g1 should be empty"

        # Check if path is not under attack
        assert not any(
            self.board.is_square_attacked((0, col), 1) for col in range(4, 7)
        ), "Castling path should not be under attack"

        # Now test castling
        assert self.board.can_castle_kingside(0)

    def test_pinned_pieces(self):
        """Test pinned piece movement restrictions"""
        # Clear the board
        self.board.state.fill(0)
        self.board.king_positions = {
            0: (0, 5),  # Move king to f1
            1: (7, 4),
        }

        # Setup: White king at f1, White queen at f2, Black rook at f8
        # This creates a vertical pin
        self.board.state[5, 0, 5] = 1  # White king at f1
        self.board.state[4, 1, 5] = 1  # White queen at f2
        self.board.state[9, 7, 5] = 1  # Black rook at f8
        self.board.state[12] = 1  # White to move

        # Get valid moves for the pinned queen
        valid_moves = self.board.get_valid_moves((1, 5))

        # The queen can move along the pin line (vertically), including capturing the rook
        expected_moves = {
            (2, 5),  # f3
            (3, 5),  # f4
            (4, 5),  # f5
            (5, 5),  # f6
            (6, 5),  # f7
            (7, 5),  # f8 - capturing the rook is legal and removes the pin
        }  # Only moves along the f-file
        assert (
            valid_moves == expected_moves
        ), f"Expected {expected_moves}, got {valid_moves}"

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
        """Test piece list consistency with valid pawn move"""
        # Clear the board first
        self.board.state.fill(0)

        # Set up minimal position
        self.board.state[0, 1, 4] = 1  # White pawn at e2
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[11, 7, 4] = 1  # Black king at e8
        self.board.state[12] = 1  # White to move

        # Make a valid pawn move
        from_pos = (1, 4)  # e2
        to_pos = (3, 4)  # e4
        self.board.make_move(from_pos, to_pos)

        # Get all pieces from the board state
        white_pieces = self.board.get_all_pieces(0)
        black_pieces = self.board.get_all_pieces(1)

        # Verify piece counts
        assert len(white_pieces) == 2  # King and pawn
        assert len(black_pieces) == 1  # Just king

        # Verify the moved pawn is in the correct position
        assert (to_pos, 0) in white_pieces  # Pawn at e4

    def test_blocked_pawn(self):
        """Test blocked pawn movement"""
        # Clear the board first
        self.board.state.fill(0)

        # Set up a simple blocked pawn position
        self.board.state[0, 1, 0] = 1  # White pawn at a2
        self.board.state[6, 2, 0] = 1  # Black pawn at a3
        self.board.state[12] = 1  # White to move

        moves = self.board.get_valid_moves((1, 0))  # Get moves for white pawn at a2
        assert len(moves) == 0  # Pawn should have no valid moves

    def test_pawn_captures(self):
        """Test pawn capture moves"""
        # Clear the board first
        self.board.state.fill(0)

        # Set up a simple pawn capture position with both kings
        self.board.state[0, 1, 0] = 1  # White pawn at a2
        self.board.state[6, 2, 1] = 1  # Black pawn at b3
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[11, 7, 4] = 1  # Black king at e8
        self.board.state[12] = 1  # White to move

        moves = self.board.get_valid_moves((1, 0))
        assert (2, 0) in moves  # Single push to a3
        assert (3, 0) in moves  # Double push to a4
        assert (2, 1) in moves  # Capture to b3
        assert len(moves) == 3  # All three moves should be possible

    def test_stalemate(self):
        """Test stalemate detection with classic stalemate position"""
        # Clear the board first
        self.board.state.fill(0)
        self.board.king_positions = {
            0: (0, 6),  # White king at g1
            1: (0, 0),  # Black king at a1
        }

        # Set up a classic stalemate position:
        # Black king at a1, White queen at c2, White king at g1
        self.board.state[11, 0, 0] = 1  # Black king at a1
        self.board.state[4, 1, 2] = 1  # White queen at c2
        self.board.state[5, 0, 6] = 1  # White king at g1
        self.board.state[12] = 0  # Black to move

        # Verify stalemate
        assert self.board.is_stalemate(1)  # Black should be in stalemate
        assert not self.board.is_in_check(1)  # Black should not be in check
        assert self.board.is_game_over()  # Game should be over

    def test_insufficient_material_draws(self):
        """Test various insufficient material scenarios"""
        # Clear the board first
        self.board.state.fill(0)

        # Test 1: King vs King
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[11, 7, 4] = 1  # Black king at e8
        assert self.board.is_draw()

        # Test 2: King + Knight vs King
        self.board.state[1, 0, 1] = 1  # Add white knight at b1
        assert self.board.is_draw()

        # Test 3: King + Bishop vs King
        self.board.state.fill(0)  # Clear board
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[11, 7, 4] = 1  # Black king at e8
        self.board.state[2, 0, 2] = 1  # Add white bishop at c1
        assert self.board.is_draw()

        # Test 4: King + 2 Knights vs King
        self.board.state.fill(0)  # Clear board
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[11, 7, 4] = 1  # Black king at e8
        self.board.state[1, 0, 1] = 1  # White knight at b1
        self.board.state[1, 0, 6] = 1  # White knight at g1
        assert self.board.is_draw()

        # Test 5: King + Bishop vs King + Bishop (same color squares)
        self.board.state.fill(0)  # Clear board
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[11, 7, 4] = 1  # Black king at e8
        self.board.state[2, 0, 2] = 1  # White bishop at c1 (light square)
        self.board.state[8, 7, 5] = 1  # Black bishop at f8 (light square)
        assert self.board.is_draw()

        # Test 6: King + Bishop vs King + Bishop (different color squares - not a draw)
        self.board.state.fill(0)  # Clear board
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[11, 7, 4] = 1  # Black king at e8
        self.board.state[2, 0, 2] = 1  # White bishop at c1 (light square)
        self.board.state[8, 7, 4] = 1  # Black bishop at e8 (dark square)
        assert not self.board.is_draw()

        # Test 7: King + Knight vs King + Knight
        self.board.state.fill(0)  # Clear board
        self.board.state[5, 0, 4] = 1  # White king at e1
        self.board.state[11, 7, 4] = 1  # Black king at e8
        self.board.state[1, 0, 1] = 1  # White knight at b1
        self.board.state[7, 7, 6] = 1  # Black knight at g8
        assert self.board.is_draw()
