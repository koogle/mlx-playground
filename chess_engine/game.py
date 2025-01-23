from board import Board, Color, Piece, PieceType
from typing import Tuple, List


class ChessGame:
    def __init__(self):
        self.board = Board()
        self.current_turn = Color.WHITE
        self.move_history: List[str] = []

    def _get_piece_symbol(self, piece: Piece) -> str:
        """Get the algebraic notation symbol for a piece."""
        if piece.piece_type == PieceType.PAWN:
            return ""
        piece_symbols = {
            PieceType.KING: "K",
            PieceType.QUEEN: "Q",
            PieceType.ROOK: "R",
            PieceType.BISHOP: "B",
            PieceType.KNIGHT: "N",
        }
        return piece_symbols[piece.piece_type]

    def _find_similar_pieces(
        self, piece_type: PieceType, target_square: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Find all pieces of the same type that could move to the target square."""
        similar_pieces = []
        for row in range(8):
            for col in range(8):
                piece = self.board.squares[row][col]
                if (
                    piece
                    and piece.piece_type == piece_type
                    and piece.color == self.current_turn
                ):
                    # Check if the piece can move to the target square
                    if self._can_move(piece, (row, col), target_square):
                        similar_pieces.append((row, col))
        return similar_pieces

    def _can_move(
        self, piece: Piece, from_square: Tuple[int, int], to_square: Tuple[int, int]
    ) -> bool:
        """Check if a piece can move from from_square to to_square."""
        from_row, from_col = from_square
        to_row, to_col = to_square

        if piece.piece_type == PieceType.PAWN:
            # Pawns move differently based on color
            direction = 1 if piece.color == Color.WHITE else -1
            start_row = 1 if piece.color == Color.WHITE else 6

            # Check for standard pawn move (must be same column and target square empty)
            if from_col == to_col and self.board.squares[to_row][to_col] is None:
                # Can only move forward one square
                if to_row == from_row + direction:
                    return True
                # Initial double move (must be on start row and path must be clear)
                if (
                    from_row == start_row
                    and to_row == from_row + 2 * direction
                    and self.board.squares[from_row + direction][from_col] is None
                ):
                    return True
                return False  # Any other forward move is invalid

            # Check for capture (must be diagonal and target square must have opponent's piece)
            if abs(from_col - to_col) == 1 and to_row == from_row + direction:
                target_piece = self.board.squares[to_row][to_col]
                return target_piece is not None and target_piece.color != piece.color

            return False  # Any other move is invalid

        # TODO: Add move validation for other piece types
        return True

    def make_move(self, move: str) -> bool:
        """
        Make a move using standard algebraic notation (e.g., 'e4', 'Nf3', 'exd5', 'O-O')
        Returns True if the move is valid and was executed
        """
        move = move.strip()

        # Handle castling
        if move in ["O-O", "0-0"]:
            # TODO: Implement kingside castling
            self.move_history.append(move)
            return True
        elif move in ["O-O-O", "0-0-0"]:
            # TODO: Implement queenside castling
            self.move_history.append(move)
            return True

        # Parse the move
        is_capture = "x" in move
        is_check = "+" in move or "#" in move
        is_promotion = "=" in move

        # Remove check/mate symbols for processing
        move = move.rstrip("+#")

        # Handle promotion
        promotion_piece = None
        if is_promotion:
            move, promotion_piece = move.split("=")

        # Get destination square
        dest_square = move[-2:]
        to_col = ord(dest_square[0]) - ord("a")
        to_row = int(dest_square[1]) - 1

        # Validate destination square
        if not (0 <= to_col < 8 and 0 <= to_row < 8):
            return False

        # Determine piece type
        piece_type = (
            PieceType.PAWN
            if move[0].islower()
            else {
                "K": PieceType.KING,
                "Q": PieceType.QUEEN,
                "R": PieceType.ROOK,
                "B": PieceType.BISHOP,
                "N": PieceType.KNIGHT,
            }[move[0]]
        )

        # Find all pieces of this type that could move to the destination
        similar_pieces = self._find_similar_pieces(piece_type, (to_row, to_col))
        if not similar_pieces:
            return False

        # TODO: Use disambiguation information if provided in the move

        # TODO: Implement proper move validation
        # For now, just move the piece if found
        if similar_pieces:
            from_row, from_col = similar_pieces[0]
            piece = self.board.squares[from_row][from_col]
            captured_piece = self.board.squares[to_row][to_col]

            # Execute move
            self.board.squares[to_row][to_col] = piece
            self.board.squares[from_row][from_col] = None

            # Record move in algebraic notation
            move_str = move
            if is_check:
                move_str += "+" if "+" in move else "#"
            self.move_history.append(move_str)

            # Switch turns
            self.current_turn = (
                Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
            )
            return True

        return False

    def get_current_turn(self) -> Color:
        return self.current_turn

    def __str__(self):
        return str(self.board)
