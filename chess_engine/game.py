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
                    # TODO: Add proper move validation here
                    similar_pieces.append((row, col))
        return similar_pieces

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

        # Handle pawn moves
        if move[0].islower():
            # Pawn move or capture
            from_col = ord(move[0]) - ord("a")
            piece_type = PieceType.PAWN
            # TODO: Implement pawn move logic
        else:
            # Piece move
            piece_type = {
                "K": PieceType.KING,
                "Q": PieceType.QUEEN,
                "R": PieceType.ROOK,
                "B": PieceType.BISHOP,
                "N": PieceType.KNIGHT,
            }[move[0]]

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
