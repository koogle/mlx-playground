from board import Board, Color, Piece, PieceType
from typing import Tuple, List, Optional


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
        return self.board.is_valid_move(from_square, to_square)

    def make_move(self, move: str) -> bool:
        """Make a move using standard algebraic notation."""
        from_pos, to_pos = self._parse_move(move)
        if not from_pos or not to_pos:
            return False

        # Validate move
        piece = self.board.squares[from_pos[0]][from_pos[1]]
        if not piece or piece.color != self.current_turn:
            return False

        # Check if move is valid
        valid_moves = self.board.get_valid_moves(from_pos)
        if to_pos not in valid_moves:
            return False

        # Execute move
        self.board.move_piece(from_pos, to_pos)
        self.move_history.append(move)
        self.current_turn = (
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        )
        return True

    def _parse_move(
        self, move: str
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Parse a move in algebraic notation and return (from_pos, to_pos)."""
        move = move.strip()

        # Handle castling
        if move in ["O-O", "0-0"]:  # Kingside castling
            row = 0 if self.current_turn == Color.WHITE else 7
            return (row, 4), (row, 6)
        elif move in ["O-O-O", "0-0-0"]:  # Queenside castling
            row = 0 if self.current_turn == Color.WHITE else 7
            return (row, 4), (row, 2)

        # Remove capture and check symbols
        move = move.replace("x", "").rstrip("+#")

        # Get destination square
        dest_square = move[-2:]
        to_col = ord(dest_square[0]) - ord("a")
        to_row = int(dest_square[1]) - 1
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            return None, None
        to_pos = (to_row, to_col)

        # Handle pawn moves
        if len(move) == 2:  # Simple pawn move like "e4"
            # Find pawn that can move to this square
            direction = 1 if self.current_turn == Color.WHITE else -1
            possible_rows = [to_row - direction]
            if (self.current_turn == Color.WHITE and to_row == 3) or (
                self.current_turn == Color.BLACK and to_row == 4
            ):
                possible_rows.append(to_row - 2 * direction)

            for from_row in possible_rows:
                if 0 <= from_row < 8:
                    piece = self.board.squares[from_row][to_col]
                    if (
                        piece
                        and piece.piece_type == PieceType.PAWN
                        and piece.color == self.current_turn
                        and to_pos in self.board.get_valid_moves((from_row, to_col))
                    ):
                        return (from_row, to_col), to_pos
            return None, None

        elif len(move) == 3 and move[0] in "abcdefgh":  # Pawn capture like "exd5"
            from_col = ord(move[0]) - ord("a")
            if not (0 <= from_col < 8):
                return None, None

            # Find pawn that can capture to this square
            direction = 1 if self.current_turn == Color.WHITE else -1
            from_row = to_row - direction
            if 0 <= from_row < 8:
                piece = self.board.squares[from_row][from_col]
                if (
                    piece
                    and piece.piece_type == PieceType.PAWN
                    and piece.color == self.current_turn
                    and to_pos in self.board.get_valid_moves((from_row, from_col))
                ):
                    return (from_row, from_col), to_pos
            return None, None

        # Handle piece moves
        if move[0] in "KQRBN":
            piece_type = {
                "K": PieceType.KING,
                "Q": PieceType.QUEEN,
                "R": PieceType.ROOK,
                "B": PieceType.BISHOP,
                "N": PieceType.KNIGHT,
            }[move[0]]

            # Find all pieces of this type that can move to the target square
            pieces = self.board.get_pieces_by_type(self.current_turn, piece_type)
            for piece, from_pos in pieces:
                if to_pos in self.board.get_valid_moves(from_pos):
                    return from_pos, to_pos

        return None, None

    def get_current_turn(self) -> Color:
        return self.current_turn

    def get_game_state(self) -> str:
        """Get the current state of the game (check, checkmate, or normal)."""
        if not self.move_history:  # Game hasn't started yet
            return "Normal"

        if self.board.is_checkmate(self.current_turn):
            winner = "Black" if self.current_turn == Color.WHITE else "White"
            return f"Checkmate! {winner} wins!"
        elif self.board.is_in_check(self.current_turn):
            return "Check!"
        return "Normal"

    def __str__(self):
        return str(self.board)

    def get_all_valid_moves(self) -> List[str]:
        """Get all valid moves for the current player in algebraic notation."""
        valid_moves = []

        # Add castling moves if available
        if not self.board.is_in_check(self.current_turn):
            king_pieces = self.board.get_pieces_by_type(
                self.current_turn, PieceType.KING
            )
            rook_pieces = self.board.get_pieces_by_type(
                self.current_turn, PieceType.ROOK
            )
            if king_pieces and not king_pieces[0][0].has_moved:
                for rook, rook_pos in rook_pieces:
                    if not rook.has_moved:
                        if self._can_castle(king_pieces[0][1], rook_pos):
                            valid_moves.append("O-O" if rook_pos[1] == 7 else "O-O-O")

        # Get moves for all pieces
        pieces = self.board.get_pieces(self.current_turn)
        for piece, (from_row, from_col) in pieces:
            moves = self.board.get_piece_moves((from_row, from_col))
            for to_row, to_col in moves:
                move = self._format_move(piece, (from_row, from_col), (to_row, to_col))
                valid_moves.append(move)

        return valid_moves

    def _can_castle(self, king_pos: Tuple[int, int], rook_pos: Tuple[int, int]) -> bool:
        """Check if castling is possible for the given king and rook positions."""
        king_row, king_col = king_pos
        rook_row, rook_col = rook_pos

        # Check if king and rook are in their initial positions
        king = self.board.squares[king_row][king_col]
        rook = self.board.squares[rook_row][rook_col]

        if not king or not rook:
            return False

        if (
            king.piece_type != PieceType.KING
            or rook.piece_type != PieceType.ROOK
            or king.color != self.current_turn
            or rook.color != self.current_turn
        ):
            return False

        # Check if squares between king and rook are empty
        min_col = min(king_col, rook_col)
        max_col = max(king_col, rook_col)
        for col in range(min_col + 1, max_col):
            if self.board.squares[king_row][col] is not None:
                return False

        # Check if king is in check
        if self.board.is_square_under_attack(
            king_pos, Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        ):
            return False

        # Check if king passes through attacked squares
        direction = 1 if king_col < rook_col else -1
        for col in range(king_col + direction, rook_col + direction, direction):
            if self.board.is_square_under_attack(
                (king_row, col),
                Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE,
            ):
                return False

        return True
