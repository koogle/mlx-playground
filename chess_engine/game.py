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
        return self.board.is_valid_move(from_square, to_square)

    def make_move(self, move: str) -> bool:
        """
        Make a move using standard algebraic notation (e.g., 'e4', 'Nf3', 'exd5', 'O-O')
        Returns True if the move is valid and was executed
        """

        print("checking move: ", move)
        move = move.strip()

        # Handle castling
        if move in ["O-O", "0-0"]:  # Kingside castling
            if self.board.is_in_check(self.current_turn):
                return False  # Can't castle while in check
            if self.current_turn == Color.WHITE:
                king_pos = (0, 4)
                rook_pos = (0, 7)
                new_king_pos = (0, 6)
                new_rook_pos = (0, 5)
            else:
                king_pos = (7, 4)
                rook_pos = (7, 7)
                new_king_pos = (7, 6)
                new_rook_pos = (7, 5)

            return self._handle_castling(king_pos, rook_pos, new_king_pos, new_rook_pos)

        elif move in ["O-O-O", "0-0-0"]:  # Queenside castling
            if self.current_turn == Color.WHITE:
                king_pos = (0, 4)
                rook_pos = (0, 0)
                new_king_pos = (0, 2)
                new_rook_pos = (0, 3)
            else:
                king_pos = (7, 4)
                rook_pos = (7, 0)
                new_king_pos = (7, 2)
                new_rook_pos = (7, 3)

            return self._handle_castling(king_pos, rook_pos, new_king_pos, new_rook_pos)

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

        # Validate move using board's is_valid_move method
        valid_move = False
        for from_row, from_col in similar_pieces:
            if self.board.is_valid_move((from_row, from_col), (to_row, to_col)):
                valid_move = True
                piece = self.board.squares[from_row][from_col]
                captured_piece = self.board.squares[to_row][to_col]

                # Execute move
                self.board.squares[to_row][to_col] = piece
                self.board.squares[from_row][from_col] = None

                # Handle promotion
                if promotion_piece:
                    self.board.squares[to_row][to_col] = Piece(
                        {
                            "Q": PieceType.QUEEN,
                            "R": PieceType.ROOK,
                            "B": PieceType.BISHOP,
                            "N": PieceType.KNIGHT,
                        }[promotion_piece],
                        self.current_turn,
                    )

                # Check if this move puts opponent in check/checkmate
                opponent_color = (
                    Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
                )
                is_check = self.board.is_in_check(opponent_color)
                is_checkmate = is_check and self.board.is_checkmate(opponent_color)

                # Record move in algebraic notation
                move_str = move
                if is_checkmate:
                    move_str += "#"
                elif is_check:
                    move_str += "+"
                self.move_history.append(move_str)

                # Switch turns
                self.current_turn = opponent_color
                break

        return valid_move

    def _handle_castling(
        self,
        king_pos: Tuple[int, int],
        rook_pos: Tuple[int, int],
        new_king_pos: Tuple[int, int],
        new_rook_pos: Tuple[int, int],
    ) -> bool:
        """Handle castling move validation and execution."""
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
        direction = 1 if new_king_pos[1] > king_col else -1
        for col in range(king_col + direction, new_king_pos[1] + direction, direction):
            if self.board.is_square_under_attack(
                (king_row, col),
                Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE,
            ):
                return False

        # Execute castling
        self.board.squares[king_row][king_col] = None
        self.board.squares[rook_row][rook_col] = None
        self.board.squares[new_king_pos[0]][new_king_pos[1]] = king
        self.board.squares[new_rook_pos[0]][new_rook_pos[1]] = rook

        # Record move and switch turns
        self.move_history.append("O-O" if new_king_pos[1] == 6 else "O-O-O")
        self.current_turn = (
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        )

        return True

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

        # Check castling possibilities
        if self.current_turn == Color.WHITE:
            king_pos, rook_pos = (0, 4), (0, 7)
            if self._can_castle(king_pos, rook_pos):
                valid_moves.append("O-O")
            king_pos, rook_pos = (0, 4), (0, 0)
            if self._can_castle(king_pos, rook_pos):
                valid_moves.append("O-O-O")
        else:
            king_pos, rook_pos = (7, 4), (7, 7)
            if self._can_castle(king_pos, rook_pos):
                valid_moves.append("O-O")
            king_pos, rook_pos = (7, 4), (7, 0)
            if self._can_castle(king_pos, rook_pos):
                valid_moves.append("O-O-O")

        # Check all possible moves for each piece
        for from_row in range(8):
            for from_col in range(8):
                piece = self.board.squares[from_row][from_col]
                if not piece or piece.color != self.current_turn:
                    continue

                for to_row in range(8):
                    for to_col in range(8):
                        if self.board.is_valid_move(
                            (from_row, from_col), (to_row, to_col)
                        ):
                            # Convert to algebraic notation
                            from_square = f"{chr(from_col + ord('a'))}{from_row + 1}"
                            to_square = f"{chr(to_col + ord('a'))}{to_row + 1}"

                            # Add piece symbol for non-pawns
                            if piece.piece_type != PieceType.PAWN:
                                symbol = self._get_piece_symbol(piece)
                                move = f"{symbol}{to_square}"
                            else:
                                move = to_square

                            # Check if it's a capture
                            if self.board.squares[to_row][to_col] is not None:
                                if piece.piece_type == PieceType.PAWN:
                                    move = f"{chr(from_col + ord('a'))}x{to_square}"
                                else:
                                    move = f"{symbol}x{to_square}"

                            # Try the move to see if it results in check/checkmate
                            original_target = self.board.squares[to_row][to_col]
                            self.board.squares[to_row][to_col] = piece
                            self.board.squares[from_row][from_col] = None

                            opponent_color = (
                                Color.BLACK
                                if self.current_turn == Color.WHITE
                                else Color.WHITE
                            )
                            if self.board.is_in_check(opponent_color):
                                if self.board.is_checkmate(opponent_color):
                                    move += "#"
                                else:
                                    move += "+"

                            # Undo the move
                            self.board.squares[from_row][from_col] = piece
                            self.board.squares[to_row][to_col] = original_target

                            valid_moves.append(move)

        return valid_moves

    def _can_castle(self, king_pos: Tuple[int, int], rook_pos: Tuple[int, int]) -> bool:
        """Check if castling is possible for the given king and rook positions."""
        king_row, king_col = king_pos
        rook_row, rook_col = rook_pos

        # Check if king and rook are in position
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

        # Check if squares between are empty
        min_col = min(king_col, rook_col)
        max_col = max(king_col, rook_col)
        for col in range(min_col + 1, max_col):
            if self.board.squares[king_row][col] is not None:
                return False

        # Check if king is in check or passes through check
        opponent_color = (
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        )
        direction = 1 if rook_col > king_col else -1

        if self.board.is_in_check(self.current_turn):
            return False

        for col in range(king_col, king_col + 3 * direction, direction):
            if self.board.is_square_under_attack((king_row, col), opponent_color):
                return False

        return True
