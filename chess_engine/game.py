from board import Board, Color, Piece, PieceType
from typing import Tuple, List, Optional
import random


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
        try:
            # Handle special commands
            if move.lower() == "random":
                return self.make_ai_move()

            # Parse and validate the move
            from_pos, to_pos = self._parse_move(move)
            if not from_pos or not to_pos:
                print("DEBUG: Invalid move parsing")
                return False

            # Validate that current player is not in check before making move
            if self.board.is_in_check(self.current_turn):
                print(f"DEBUG: {self.current_turn} is in check!")

            # Validate move
            piece = self.board.squares[from_pos[0]][from_pos[1]]
            if not piece or piece.color != self.current_turn:
                print("DEBUG: Invalid piece or wrong color")
                return False

            # Check if move is valid
            valid_moves = self.board.get_valid_moves(from_pos)
            if to_pos not in valid_moves:
                print(
                    f"DEBUG: Move {from_pos} to {to_pos} not in valid moves: {valid_moves}"
                )
                return False

            # Execute move
            self.board.move_piece(from_pos, to_pos)

            # Validate that move didn't leave/put own king in check
            if self.board.is_in_check(self.current_turn):
                print(f"DEBUG: Move would leave/put own king in check")
                # Undo the move
                self.board.move_piece(to_pos, from_pos)
                return False

            self.move_history.append(move)
            self.current_turn = (
                Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
            )
            return True

        except (ValueError, IndexError, KeyError, AttributeError) as e:
            print(f"DEBUG: Exception in make_move: {e}")
            return False

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

        # Determine piece type
        piece_type = {
            "K": PieceType.KING,
            "Q": PieceType.QUEEN,
            "R": PieceType.ROOK,
            "B": PieceType.BISHOP,
            "N": PieceType.KNIGHT,
        }.get(move[0], PieceType.PAWN)

        # For pawn captures, use the source file
        source_file = None
        if piece_type == PieceType.PAWN and len(move) == 3 and move[0] in "abcdefgh":
            source_file = ord(move[0]) - ord("a")

        # Get all pieces of this type
        pieces = self.board.get_pieces_by_type(self.current_turn, piece_type)

        # Find piece that can make this move
        for piece, from_pos in pieces:
            if to_pos in self.board.get_valid_moves(from_pos):
                # For pawns with source file, check column matches
                if source_file is not None and from_pos[1] != source_file:
                    continue
                return from_pos, to_pos

        return None, None

    def get_current_turn(self) -> Color:
        return self.current_turn

    def get_game_state(self) -> str:
        """Get the current state of the game."""
        if self.board.is_checkmate(self.current_turn):
            winner = "Black" if self.current_turn == Color.WHITE else "White"
            return f"Checkmate! {winner} wins!"
        elif self.board.is_in_check(self.current_turn):
            return "Check!"
        elif self.board.is_draw():
            return "Draw!"
        elif self.board.is_stalemate(self.current_turn):
            return "Stalemate! Game is a draw!"
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

    def make_ai_move(self) -> bool:
        """Make a random valid move for the current player."""
        # Get all pieces of current color
        pieces = self.board.get_pieces(self.current_turn)

        # Collect all valid moves
        possible_moves = []
        for piece, from_pos in pieces:
            valid_moves = self.board.get_valid_moves(from_pos)
            for to_pos in valid_moves:
                possible_moves.append((from_pos, to_pos))

        # If no valid moves, game is over
        if not possible_moves:
            return False

        # Choose a random move
        from_pos, to_pos = random.choice(possible_moves)

        # Execute the move
        self.board.move_piece(from_pos, to_pos)

        # Update game state
        self.move_history.append(
            self._format_move(
                self.board.squares[to_pos[0]][to_pos[1]], from_pos, to_pos
            )
        )
        self.current_turn = (
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        )

        return True

    def _format_move(
        self, piece: Piece, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> str:
        """Convert a move into standard algebraic notation."""
        # Handle castling
        if piece.piece_type == PieceType.KING and abs(from_pos[1] - to_pos[1]) == 2:
            return "O-O" if to_pos[1] > from_pos[1] else "O-O-O"

        # Get basic move components
        piece_symbol = self._get_piece_symbol(piece)
        from_square = chr(from_pos[1] + ord("a")) + str(from_pos[0] + 1)
        to_square = chr(to_pos[1] + ord("a")) + str(to_pos[0] + 1)

        # Check if move is a capture
        target = self.board.squares[to_pos[0]][to_pos[1]]
        is_capture = target is not None

        # Special handling for pawns
        if piece.piece_type == PieceType.PAWN:
            if is_capture:
                return f"{chr(from_pos[1] + ord('a'))}x{to_square}"
            return to_square

        # Handle piece moves
        move = piece_symbol

        # Check if we need to disambiguate the move
        similar_pieces = self.board.get_pieces_by_type(piece.color, piece.piece_type)
        disambiguation_needed = False
        for other_piece, other_pos in similar_pieces:
            if other_pos != from_pos and to_pos in self.board.get_valid_moves(
                other_pos
            ):
                disambiguation_needed = True
                break

        if disambiguation_needed:
            move += from_square

        # Add capture symbol if needed
        if is_capture:
            move += "x"

        move += to_square

        # Add check/checkmate symbol
        opponent_color = Color.BLACK if piece.color == Color.WHITE else Color.WHITE
        if self.board.is_checkmate(opponent_color):
            move += "#"
        elif self.board.is_in_check(opponent_color):
            move += "+"

        return move

    def get_game_history_str(self) -> str:
        """Get the game history as a formatted string."""
        if not self.move_history:
            return "No moves played"

        # Format moves in pairs with move numbers
        formatted_moves = []
        for i in range(0, len(self.move_history), 2):
            move_num = i // 2 + 1
            white_move = self.move_history[i]
            black_move = (
                self.move_history[i + 1] if i + 1 < len(self.move_history) else ""
            )
            formatted_moves.append(f"{move_num}. {white_move} {black_move}")

        return "\n".join(formatted_moves)

    def load_game_history(self, history_str: str) -> bool:
        """Load and replay a game from a history string."""
        # Reset the game
        self.__init__()

        try:
            # Parse moves from history string
            moves = []
            for line in history_str.strip().split("\n"):
                if not line or line.startswith("Move history:"):
                    continue
                # Extract move number and moves
                parts = line.split(".")
                if len(parts) != 2:
                    continue
                # Split moves, handling potential single move at end
                move_parts = parts[1].strip().split()
                moves.extend(move_parts)

            # Replay moves
            print("\nReplaying game history:")
            for i, move in enumerate(moves):
                if not self.make_move(move):
                    print(f"Failed to replay move: {move}")
                    return False

                # Print board state after each move
                print(f"\nMove {i+1}: {move}")
                print(self.board)

            return True

        except Exception as e:
            print(f"Error loading game history: {e}")
            self.__init__()  # Reset on error
            return False

    def __str__(self) -> str:
        """String representation including board and game history."""
        board_str = str(self.board)
        history_str = self.get_game_history_str()
        return f"{board_str}\n\nMove history:\n{history_str}"
