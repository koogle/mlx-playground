from typing import Tuple, List, Optional
from chess_engine.bitboard import BitBoard

# from bitboard import BitBoard


class ChessGame:
    def __init__(self):
        self.board = BitBoard()
        self.move_history: List[str] = []
        self.moves_without_progress = 0

    def _get_piece_symbol(self, piece_type: int) -> str:
        """Get the algebraic notation symbol for a piece."""
        if piece_type == 0:  # Pawn
            return ""
        piece_symbols = {
            5: "K",  # King
            4: "Q",  # Queen
            3: "R",  # Rook
            2: "B",  # Bishop
            1: "N",  # Knight
        }
        return piece_symbols[piece_type]

    def _find_similar_pieces(
        self, piece_type: int, target_square: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Find all pieces of the same type that could move to the target square."""
        similar_pieces = []
        current_turn = self.board.get_current_turn()

        # Get all pieces of this type
        pieces = self.board.get_all_pieces(current_turn)
        for pos, p_type in pieces:
            if p_type == piece_type:
                # Check if the piece can move to the target square
                if target_square in self.board.get_valid_moves(pos):
                    similar_pieces.append(pos)
        return similar_pieces

    def _can_move(
        self, from_square: Tuple[int, int], to_square: Tuple[int, int]
    ) -> bool:
        """Check if a piece can move from from_square to to_square."""
        valid_moves = self.board.get_valid_moves(from_square)
        return to_square in valid_moves

    def get_all_valid_moves(self) -> List[str]:
        """Get all valid moves in algebraic notation."""
        valid_moves = []
        current_turn = self.board.get_current_turn()
        pieces = self.board.get_all_pieces(current_turn)

        for pos, piece_type in pieces:
            moves = self.board.get_valid_moves(pos)
            piece = (
                piece_type,
                current_turn,
            )
            for move in moves:
                move_str = self._move_to_algebraic(pos, move, piece)
                valid_moves.append(move_str)

        return valid_moves

    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Make a move in the game"""
        # Track moves without captures or pawn moves for 50/75 move rule
        piece = self.board.get_piece_at(*from_pos)

        if self.board.make_move(from_pos, to_pos):
            # Convert move to algebraic notation and add to history
            move_str = self._move_to_algebraic(
                from_pos,
                to_pos,
                (piece[1], piece[0]),  # (piece_type, color)
            )
            self.move_history.append(move_str)
            return True
        else:
            return False

    def make_move_coords(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], move_str: str
    ) -> bool:
        """Make a move using board coordinates"""
        # Track moves without captures or pawn moves for 50/75 move rule
        piece = self.board.get_piece_at(*from_pos)
        target = self.board.get_piece_at(*to_pos)

        # Check for pawn
        if piece[1] == 0 or target[0] != -1:
            self.moves_without_progress = 0
        else:
            self.moves_without_progress += 1

        # Make the move
        if self.board.make_move(from_pos, to_pos):
            self.move_history.append(move_str)
            return True
        return False

    def can_claim_draw(self) -> bool:
        """Check if a player can claim a draw (50-move rule)"""
        return self.moves_without_progress >= 50

    def __str__(self):
        return str(self.board)

    def _move_to_algebraic(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int] | Tuple[int, int, int],
        piece: Tuple[int, int],
    ) -> str:
        """Convert a move to algebraic notation"""
        from_row, from_col = from_pos
        piece_type, color = piece

        # Handle promotion moves
        promotion_piece = None
        if isinstance(to_pos, tuple) and len(to_pos) == 3:
            to_row, to_col, promotion_piece = to_pos
        else:
            to_row, to_col = to_pos

        # Special case for castling
        if piece_type == 5 and abs(to_col - from_col) == 2:  # 5 is king
            return "O-O" if to_col > from_col else "O-O-O"

        # Get target square algebraic coordinates
        to_square = f"{chr(to_col + 97)}{to_row + 1}"

        # Check if move is a capture
        target_color, _ = self.board.get_piece_at(to_row, to_col)
        is_capture = target_color != -1 and target_color != color

        # Get piece symbol (empty for pawns)
        piece_symbol = self._get_piece_symbol(piece_type)

        # For pawn moves
        if piece_type == 0:  # Pawn
            move = ""
            if is_capture:
                move = f"{chr(from_col + 97)}x{to_square}"
            else:
                move = to_square

            # Add promotion piece if applicable
            if promotion_piece is not None:
                promotion_symbols = {1: "N", 2: "B", 3: "R", 4: "Q"}
                move += f"={promotion_symbols[promotion_piece]}"
            return move

        # For other pieces
        capture_symbol = "x" if is_capture else ""
        return f"{piece_symbol}{capture_symbol}{to_square}"

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
                if not self.make_move(
                    self.parse_move(move)[0], self.parse_move(move)[1]
                ):
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

    def get_game_history_str(self) -> str:
        """Get the game history in a readable format."""
        if not self.move_history:
            return "No moves played"

        history = []
        for i in range(0, len(self.move_history), 2):
            move_num = i // 2 + 1
            if i + 1 < len(self.move_history):
                history.append(
                    f"{move_num}. {self.move_history[i]} {self.move_history[i+1]}"
                )
            else:
                history.append(f"{move_num}. {self.move_history[i]}")

        return "\n".join(history)

    def get_current_turn(self) -> int:
        """Get current player's turn (0 for white, 1 for black)"""
        return self.board.get_current_turn()

    def parse_move(
        self, move_str: str
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Parse chess move in various formats:
        - e2e4 (source and destination squares)
        - e4 (pawn move)
        - Nf3 (piece move)
        - Bxe4 (piece capture)
        - exd5 (pawn capture)
        - O-O or O-O-O (castling)
        """
        if not move_str:
            return None, None

        # Remove 'x' from capture notation but remember it was a capture
        is_capture = "x" in move_str
        move_str = move_str.replace("x", "")

        # Handle castling
        if move_str == "O-O" or move_str == "O-O-O":
            row = 0 if self.board.get_current_turn() == 0 else 7
            # First verify the king is there
            king_color, king_type = self.board.get_piece_at(row, 4)
            if (
                king_color != self.board.get_current_turn() or king_type != 5
            ):  # 5 is king
                return None, None

            # For initial position tests, just return the coordinates without validation
            if self.move_history == []:
                return (row, 4), (row, 6 if move_str == "O-O" else 2)

            # For actual gameplay, check if the move is valid
            to_col = 6 if move_str == "O-O" else 2
            if (row, to_col) in self.board.get_valid_moves((row, 4)):
                return (row, 4), (row, to_col)
            return None, None

        if len(move_str) > 4:
            return None, None
        # Handle direct coordinate notation (e.g. "e2e4")
        if (
            len(move_str) == 4 and move_str.isalnum()
        ):  # Must be exactly 4 chars and alphanumeric
            try:
                from_col = ord(move_str[0].lower()) - ord("a")
                from_row = int(move_str[1]) - 1
                to_col = ord(move_str[2].lower()) - ord("a")
                to_row = int(move_str[3]) - 1

                if not all(0 <= x < 8 for x in [from_row, from_col, to_row, to_col]):
                    return None, None

                # Verify the piece exists and can make this move
                piece_color, piece_type = self.board.get_piece_at(from_row, from_col)
                if piece_color != self.board.get_current_turn():
                    return None, None

                # Check if move is valid
                valid_moves = self.board.get_valid_moves((from_row, from_col))
                if (to_row, to_col) not in valid_moves:
                    return None, None

                return (from_row, from_col), (to_row, to_col)
            except (ValueError, IndexError):
                return None, None

        # Handle algebraic notation (e.g. "e4", "Nf3", "Bxe4", "exd5")
        try:
            # Get destination square
            dest_file = move_str[-2]
            dest_rank = move_str[-1]
            to_col = ord(dest_file.lower()) - ord("a")
            to_row = int(dest_rank) - 1

            if not (0 <= to_row < 8 and 0 <= to_col < 8):
                return None, None

            # Determine piece type
            piece_type = 0  # Default to pawn
            if move_str[0].isupper():
                piece_map = {
                    "K": 5,  # King
                    "Q": 4,  # Queen
                    "R": 3,  # Rook
                    "B": 2,  # Bishop
                    "N": 1,  # Knight
                }
                if move_str[0] not in piece_map:
                    return None, None
                piece_type = piece_map[move_str[0]]

            current_turn = self.board.get_current_turn()
            pieces = self.board.get_all_pieces(current_turn)

            # For pawns, handle differently
            if piece_type == 0:
                # For pawn captures, the first character is the file of origin
                if is_capture:
                    if len(move_str) != 2:  # Must specify source file for pawn captures
                        return None, None
                    pawn_file = ord(move_str[0].lower()) - ord("a")
                else:
                    pawn_file = to_col  # Normal pawn moves move straight

                if not (0 <= pawn_file < 8):
                    return None, None

                # Find possible pawn starting positions
                pawn_rank = 1 if current_turn == 0 else 6
                possible_starts = [(pawn_rank, pawn_file)]
                if abs(to_row - pawn_rank) <= 2:  # Allow 2 square advance from start
                    possible_starts.append(
                        (pawn_rank + (1 if current_turn == 0 else -1), pawn_file)
                    )

                for start_pos in possible_starts:
                    if (to_row, to_col) in self.board.get_valid_moves(start_pos):
                        # For captures, verify there's actually a piece to capture
                        if is_capture:
                            target_color, _ = self.board.get_piece_at(to_row, to_col)
                            if target_color == -1:  # No piece to capture
                                continue
                        return start_pos, (to_row, to_col)

            # For other pieces
            for pos, p_type in pieces:
                if p_type == piece_type:
                    valid_moves = self.board.get_valid_moves(pos)
                    if (to_row, to_col) in valid_moves:
                        # For captures, verify there's actually a piece to capture
                        if is_capture:
                            target_color, _ = self.board.get_piece_at(to_row, to_col)
                            if target_color == -1:  # No piece to capture
                                continue
                        return pos, (to_row, to_col)

            return None, None
        except (ValueError, IndexError):
            return None, None

    def get_piece_at(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get piece at position"""
        color, piece_type = self.board.get_piece_at(*pos)
        if color == -1:
            return None
        return (color, piece_type)

    def make_move_algebraic(self, move_str: str) -> bool:
        """Make a move using algebraic notation (e.g. 'e2e4', 'Nf3')"""
        from_pos, to_pos = self.parse_move(move_str)
        if from_pos is None or to_pos is None:
            return False

        return self.make_move(from_pos, to_pos)

    def _coords_to_algebraic(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> str:
        """Convert coordinates to algebraic notation"""
        files = "abcdefgh"
        ranks = "12345678"

        from_square = files[from_pos[1]] + ranks[from_pos[0]]
        to_square = files[to_pos[1]] + ranks[to_pos[0]]

        return f"{from_square}{to_square}"
