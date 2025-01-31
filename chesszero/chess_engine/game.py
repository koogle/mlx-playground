from board import Color, Piece, PieceType
from typing import Tuple, List, Optional, Set
from bitboard import BitBoard


class ChessGame:
    def __init__(self):
        self.board = BitBoard()
        self.move_history: List[str] = []
        self.moves_without_progress = 0  # Counter for 50/75 move rule
        self.DEBUG = False

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
        self, piece: Piece, from_square: Tuple[int, int], to_square: Tuple[int, int]
    ) -> bool:
        """Check if a piece can move from from_square to to_square."""
        attack_info = self.board.get_attack_info(piece.color)
        valid_moves = self.board.get_valid_moves(from_square, attack_info)
        return to_square in valid_moves

    def get_all_valid_moves(self) -> List[str]:
        """Get all valid moves in algebraic notation."""
        valid_moves = []
        pieces = (
            self.board.white_pieces
            if self.board.current_turn == Color.WHITE
            else self.board.black_pieces
        )

        # Get attack info once for all pieces
        attack_info = self.board.get_attack_info(self.board.current_turn)

        for piece, pos in pieces:
            moves = self.board.get_valid_moves(pos, attack_info)
            for move in moves:
                move_str = self._move_to_algebraic(pos, move, piece)
                valid_moves.append(move_str)

        return valid_moves

    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Make a move in the game"""
        if self.board.make_move(from_pos, to_pos):
            # Convert move to algebraic notation
            color, piece_type = self.board.get_piece_at(*to_pos)
            piece_symbol = self._get_piece_symbol(piece_type)
            move_str = f"{piece_symbol}{chr(to_pos[1] + ord('a'))}{to_pos[0] + 1}"

            # Add check/mate symbols
            enemy_color = 1 - self.board.get_current_turn()
            if self.board.is_checkmate(enemy_color):
                move_str += "#"
            elif self.board.is_in_check(enemy_color):
                move_str += "+"

            self.move_history.append(move_str)
            return True
        return False

    def make_move_coords(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], move_str: str
    ) -> bool:
        """Make a move using board coordinates"""
        # Track moves without captures or pawn moves for 50/75 move rule
        piece = self.board.squares[from_pos[0]][from_pos[1]]
        target = self.board.squares[to_pos[0]][to_pos[1]]
        if piece.piece_type == PieceType.PAWN or target:
            self.moves_without_progress = 0
        else:
            self.moves_without_progress += 1

        # Make the move
        if self.board.move_piece(from_pos, to_pos):
            self.move_history.append(move_str)
            return True
        return False

    def get_valid_moves(self, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get valid moves for a piece"""
        return self.board.get_valid_moves(pos)

    def get_game_state(self) -> str:
        """Get the current game state as a string"""
        current_turn = self.board.get_current_turn()

        if self.board.is_checkmate(current_turn):
            winner = "Black" if current_turn == 0 else "White"
            return f"Checkmate - {winner} wins"

        if self.board.is_stalemate(current_turn):
            return "Draw by stalemate"

        if self.board.is_draw():
            return "Draw by insufficient material"

        if self.moves_without_progress >= 75:
            return "Draw by 75-move rule"

        if len(self.move_history) >= 200:
            return "Draw by maximum moves"

        if self.board.is_in_check(current_turn):
            return "Check"

        return "Ongoing"

    def get_result(self) -> float:
        """Get the game result from current player's perspective"""
        if self.board.is_game_over():
            return self.board.get_game_result()
        elif self.moves_without_progress >= 75 or len(self.move_history) >= 200:
            return 0.0  # Draw by move limit or 75-move rule
        return 0.0  # Game not over

    def can_claim_draw(self) -> bool:
        """Check if a player can claim a draw (50-move rule)"""
        return self.moves_without_progress >= 50

    def __str__(self):
        return str(self.board)

    def _move_to_algebraic(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], piece: Piece
    ) -> str:
        """Convert a move to algebraic notation."""
        files = "abcdefgh"
        ranks = "12345678"

        # Special case for castling
        if piece.piece_type == PieceType.KING and abs(to_pos[1] - from_pos[1]) == 2:
            return "O-O" if to_pos[1] > from_pos[1] else "O-O-O"

        piece_symbol = ""
        if piece.piece_type != PieceType.PAWN:
            piece_symbol = piece.piece_type.name[0]
            if piece.piece_type == PieceType.KNIGHT:
                piece_symbol = "N"

        # Add capture symbol if needed
        target = self.board.squares[to_pos[0]][to_pos[1]]
        capture = ""
        if target:
            # For pawns, include the file of origin
            if piece.piece_type == PieceType.PAWN:
                capture = files[from_pos[1]] + "x"
            else:
                capture = "x"

        # Add destination square
        destination = files[to_pos[1]] + ranks[to_pos[0]]

        # Add promotion indicator
        promotion = ""
        if piece.piece_type == PieceType.PAWN:
            if (piece.color == Color.WHITE and to_pos[0] == 7) or (
                piece.color == Color.BLACK and to_pos[0] == 0
            ):
                promotion = "=Q"  # Default to queen promotion

        return f"{piece_symbol}{capture}{destination}{promotion}"

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

    def is_over(self) -> bool:
        """Check if the game is over (checkmate, stalemate, or draw)"""
        return (
            self.board.is_checkmate(self.board.current_turn)
            or self.board.is_stalemate(self.board.current_turn)
            or len(self.move_history) >= 200  # Maximum game length
            or self.moves_without_progress >= 75  # 75-move rule
            or self.board.is_draw()  # Other draw conditions
        )

    def is_draw(self) -> bool:
        """Check if the game is a draw"""
        return (
            self.board.is_draw()  # Original draw conditions (stalemate, insufficient material)
            or self.moves_without_progress >= 75  # 75-move rule
            or len(self.move_history) >= 200  # Maximum game length
        )

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
        - O-O or O-O-O (castling)
        """
        if not move_str:
            return None, None

        # Handle castling
        if move_str == "O-O":
            row = 0 if self.board.get_current_turn() == 0 else 7
            return (row, 4), (row, 6)
        elif move_str == "O-O-O":
            row = 0 if self.board.get_current_turn() == 0 else 7
            return (row, 4), (row, 2)

        # Handle direct coordinate notation (e.g. "e2e4")
        if len(move_str) == 4:
            try:
                from_col = ord(move_str[0].lower()) - ord("a")
                from_row = int(move_str[1]) - 1
                to_col = ord(move_str[2].lower()) - ord("a")
                to_row = int(move_str[3]) - 1

                if all(0 <= x < 8 for x in [from_row, from_col, to_row, to_col]):
                    return (from_row, from_col), (to_row, to_col)
            except (ValueError, IndexError):
                pass

        # Handle algebraic notation (e.g. "e4", "Nf3")
        # Get destination square
        dest_square = move_str[-2:]
        try:
            to_col = ord(dest_square[0].lower()) - ord("a")
            to_row = int(dest_square[1]) - 1
            if not (0 <= to_row < 8 and 0 <= to_col < 8):
                return None, None
            to_pos = (to_row, to_col)
        except (ValueError, IndexError):
            return None, None

        # Determine piece type
        piece_type = {
            "K": 5,  # King
            "Q": 4,  # Queen
            "R": 3,  # Rook
            "B": 2,  # Bishop
            "N": 1,  # Knight
        }.get(
            move_str[0], 0
        )  # Default to pawn

        # Find piece that can make this move
        current_turn = self.board.get_current_turn()
        pieces = self.board.get_all_pieces(current_turn)

        for pos, p_type in pieces:
            if p_type == piece_type:
                if to_pos in self.board.get_valid_moves(pos):
                    return pos, to_pos

        return None, None

    def get_piece_at(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get piece at position"""
        color, piece_type = self.board.get_piece_at(*pos)
        if color == -1:
            return None
        return (color, piece_type)

    def is_game_over(self) -> bool:
        """Check if the game is over (checkmate, stalemate, or draw)"""
        current_turn = self.board.get_current_turn()

        # Check for checkmate or stalemate
        if self.board.is_checkmate(current_turn) or self.board.is_stalemate(
            current_turn
        ):
            return True

        # Check for insufficient material
        if self.board.is_draw():
            return True

        # Check for 75-move rule
        if self.moves_without_progress >= 75:
            return True

        # Check for maximum game length
        if len(self.move_history) >= 200:
            return True

        return False

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
